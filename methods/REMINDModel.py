import torch.nn as nn
import torch
import copy
import faiss
import numpy as np
from main import parse_args

class REMINDModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        spatial_feat_dim: int,
        num_codebooks: int,
        codebook_size: int
    ):
        super().__init__()
        self.args = parse_args()
        self.model = backbone
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.classifier = nn.Linear(self.model.num_features, 1)
        
        self.dataset = self.args.dataset
        self.model_tofreeze = G_Model(self.model, self.dataset).cuda()
        self.model_totrain = F_Model(self.model, self.dataset).cuda()
        
        self.after_baseinit = False
        self.start_ix = 0

        self.spatial_feat_dim = spatial_feat_dim
        self.num_codebooks = num_codebooks
        nbits = int(np.log2(codebook_size))
        self.num_channels = self.model_tofreeze.out_channels
        self.pq = faiss.ProductQuantizer(self.num_channels, self.num_codebooks, nbits)

        self.stream_batch_size = self.args.batch_size//2
        
        self.saved_model_tofreeze = copy.deepcopy(self.model_tofreeze)
        self.saved_model_totrain = copy.deepcopy(self.model_totrain)
        self.saved_classifier = copy.deepcopy(self.classifier)

    def observe_novel_class(self, num_learned_class):
        self.num_learned_class = num_learned_class
                
        prev_weight = copy.deepcopy(self.classifier.weight.data)
        prev_bias = copy.deepcopy(self.classifier.bias.data)
        self.classifier = nn.Linear(self.classifier.in_features, num_learned_class).cuda()
        with torch.no_grad():
            if num_learned_class > 1:
                self.classifier.weight[:prev_weight.shape[0]] = prev_weight
                self.classifier.bias[:prev_weight.shape[0]] = prev_bias
        
        saved_prev_weight = copy.deepcopy(self.saved_classifier.weight.data)
        saved_prev_bias = copy.deepcopy(self.saved_classifier.bias.data)
        self.saved_classifier = nn.Linear(self.saved_classifier.in_features, num_learned_class).cuda()
        with torch.no_grad():
            if num_learned_class > 1:
                self.saved_classifier.weight[:saved_prev_weight.shape[0]] = saved_prev_weight
                self.saved_classifier.bias[:saved_prev_weight.shape[0]] = saved_prev_bias

    def forward(self, batch, return_feature=False, mem_features=None, mem_labels=None, **kwargs):
        if self.training:
            x, y, ids = batch
            batch_length = len(y)
            
            if not self.after_baseinit:
                feat = self.model_tofreeze(x)
                preds = self.classifier(self.model_totrain(feat))
            else:
                data_batch = self.model_tofreeze(x).detach()
                data_batch = data_batch.permute(0, 2, 3, 1)
                data_batch = data_batch.reshape(-1, self.num_channels).cpu().numpy()
                codes = torch.from_numpy(self.pq.compute_codes(data_batch)).cuda()
                codes = codes.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_codebooks)
    
                if mem_features is not None:
                    mem_features = torch.stack(mem_features).cuda()
                    codes[self.stream_batch_size:] = mem_features
                    y[self.stream_batch_size:] = torch.stack(mem_labels).cuda()

                data_codes = codes.reshape(batch_length * self.spatial_feat_dim * self.spatial_feat_dim, self.num_codebooks)
                data_batch_reconstructed = self.pq.decode(data_codes.cpu().numpy())
                data_batch_reconstructed = torch.from_numpy(data_batch_reconstructed).cuda()
                data_batch_reconstructed = data_batch_reconstructed.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_channels)
                data_batch_reconstructed = data_batch_reconstructed.permute(0, 3, 1, 2)
                preds = self.classifier(self.model_totrain(data_batch_reconstructed))
            loss = self.criterion(preds, y)
            loss = loss.mean()
            if self.after_baseinit:
                return codes.cpu(), loss
            return loss

        else:
        
            if self.after_baseinit:
                if len(batch)>2:
                    x, y, _ = batch
                    data_batch = self.model_tofreeze(x).detach()
                else:
                    x, y = batch
                    data_batch = self.model_tofreeze(x[0]).detach()
                batch_length = len(y)
                data_batch = data_batch.permute(0, 2, 3, 1)
                data_batch = data_batch.reshape(-1, self.num_channels).cpu().numpy()
                codes = torch.from_numpy(self.pq.compute_codes(data_batch)).cuda()
                codes = codes.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_codebooks)
                data_codes = codes.reshape(batch_length * self.spatial_feat_dim * self.spatial_feat_dim, self.num_codebooks)
                data_batch_reconstructed = self.pq.decode(data_codes.cpu().numpy())
                data_batch_reconstructed = torch.from_numpy(data_batch_reconstructed).cuda()
                data_batch_reconstructed = data_batch_reconstructed.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_channels)
                data_batch_reconstructed = data_batch_reconstructed.permute(0, 3, 1, 2)
                preds = self.classifier(self.model_totrain(data_batch_reconstructed))
            else:
                x, y = batch
                feature = self.model_tofreeze(x[0])
                preds = self.classifier(self.model_totrain(feature))
            return preds.argmax(dim=1) == y


    def finalize_baseinit(self, train_data_num):
        self.features_data = torch.empty((train_data_num, self.num_channels, self.spatial_feat_dim, self.spatial_feat_dim), dtype=torch.float32)
        self.labels_data = np.empty((train_data_num), dtype=int)
        self.item_ixs_data = np.empty((train_data_num), dtype=int)
        self.pq_train_num = train_data_num
        for name, param in self.model_tofreeze.named_parameters():
            param.requires_grad = False
            
    
    def train_pq(self, baseinit_batch):
        codes = None
        if self.start_ix is not self.pq_train_num:
            x, y, ids = baseinit_batch
            output = self.model_tofreeze(x[0]).detach()
            end_ix = min(self.start_ix + len(y), self.pq_train_num)
            self.features_data[self.start_ix:end_ix] = output[:end_ix-self.start_ix]
            self.labels_data[self.start_ix:end_ix] = y.cpu().numpy()[:end_ix-self.start_ix]
            self.item_ixs_data[self.start_ix:end_ix] = ids.cpu().numpy()[:end_ix-self.start_ix]
            self.start_ix = end_ix
            if end_ix == self.pq_train_num:
                train_data_base_init = self.features_data.permute(0, 2, 3, 1)
                train_data_base_init = train_data_base_init.reshape(-1, self.num_channels).cpu().numpy()
                self.pq.train(train_data_base_init)

                data_batch = self.features_data.permute(0, 2, 3, 1)
                data_batch = data_batch.reshape(-1, self.num_channels).cpu().numpy()
                codes = torch.from_numpy(self.pq.compute_codes(data_batch)).cuda()
                codes = codes.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_codebooks)
        return codes, self.labels_data, self.item_ixs_data
    
    def stream_train_acc(self, batch, observed_classes=None):
        
        x, y = batch
        feature = self.saved_model_tofreeze(x)
        preds = self.saved_classifier(self.saved_model_totrain(feature))
        train_acc = torch.sum(preds.argmax(dim=1).detach() == y)
        
        self.saved_model_tofreeze = copy.deepcopy(self.model_tofreeze)
        self.saved_model_totrain = copy.deepcopy(self.model_totrain)
        self.saved_classifier = copy.deepcopy(self.classifier)
        
        return train_acc

class G_Model(nn.Module):
    def __init__(self, net, dataset):
        super(G_Model, self).__init__()
        self.model_G = copy.deepcopy(net)
        if dataset == "imagenet":
            self.out_channels = self.model_G.layer4[-1].conv2.out_channels
            self.imagenet = True
        else:
            self.out_channels = self.model_G.layer3[-1].conv2.out_channels
            self.imagenet = False
        
    
    def forward(self, x):
        x = self.model_G.conv1(x)
        x = self.model_G.bn1(x)
        x = self.model_G.relu(x)
        out0 = self.model_G.maxpool(x)
        out1 = self.model_G.layer1(out0)
        out2 = self.model_G.layer2(out1)
        out3 = self.model_G.layer3(out2)
        if self.imagenet:
            out3 = self.model_G.layer4[0](out3)
        return out3
        

class F_Model(nn.Module):
    def __init__(self, net, dataset):
        super(F_Model, self).__init__()
        self.model_F = copy.deepcopy(net)
        if dataset == "imagenet":
            self.imagenet = True
        else:
            self.imagenet = False
        
    def forward(self, x):
        # x = self.model_F.layer3(x)
        if self.imagenet:
            out0 = self.model_F.layer4[1](x)
        else:
            out0 = self.model_F.layer4(x)
        out1 = self.model_F.avgpool(out0)
        feature = torch.flatten(out1, 1)
        return feature
