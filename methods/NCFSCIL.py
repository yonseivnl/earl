import torch.nn as nn
from losses import DotRegressionLoss
import torch
from utils import etf_initialize, dot_regression_accuracy
import torch.nn.functional as F
import numpy as np
from main import parse_args
import copy
import faiss


class MLP(nn.Sequential):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 norm_layer = lambda dim: nn.LayerNorm(dim, eps=1e-6),
                 act_layer = lambda: nn.LeakyReLU(0.1)):
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(norm_layer(hidden_dim))
            layers.append(act_layer())
        layers.append(nn.Linear(in_dim if num_layers == 1 else hidden_dim, out_dim))
        super().__init__(*layers)


class MLPFFNNeck(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, hidden_channels=1024):
        super().__init__()
        self.in_dim = in_channels
        self.hidden_dim = hidden_channels
        self.out_dim = out_channels
        self.shortcut = nn.Linear(in_channels, out_channels, bias=False)
        self.residual = MLP(in_channels, hidden_channels, out_channels, num_layers=3)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = self.shortcut(x) + self.residual(x)
        return x
        

class NCFSCIL(nn.Module):
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
        self.backbone = backbone
        self.neck = MLPFFNNeck(in_channels=self.backbone.num_features, out_channels=self.backbone.num_features, hidden_channels=self.backbone.num_features*2)
        self.num_classes = num_classes
        self.register_buffer("etf_classifier", etf_initialize(self.neck.out_dim, self.num_classes))
        self.criterion = DotRegressionLoss(reduction="none")
        
        self.model_tofreeze = G_Model(self.backbone).cuda()
        self.model_totrain = F_Model(self.backbone).cuda()
        
        self.after_baseinit = False
        self.start_ix = 0

        self.spatial_feat_dim = spatial_feat_dim
        self.num_codebooks = num_codebooks
        nbits = int(np.log2(codebook_size))
        self.num_channels = self.model_tofreeze.out_channels
        self.pq = faiss.ProductQuantizer(self.num_channels, self.num_codebooks, nbits)

        self.stream_batch_size = self.args.batch_size//2
        
    def pre_logits(self, x):
        return F.normalize(x, dim=1)

    def observe_novel_class(self, num_learned_class):
        self.num_learned_class = num_learned_class

    def forward(self, batch, return_feature=False, mem_features=None, mem_labels=None, **kwargs):
        observed_classes = kwargs.get("observed_classes", kwargs["observed_classes"])
        if self.training:
            x, y, _ = batch
            batch_length = len(y)
            if not self.after_baseinit:
                features = self.model_totrain(self.model_tofreeze(x))
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
                features = self.model_totrain(data_batch_reconstructed)

            preds = self.pre_logits(self.neck(features))
            target = self.etf_classifier[:, y].t()
            cls_score = preds @ self.etf_classifier
            
            loss = self.criterion(preds, target)

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
                data_batch_reconstructed = torch.from_numpy(self.pq.decode(data_codes.cpu().numpy())).cuda()
                data_batch_reconstructed = data_batch_reconstructed.reshape(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_channels)
                data_batch_reconstructed = data_batch_reconstructed.permute(0, 3, 1, 2)
                features = self.model_totrain(data_batch_reconstructed)
            else:
                x, y = batch
                features = self.model_totrain(self.model_tofreeze(x[0]))

            preds = self.pre_logits(self.neck(features))
            cls_score = preds.detach() @ self.etf_classifier
            resmem_correct = dot_regression_accuracy(cls_score[:, observed_classes], y)

            return resmem_correct

                
    def finalize_baseinit(self, train_data_num):
        self.features_data = torch.empty((train_data_num, self.num_channels, self.spatial_feat_dim, self.spatial_feat_dim), dtype=torch.float32)
        self.labels_data = np.empty((train_data_num), dtype=int)
        self.item_ixs_data = np.empty((train_data_num), dtype=int)
        self.pq_train_num = train_data_num
        for name, param in self.model_tofreeze.named_parameters():
            param.requires_grad = False
        print("Train PQ")
    
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
                print("Finish Training PQ")
        return codes, self.labels_data, self.item_ixs_data
        
        
class G_Model(nn.Module):
    def __init__(self, net):
        super(G_Model, self).__init__()
        self.model_G = copy.deepcopy(net)
        self.out_channels = self.model_G.layer3[-1].conv2.out_channels
    
    def forward(self, x):
        x = self.model_G.conv1(x)
        x = self.model_G.bn1(x)
        x = self.model_G.relu(x)
        out0 = self.model_G.maxpool(x)
        out1 = self.model_G.layer1(out0)
        out2 = self.model_G.layer2(out1)
        out3 = self.model_G.layer3(out2)
        # out = self.model_G.layer4(out3)
        return out3
        

class F_Model(nn.Module):
    def __init__(self, net):
        super(F_Model, self).__init__()
        self.model_F = copy.deepcopy(net)
    
    def forward(self, x):
        # x = self.model_F.layer3(x)
        out4 = self.model_F.layer4(x)
        out5 = self.model_F.avgpool(out4)
        feature = torch.flatten(out5, 1)
        return feature

