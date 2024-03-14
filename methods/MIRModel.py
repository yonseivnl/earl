import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import BatchSampler
import copy

class MIRModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.backbone.num_features, 1)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.cand_size = 50
        
        self.saved_backbone = copy.deepcopy(self.backbone)
        self.saved_classifier = copy.deepcopy(self.classifier)

    def observe_novel_class(self, num_learned_class):
        prev_weight = copy.deepcopy(self.classifier.weight.data)
        prev_bias = copy.deepcopy(self.classifier.bias.data)
        self.classifier = nn.Linear(self.backbone.num_features, num_learned_class).cuda()
        with torch.no_grad():
            if num_learned_class > 1:
                self.classifier.weight[:prev_weight.shape[0]] = prev_weight
                self.classifier.bias[:prev_weight.shape[0]] = prev_bias
        
        saved_prev_weight = copy.deepcopy(self.saved_classifier.weight.data)
        saved_prev_bias = copy.deepcopy(self.saved_classifier.bias.data)
        self.saved_classifier = nn.Linear(self.saved_backbone.num_features, num_learned_class).cuda()
        with torch.no_grad():
            if num_learned_class > 1:
                self.saved_classifier.weight[:saved_prev_weight.shape[0]] = saved_prev_weight
                self.saved_classifier.bias[:saved_prev_weight.shape[0]] = saved_prev_bias

    def forward(self, batch, return_feature=False, feature=None, distill_coeff = 0.01, **kwargs):
        if self.training:
            bs = kwargs['batch_sampler'].temp_batch_size
            str_to_mem = kwargs['batch_sampler'].str_to_mem
            _, memory_idx = kwargs['batch_sampler'].return_idx()
            memory_idx = [str_to_mem[i] for i in memory_idx]
            x, y = batch

            features = self.backbone(x[:bs])
            if return_feature:
                return features.detach()

            preds = self.classifier(features)
            loss = self.criterion(preds, y[:bs])

            if feature is not None:
                distill_loss = ((features - feature.detach()) ** 2).sum(dim=1)
                loss += distill_coeff * distill_loss
            loss = loss.mean()
            loss.backward()
            
            grads = {}
            for name, param in self.backbone.named_parameters():
                grads[name] = param.grad.data
            for name, param in self.classifier.named_parameters():
                grads[name] = param.grad.data
            
            if len(memory_idx) > 0:
                memory_idx = np.random.choice(memory_idx, size=min(len(memory_idx), self.cand_size), replace=False)
                memory_loader = torch.utils.data.DataLoader(
                    kwargs['memory'],
                    batch_sampler=BatchSampler(torch.LongTensor(memory_idx), batch_size=len(memory_idx), drop_last=False),
                    num_workers=2,
                )
                for i in memory_loader:
                    lr = kwargs['optimizer'].param_groups[0]['lr']
                    new_backbone = copy.deepcopy(self.backbone)
                    new_classifier = copy.deepcopy(self.classifier)
                    for name, param in new_backbone.named_parameters():
                        param.data = param.data - lr * grads[name]
                    for name, param in new_classifier.named_parameters():
                        param.data = param.data - lr * grads[name]
                        
                    memory_cands = i[0][0].cuda()
                    memory_cands_test = i[0][1][0].cuda()
                    memory_label = i[1].cuda()
                    with torch.no_grad():
                        logit_pre = self.classifier(self.backbone(memory_cands_test))
                        logit_post = new_classifier(new_backbone(memory_cands_test))
                        pre_loss = F.cross_entropy(logit_pre, memory_label, reduction='none')
                        post_loss = F.cross_entropy(logit_post, memory_label, reduction='none')
                        scores = post_loss - pre_loss
                    selected_samples = torch.argsort(scores, descending=True)[:len(memory_idx)]
                    mem_x = memory_cands[selected_samples]
                    mem_y = memory_label[selected_samples]
                    x = torch.cat([x[:bs], mem_x]).cuda()
                    y = torch.cat([y[:bs], mem_y]).cuda()
            
            kwargs['optimizer'].zero_grad()
            preds = self.classifier(self.backbone(x))
            loss = self.criterion(preds, y)
            loss = loss.mean()
            return loss

        else:
            x, y = batch
            preds = self.classifier(self.backbone(x[0])).detach()
            return preds.argmax(dim=1) == y
        
    def stream_train_acc(self, batch, observed_classes=None):
        
        x, y = batch
        features = self.saved_backbone(x)
        preds = self.saved_classifier(features)
        train_acc = torch.sum(preds.argmax(dim=1).detach() == y)
        
        self.saved_backbone = copy.deepcopy(self.backbone)
        self.saved_classifier = copy.deepcopy(self.classifier)
        
        return train_acc
