import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Tuple
import copy

class DERModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
    ):
        super().__init__()
        self.logit_container = DERLogit()
        self.backbone = backbone
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.backbone.num_features, 1)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.alpha = 0.5
        self.beta = 0.5
        
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

    def before_model_update_(self, iter_num, num_iter, float_iter, batch_sampler):
        if float_iter != 1:
            for _ in range(int(1 / float_iter) - 1):
                batch_sampler.return_idx()

    def forward(self, batch, return_feature=False, feature=None, distill_coeff = 0.01, **kwargs):
        if self.training:
            bs = kwargs['batch_sampler'].temp_batch_size
            stream_idx, memory_idx = kwargs['batch_sampler'].return_idx()
            x, y = batch
            
            features = self.backbone(x)
            if return_feature:
                return features.detach()

            preds = self.classifier(features)
            loss = self.criterion(preds[:bs*2], y[:bs*2])
            loss = loss[:bs].mean() + self.alpha * loss[bs:2*bs].mean()
            
            self.logit_container.add_logit(stream_idx, preds)

            if feature is not None:
                distill_loss = ((features - feature.detach()) ** 2).sum(dim=1)
                loss += distill_coeff * distill_loss

            if len(x[bs*2:]) <= 0:
                return loss
            
            last_features = self.logit_container.return_logit(memory_idx[-len(preds[2*bs:]):], len(preds[0]))
            loss += self.beta * F.mse_loss(preds[2*bs:], last_features)
            
            return loss

        else:
            x, y = batch
            preds = self.classifier(self.backbone(x[0])).detach()
            return preds.argmax(dim=1) == y

    def stream_train_acc(self, batch, observed_classes=None):
        
        x,y = batch
        features = self.saved_backbone(x)
        preds = self.saved_classifier(features)
        train_acc = torch.sum(preds.argmax(dim=1).detach() == y)
        
        self.saved_backbone = copy.deepcopy(self.backbone)
        self.saved_classifier = copy.deepcopy(self.classifier)
        
        return train_acc

class DERLogit:
    def __init__(self):
        super().__init__()
        self.logits = {}
        self.task_ids = {}
        
    def add_logit(self, idx, logit):
        for i, x in enumerate(idx):
            self.logits[x] = logit[i].detach()
            
    def return_logit(self, idx, max_len=-1):
        if max_len == -1:
            max_len = len(self.logits[idx[-1]])
        return torch.stack([torch.cat((self.logits[x], torch.tensor([0] * (max_len - len(self.logits[x]))).cuda())) for x in idx])
    
    def update_logits(self, indices, new_logits):
        for new_logit, indice in zip(new_logits, indices):
            self.logits[indice.item()] = new_logit.detach()

    def update_task_ids(self, indices, new_task_id):
        for indice in indices:
            self.task_ids[indice] = new_task_id      
