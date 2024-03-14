import torch.nn as nn
import torch
from typing import Tuple
import copy

class SimpleModel(nn.Module):
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

    def forward(self, batch, return_feature=False, feature=None, distill_coeff = 0.01, stream_batch_size = None, **kwargs):
        if self.training:
            x, y = batch
            
            features = self.backbone(x)
            if return_feature:
                return features.detach()
            preds = self.classifier(features)
            loss = self.criterion(preds, y)

            if feature is not None:
                distill_loss = ((features - feature.detach()) ** 2).sum(dim=1)
                loss += distill_coeff * distill_loss
            loss = loss.mean()
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
        
        
        
        
        