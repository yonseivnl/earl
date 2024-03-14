import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from losses import SupConLoss
from kornia.augmentation import RandomHorizontalFlip, ColorJitter, RandomGrayscale
import copy


class SCRModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.criterion = SupConLoss()
        self.transform = nn.Sequential(
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        )
        self.exemplar_means = {}
        self.saved_backbone = copy.deepcopy(self.backbone)
        
    def observe_novel_class(self, num_learned_class):
        pass
    
    def update_residual(self, memory_loader, device, **kwargs):
        self.exemplar_means = {}
        cls_exemplar = {cls: [] for cls in np.unique(memory_loader.dataset.labels)}
        
        for x, y in memory_loader:
            for i, j in zip(x[0], y):
                cls_exemplar[j.item()].append(i)
        for cls, exemplar in cls_exemplar.items():
            features = []
            for ex in exemplar:     
                ex = ex.to(device)
                feature = F.normalize(self.backbone(ex.unsqueeze(0)))
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm() 
                features.append(feature.detach())
            if len(features) == 0:
                mu_y = torch.normal(0, 1, size=(self.backbone.num_features,)).to(device)
            else:
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()
            self.exemplar_means[cls] = mu_y
    
    def forward(self, batch, return_feature=False, feature=None, distill_coeff = 0.01, **kwargs):
        if self.training:
            x, y = batch
            aug_x = self.transform(x)

            features = torch.cat([F.normalize(self.backbone(x)).unsqueeze(1), F.normalize(self.backbone(aug_x)).unsqueeze(1)], dim=1)

            if return_feature:
                return features.detach()
            
            loss = self.criterion(features, y)

            if feature is not None:
                distill_loss = ((features - feature.detach()) ** 2).sum(dim=1)
                loss += distill_coeff * distill_loss
            return loss

        else:   
            x, y = batch
            feature = F.normalize(self.backbone(x[0]), dim=1).unsqueeze(2)   
            for i in kwargs['observed_classes'] - self.exemplar_means.keys():
                mu_y = torch.normal(0, 1, size=(self.backbone.num_features,)).to("cuda")
                mu_y.data = mu_y.data / mu_y.data.norm()    
                self.exemplar_means[i] = mu_y
            means = torch.stack([self.exemplar_means[cls] for cls in kwargs['observed_classes']])
            
            means = torch.stack([means] * x[0].size(0))
            means = means.transpose(1, 2)
            feature = feature.expand_as(means)
            dists = (feature - means).pow(2).sum(1)
            _, pred_label = dists.min(1)            
            correct = torch.tensor(np.array(kwargs['observed_classes'])[
                                pred_label.tolist()] == y.cpu().numpy())
            
            return correct
    
    def stream_train_acc(self, batch, observed_classes=None):


        x, y = batch
        feature = F.normalize(self.saved_backbone(x), dim=1).unsqueeze(2)   
        for i in observed_classes - self.exemplar_means.keys():
            mu_y = torch.normal(0, 1, size=(self.saved_backbone.num_features,)).to("cuda")
            mu_y.data = mu_y.data / mu_y.data.norm()    
            self.exemplar_means[i] = mu_y
        means = torch.stack([self.exemplar_means[cls] for cls in observed_classes])
        
        means = torch.stack([means] * x.size(0))
        means = means.transpose(1, 2)
        feature = feature.expand_as(means)
        dists = (feature - means).pow(2).sum(1)
        _, pred_label = dists.min(1)            
        correct = torch.tensor(np.array(observed_classes)[
                            pred_label.tolist()] == y.cpu().numpy())
        train_acc = torch.sum(correct)        
        self.saved_backbone = copy.deepcopy(self.backbone)

        return train_acc
