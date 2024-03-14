import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple
from collections import defaultdict


class POLRS(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        init_lr: int,
        train_num: int
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.backbone.num_features, 1)
        self.init_lr = init_lr
        self.lr_n = train_num // 10
        self.backbones = [copy.deepcopy(self.backbone).cuda(), copy.deepcopy(self.backbone).cuda()]
        self.classifiers = [nn.Linear(self.backbone.num_features, 1).cuda(), nn.Linear(self.backbone.num_features, 1).cuda()]
        self.optimizers = [optim.Adam(list(self.backbones[0].parameters())+list(self.classifiers[0].parameters()), lr=self.init_lr/2), optim.Adam(list(self.backbones[1].parameters())+list(self.classifiers[1].parameters()), lr=self.init_lr*2)]
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
                
        prev_weights = [copy.deepcopy(self.classifiers[0].weight.data), copy.deepcopy(self.classifiers[1].weight.data)]
        prev_biases = [copy.deepcopy(self.classifiers[0].bias.data), copy.deepcopy(self.classifiers[1].bias.data)]
        self.classifiers = [nn.Linear(self.backbone.num_features, num_learned_class).cuda(), nn.Linear(self.backbone.num_features, num_learned_class).cuda()]
        with torch.no_grad():
            if num_learned_class > 1:
                for i in range(2):
                    self.classifiers[i].weight[:prev_weights[i].shape[0]] = prev_weights[i]
                    self.classifiers[i].bias[:prev_weights[i].shape[0]] = prev_biases[i]
        self.optimizers = [optim.Adam(list(self.backbones[0].parameters())+list(self.classifiers[0].parameters()), lr=self.init_lr/2), optim.Adam(list(self.backbones[1].parameters())+list(self.classifiers[1].parameters()), lr=self.init_lr*2)]
        
        saved_prev_weight = copy.deepcopy(self.saved_classifier.weight.data)
        saved_prev_bias = copy.deepcopy(self.saved_classifier.bias.data)
        self.saved_classifier = nn.Linear(self.saved_backbone.num_features, num_learned_class).cuda()
        with torch.no_grad():
            if num_learned_class > 1:
                self.saved_classifier.weight[:saved_prev_weight.shape[0]] = saved_prev_weight
                self.saved_classifier.bias[:saved_prev_weight.shape[0]] = saved_prev_bias


    def forward(self, batch, return_feature=False, feature=None, distill_coeff = 0.01, **kwargs):
        if self.training:
            x, y = batch
            
            for i in range(2):
                self.optimizers[i].zero_grad()
                features = self.backbones[i](x)
                preds = self.classifiers[i](features)
                loss = self.criterion(preds, y)
                loss = loss.mean()
                loss.backward()
                self.optimizers[i].step()
                
            features = self.backbone(x)
            if return_feature:
                return features.detach()

            preds = self.classifier(features)
            loss = self.criterion(preds, y)
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
 
    def lr_check(self, test_dataloader, iteration, test_freq, optimizer):
        total_num_dict = defaultdict(int)
        correct_num_dict = defaultdict(int)
        feature_vec_dict = defaultdict(list)
        
        for i in range(2):
            self.backbones[i].eval()
            self.classifiers[i].eval()
        with torch.no_grad():
            corrects = []
            tmp, tmp2 = [], []
            for batch in test_dataloader:
                x, y = batch
                x, y = x[0].cuda(), y.cuda()
                preds = self.classifier(self.backbone(x)).detach()
                corrects.append(preds.argmax(dim=1) == y)
                correct = preds.argmax(dim=1) == y
                total_num_dict[y[0].item()] += len(correct)
                correct_num_dict[y[0].item()] += torch.sum(correct).item()
                preds = self.classifiers[0](self.backbones[0](x)).detach()
                tmp.append(preds.argmax(dim=1) == y)
                preds = self.classifiers[1](self.backbones[1](x)).detach()
                tmp2.append(preds.argmax(dim=1) == y)

            acc = torch.cat(corrects, dim=0).float().mean()
            tmp = torch.cat(tmp, dim=0).float().mean()
            tmp2 = torch.cat(tmp2, dim=0).float().mean()
            if self.lr_n < iteration and ((iteration + test_freq) // self.lr_n) - iteration // self.lr_n == 1:
                max_acc = torch.max(torch.tensor([acc, tmp, tmp2]), 0)[1].item()
                if max_acc == 0:                
                    self.backbones[0] = copy.deepcopy(self.backbone).cuda()
                    self.backbones[1] = copy.deepcopy(self.backbone).cuda()
                    self.classifiers[0] = copy.deepcopy(self.classifier).cuda()
                    self.classifiers[1] = copy.deepcopy(self.classifier).cuda()
                    return acc, None, total_num_dict, correct_num_dict
                elif max_acc == 1:
                    self.backbone = copy.deepcopy(self.backbones[0]).cuda()
                    self.backbones[1] = copy.deepcopy(self.backbones[0]).cuda()
                    self.classifier = copy.deepcopy(self.classifiers[0]).cuda()
                    self.classifiers[1] = copy.deepcopy(self.classifiers[0]).cuda()
                    self.init_lr = self.init_lr / 2
                    print(f"lr change to {self.init_lr}")
                    #optimizer = optim.Adam(list(self.backbone.parameters())+list(self.classifier.parameters()), lr=self.init_lr)
                    #self.optimizers[0] = optim.Adam(list(self.backbones[0].parameters())+list(self.classifiers[0].parameters()), lr=self.init_lr / 2)
                    #self.optimizers[1] = optim.Adam(list(self.backbones[1].parameters())+list(self.classifiers[1].parameters()), lr=self.init_lr * 2)
                    for param in optimizer.param_groups[0]['params']:
                        if param in optimizer.state.keys():
                            del optimizer.state[param]
                    del optimizer.param_groups[0]
                    optimizer.add_param_group({'lr': self.init_lr, 'params': list(self.backbone.parameters())+list(self.classifier.parameters())})
                    
                    for param in self.optimizers[0].param_groups[0]['params']:
                        if param in self.optimizers[0].state.keys():
                            del self.optimizers[0].state[param]
                    del self.optimizers[0].param_groups[0]
                    self.optimizers[0].add_param_group({'lr': self.init_lr / 2, 'params': list(self.backbones[0].parameters())+list(self.classifiers[0].parameters())})
                    
                    for param in self.optimizers[1].param_groups[0]['params']:
                        if param in self.optimizers[1].state.keys():
                            del self.optimizers[1].state[param]
                    del self.optimizers[1].param_groups[0]
                    self.optimizers[1].add_param_group({'lr': self.init_lr * 2, 'params': list(self.backbones[1].parameters())+list(self.classifiers[1].parameters())})
                    '''
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.init_lr
                    for param_group in self.optimizers[0].param_groups:
                        param_group['lr'] = self.init_lr / 2
                    for param_group in self.optimizers[1].param_groups:
                        param_group['lr'] = self.init_lr * 2
                    '''
                else:
                    self.backbone = copy.deepcopy(self.backbones[1]).cuda()
                    self.backbones[0] = copy.deepcopy(self.backbones[1]).cuda()
                    self.classifier = copy.deepcopy(self.classifiers[1]).cuda()
                    self.classifiers[0] = copy.deepcopy(self.classifiers[1]).cuda()
                    self.init_lr = self.init_lr * 2
                    print(f"lr change to {self.init_lr}")
                    #optimizer = optim.Adam(list(self.backbone.parameters())+list(self.classifier.parameters()), lr=self.init_lr)
                    #self.optimizers[0] = optim.Adam(list(self.backbones[0].parameters())+list(self.classifiers[0].parameters()), lr=self.init_lr / 2)
                    #self.optimizers[1] = optim.Adam(list(self.backbones[1].parameters())+list(self.classifiers[1].parameters()), lr=self.init_lr * 2)
                    for param in optimizer.param_groups[0]['params']:
                        if param in optimizer.state.keys():
                            del optimizer.state[param]
                    del optimizer.param_groups[0]
                    optimizer.add_param_group({'lr': self.init_lr, 'params': list(self.backbone.parameters())+list(self.classifier.parameters())})
                    
                    for param in self.optimizers[0].param_groups[0]['params']:
                        if param in self.optimizers[0].state.keys():
                            del self.optimizers[0].state[param]
                    del self.optimizers[0].param_groups[0]
                    self.optimizers[0].add_param_group({'lr': self.init_lr / 2, 'params': list(self.backbones[0].parameters())+list(self.classifiers[0].parameters())})
                    
                    for param in self.optimizers[1].param_groups[0]['params']:
                        if param in self.optimizers[1].state.keys():
                            del self.optimizers[1].state[param]
                    del self.optimizers[1].param_groups[0]
                    self.optimizers[1].add_param_group({'lr': self.init_lr * 2, 'params': list(self.backbones[1].parameters())+list(self.classifiers[1].parameters())})      
                    '''
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.init_lr
                    for param_group in self.optimizers[0].param_groups:
                        param_group['lr'] = self.init_lr / 2
                    for param_group in self.optimizers[1].param_groups:
                        param_group['lr'] = self.init_lr * 2
                    '''
                return acc, optimizer, total_num_dict, correct_num_dict
        return acc, None, total_num_dict, correct_num_dict
    
