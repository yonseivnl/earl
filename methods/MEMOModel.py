import torch.nn as nn
import torch
from typing import Tuple
import copy
from main import parse_args


EPSILON = 1e-8

class MEMOModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
    ):
        super().__init__()
        self.args = parse_args()
        self.backbone = backbone
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.backbone.num_features, 1)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        
        self.AdaptiveExtractors = nn.ModuleList()
        self.TaskAgnosticExtractor = Basenet(self.backbone)
        self.new_extractor = Adaptivenet(self.backbone)
        
        self.aux_classifier = None        
        self.save_prev_class_num = False
        self.prev_num_learned_class = None

        self.AdaptiveExtractors.append(copy.deepcopy(self.new_extractor))
        print("extractor added")
        
        self.iter_num = 0
        self.n_task = 1
        self.new_task = False
        
        if "imagenet" in self.args.dataset:
            self.total_tasks = 10
        elif "clear10" in self.args.dataset:
            self.total_tasks = 10
        elif "clear100" in self.args.dataset:
            self.total_tasks = 11
        else:
            self.total_tasks = 5
            
        self.saved_TaskAgnosticExtractor = copy.deepcopy(self.TaskAgnosticExtractor)
        self.saved_AdaptiveExtractors = copy.deepcopy(self.AdaptiveExtractors)
        self.saved_classifier = copy.deepcopy(self.classifier)
        
    def observe_novel_class(self, num_learned_class):
    
        self.temp_num_learned_class = num_learned_class
        print("class added")
            
        update_aux_param = False
            
        if len(self.AdaptiveExtractors)>1: 
            if self.aux_classifier is not None:
                aux_prev_weight = copy.deepcopy(self.aux_classifier.weight.data)
                aux_prev_bias = copy.deepcopy(self.aux_classifier.bias.data)
                update_aux_param = True
            prev_weight = copy.deepcopy(self.classifier.weight.data)
            prev_bias = copy.deepcopy(self.classifier.bias.data)
            
            self.aux_classifier = nn.Linear(self.backbone.num_features, self.temp_num_learned_class-self.prev_num_learned_class+1).cuda()
            self.classifier = nn.Linear(self.out_dim, self.temp_num_learned_class).cuda()

            with torch.no_grad():
                self.classifier.weight[:prev_weight.shape[0], :prev_weight.shape[1]] = prev_weight
                self.classifier.bias[:prev_weight.shape[0]] = prev_bias
                if update_aux_param:
                    self.aux_classifier.weight[:aux_prev_weight.shape[0], :aux_prev_weight.shape[1]] = aux_prev_weight
                    self.aux_classifier.bias[:aux_prev_weight.shape[0]] = aux_prev_bias
    
        else:
            prev_weight = copy.deepcopy(self.classifier.weight.data)
            prev_bias = copy.deepcopy(self.classifier.bias.data)
            self.classifier = nn.Linear(self.backbone.num_features, num_learned_class).cuda()
            with torch.no_grad():
                if num_learned_class > 1:
                    self.classifier.weight[:prev_weight.shape[0], :self.backbone.num_features] = prev_weight
                    self.classifier.bias[:prev_weight.shape[0]] = prev_bias
        
        out_dim = len(self.AdaptiveExtractors)*self.backbone.num_features
        saved_prev_weight = copy.deepcopy(self.saved_classifier.weight.data)
        saved_prev_bias = copy.deepcopy(self.saved_classifier.bias.data)
        self.saved_classifier = nn.Linear(out_dim, num_learned_class).cuda()
        with torch.no_grad():
            if num_learned_class > 1:
                self.saved_classifier.weight[:saved_prev_weight.shape[0]] = saved_prev_weight
                self.saved_classifier.bias[:saved_prev_weight.shape[0]] = saved_prev_bias
    
        
    def forward(self, batch, **kwargs):
        
        if self.training:
            x, y = batch
        
            base_feature_map = self.TaskAgnosticExtractor(x)
            features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
            features = torch.cat(features,1)
            preds = self.classifier(features)
            
            loss_clf = self.criterion(preds, y)
            loss = loss_clf
            
            if self.aux_classifier is not None:
                aux_preds = self.aux_classifier(features[:,-self.backbone.num_features:])
                aux_y = y.clone()
                aux_y = torch.where(aux_y-self.prev_num_learned_class>0, aux_y-self.prev_num_learned_class+1, 0)
                loss_aux = self.criterion(aux_preds, aux_y)
                loss += loss_aux
                
            loss = loss.mean()
            return loss

        else:
            x, y = batch
            base_feature_map = self.TaskAgnosticExtractor(x[0])
            features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
            features = torch.cat(features,1)
            preds = self.classifier(features)
            return preds.argmax(dim=1) == y
    
    def after_task_(self, **kwargs):
        if self.n_task < self.total_tasks:
            self.new_task = True
            self.aux_classifier = None
            self.prev_num_learned_class = self.temp_num_learned_class
            self.n_task += 1
            self.AdaptiveExtractors.append(copy.deepcopy(self.new_extractor))
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

            for _, param in self.AdaptiveExtractors[-2].named_parameters():
                param.requires_grad = False

            self.out_dim = len(self.AdaptiveExtractors)*self.backbone.num_features

            prev_weight = copy.deepcopy(self.classifier.weight.data)
            prev_bias = copy.deepcopy(self.classifier.bias.data)
            self.classifier = nn.Linear(self.out_dim, self.temp_num_learned_class).cuda()
            with torch.no_grad():
                self.classifier.weight[:prev_weight.shape[0], :prev_weight.shape[1]] = prev_weight
                self.classifier.bias[:prev_weight.shape[0]] = prev_bias
    
    def stream_train_acc(self, batch, observed_classes=None):
        x, y = batch
        base_feature_map = self.saved_TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.saved_AdaptiveExtractors]
        features = torch.cat(features,1)
        preds = self.saved_classifier(features)
        train_acc = torch.sum(preds.argmax(dim=1).detach() == y)
        
        self.saved_TaskAgnosticExtractor = copy.deepcopy(self.TaskAgnosticExtractor)
        self.saved_AdaptiveExtractors = copy.deepcopy(self.AdaptiveExtractors)
        self.saved_classifier = copy.deepcopy(self.classifier)
        
        
        return train_acc
        


class Basenet(nn.Module):
    def __init__(self, net):
        super(Basenet, self).__init__()
        self.basenet = copy.deepcopy(net)
    
    def forward(self, x):
        x = self.basenet.conv1(x)
        x = self.basenet.bn1(x)
        x = self.basenet.relu(x)
        out0 = self.basenet.maxpool(x)
        out1 = self.basenet.layer1(out0)
        out2 = self.basenet.layer2(out1)
        out3 = self.basenet.layer3(out2)
        return out3
        

class Adaptivenet(nn.Module):
    def __init__(self, net):
        super(Adaptivenet, self).__init__()
        self.adaptivenet = copy.deepcopy(net)
    
    def forward(self, x):    
        out0 = self.adaptivenet.layer4(x)
        out1 = self.adaptivenet.avgpool(out0)
        feature = torch.flatten(out1, 1)
        
        return feature