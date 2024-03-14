import torch.nn as nn
from collections import defaultdict
from losses import DotRegressionLoss, DotRegressionReverseLoss
from typing import Tuple
import torch
import copy
from utils import etf_initialize, dot_regression_accuracy, dynamic_etf_initialize
from batch_cka import linear_CKA
import torch.nn.functional as F
from ignite.utils import convert_tensor
from torch.nn.functional import log_softmax
import numpy as np

torch.manual_seed(10)

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
        print("in_dim", self.in_dim, "hidden_dim", self.hidden_dim, "out_dim", self.out_dim)

        self.shortcut = nn.Linear(in_channels, out_channels, bias=False)
        self.residual = MLP(in_channels, hidden_channels, out_channels, num_layers=3)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = self.shortcut(x) + self.residual(x)
        return x

class ETFModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        residual_addition: bool = False,
        residual_num: int = 50,
        knn_top_k: int = 25
    ):
        super().__init__()
        self.backbone = backbone
        print("backbone", self.backbone.neck)
        if self.backbone.neck == "default":
            self.neck = MLPFFNNeck(in_channels=self.backbone.num_features, out_channels=self.backbone.num_features*2, hidden_channels=self.backbone.num_features*4) # 512 x 2048 x 1024
        elif self.backbone.neck == "larger":
            self.neck = MLPFFNNeck(in_channels=self.backbone.num_features, out_channels=self.backbone.num_features*4, hidden_channels=self.backbone.num_features*8) # 512 x 4096 x 2048
    
        self.num_classes = num_classes

        self.realtime_backbone = copy.deepcopy(self.backbone)
        self.realtime_neck = copy.deepcopy(self.neck)
        self.realtime_classifier = nn.Linear(self.neck.out_dim, self.num_classes).cuda()
        self.register_buffer("etf_classifier", etf_initialize(self.neck.out_dim, self.num_classes))
        self.criterion = DotRegressionLoss(reduction="none")
        self.reg_criterion = DotRegressionReverseLoss()
        self.image_dict = defaultdict(list)
        self.feature_dict = defaultdict(list)
        self.residual_dict = defaultdict(list)
        self.softmax = nn.Softmax(dim=1)
        self.cls_features = []
        self.cls_feature_labels = []
        self.residual_addition = residual_addition
        self.num_feature_residual_pair = residual_num
        self.knn_top_k = knn_top_k
        self.knn_sigma = 0.9
        self.temp_teacher = 0.1
        self.temp_student = 0.3
        self.residual_index_dict = defaultdict(list)
        self.alpha_k = torch.ones(1).cuda()
        self.beta_k = torch.zeros(self.neck.out_dim).cuda()
        self.cls_mean_test_transform = defaultdict(list)
        self.selected_classifier = []
        self.candidate_classifier = set()
        self.observed_novel_class = False
        self.saved_neck = copy.deepcopy(self.neck)
        self.saved_backbone = copy.deepcopy(self.backbone)

    def reset_buffer(self):
        self.cls_features = []
        self.cls_feature_labels = []

    def interpret_cka(self, y, feature_vecs, device):
        cka_mat = linear_CKA(feature_vecs.unsqueeze(dim=2), self.etf_classifier[:, torch.arange(y+1)].T.unsqueeze(dim=2), device)
        return cka_mat

    def update_residual(self, memory_loader_test, device):
        # Only remain test transform (feature, residual) pairs
        mean_dict = defaultdict()
        std_dict = defaultdict()
        for batch in memory_loader_test:
            x, y = convert_tensor(batch, device=device, non_blocking=True)
            x = x[0]
            preds = self.pre_logits(self.etf_transform(self.neck(self.backbone(x))))
            target = self.etf_classifier[:, y].t()
            residuals = target - preds.detach()
            unique_labels = torch.unique(y[:len(x)])
            for unique_label in unique_labels:
                residual_label_indices = y == unique_label
                self.feature_dict[unique_label.item()].extend(preds.detach()[residual_label_indices].cpu())
                self.residual_dict[unique_label.item()].extend(residuals[residual_label_indices].cpu())
                self.feature_dict[unique_label.item()] = self.feature_dict[unique_label.item()][-self.num_feature_residual_pair:]
                self.residual_dict[unique_label.item()] = self.residual_dict[unique_label.item()][-self.num_feature_residual_pair:]
                
        for key in list(self.feature_dict.keys()):   
            mean_dict[key] = torch.mean(torch.stack(self.feature_dict[key]), dim=0)
            std_dict[key] = torch.std(torch.stack(self.feature_dict[key]), dim=0)

        whole_mean = torch.stack(list(mean_dict.values()))
        whole_std = torch.stack(list(std_dict.values()))
        
        for key in list(self.feature_dict.keys()):
            stacked = torch.stack(self.feature_dict[key])
            z_values = torch.mean((stacked.unsqueeze(1) - whole_mean) / whole_std, dim=2)
            preds = torch.argmax(z_values, dim=1)
            

    def etf_transform(self, features):
        return self.alpha_k * features + self.beta_k

    def pre_logits(self, x):
        return F.normalize(x, dim=1)


    def observe_novel_class(self, num_learned_class):
        self.num_learned_class = num_learned_class
        self.observed_novel_class = True


    def get_grad(self, preds, target):
        dot = torch.sum(preds * target, dim=1)
        return target * torch.mean((dot - (torch.ones_like(dot))) ** 2)
    
    def stream_train_acc(self, batch, observed_classes=None):
    
        x, y = batch
        features = self.pre_logits(self.etf_transform(self.saved_neck(self.saved_backbone(x))))

        cls_score = features.detach() @ self.etf_classifier
        train_acc = torch.sum(dot_regression_accuracy(cls_score[:, self.observed_classes], y))
        
        self.saved_neck = copy.deepcopy(self.neck)
        self.saved_backbone = copy.deepcopy(self.backbone)
        
        return train_acc

    def forward(self, batch, pseudo_loader=None, pseudo_cls_interval=None, device=None, return_feature=False, feature=None, distill_coeff=0.001, use_adaptive=False, teacher_batch=None, memory_loader=None, **kwargs):
        observed_classes = kwargs.get("observed_classes", kwargs["observed_classes"])

        if self.training:
            self.observed_classes = observed_classes.cpu().numpy()
            if teacher_batch is None:
                x, y = batch
                if pseudo_loader is not None:
                    # pseudo rotation 6
                    for pseudo_batch in pseudo_loader:
                        pseudo_x, pseudo_y = convert_tensor(pseudo_batch, device=device, non_blocking=True)
                        x = torch.cat([x, pseudo_x[0], pseudo_x[1], pseudo_x[2]])
                        y = torch.cat([y, (pseudo_y*3+301), (pseudo_y*3+302), (pseudo_y*3+303)])
            else:
                x, y = teacher_batch
            features = self.backbone(x)
            pre_preds = self.neck(features)
            preds = self.pre_logits(self.etf_transform(pre_preds))
            target = self.etf_classifier[:, y].t()
            #target = self.etf_classifier[:, torch.Tensor(self.selected_classifier + list(self.candidate_classifier)).long()[y]].t()

            if return_feature:
                return (x,y), preds.detach() #cls_score.detach()

            original_num = torch.sum(y < len(observed_classes))
            loss = self.criterion(preds, target)
            
            if feature is not None:
                grad = self.get_grad(preds.detach(), target)
                feature = feature[:original_num]
                current_feature = preds[:original_num]
                feature_sim_matrix = torch.matmul(feature, feature.T)
                features_sim_matrix = torch.matmul(current_feature, current_feature.T)
                distill_loss = ((feature_sim_matrix - features_sim_matrix) ** 2).sum(dim=1)
                
                if use_adaptive:
                    beta = torch.sqrt((grad.detach() ** 2).sum(dim=1) / (distill_loss.detach() * 4 + 1e-8)).mean()
                    loss += beta * distill_coeff * distill_loss
                else:
                    loss[:original_num] += distill_coeff * distill_loss
            else:
                loss = loss.mean()
            return loss

        else:
            x, y = batch
            features = self.pre_logits(self.etf_transform(self.neck(self.backbone(x[0]))))

            cls_score = features.detach() @ self.etf_classifier
            dot_regression_accuracy(cls_score[:, observed_classes], y)
            
            if self.residual_addition:
                # residual pre-processing
                residual_list = torch.stack(sum([v for v in self.residual_dict.values()], [])).cuda()
                feature_list = torch.stack(sum([v for v in self.feature_dict.values()], [])).cuda()
                norm_feature_value = torch.norm(feature_list, p=2, dim=0, keepdim=True)
                feature_list /= norm_feature_value

                # residual addition
                w_i_lists = -torch.norm(features.view(-1, 1, features.shape[1]) - feature_list, p=2, dim=2)
                w_i_indexs = torch.topk(w_i_lists, self.knn_top_k)[1].long()
                idx1, _ = torch.meshgrid(torch.arange(w_i_indexs.shape[0]), torch.arange(w_i_indexs.shape[1]))
                w_i_lists = self.softmax(w_i_lists[idx1, w_i_indexs] / self.knn_sigma)
                residual_lists = residual_list[w_i_indexs]
                residual_terms = torch.bmm(w_i_lists.unsqueeze(1), residual_lists).squeeze()
                features += residual_terms
            
            cls_score = features.detach() @ self.etf_classifier
            resmem_correct = dot_regression_accuracy(cls_score[:, observed_classes], y)
            
            if return_feature:
                return resmem_correct, features.detach()
            else:
                return resmem_correct

    def realtime_model_eval(self):
        self.realtime_backbone = copy.deepcopy(self.backbone)
        self.realtime_neck = copy.deepcopy(self.neck)
        self.realtime_classifier = nn.Linear(self.neck.out_dim, self.num_classes).cuda()
