import torch.nn as nn
import torch
from typing import Tuple
import copy
import logging
logger = logging.getLogger()

class EWCModel(nn.Module):
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
        self.backbone.fc = nn.Linear(self.backbone.num_features, 1)
        self.device = torch.device("cuda")

        self.regularization_terms = {}
        self.parameters_dict = {
            n: p for n, p in list(self.backbone.named_parameters())[:-2] if p.requires_grad
        }
        self.epoch_score = {}
        self.epoch_fisher = {}
        for n, p in self.parameters_dict.items():
            self.epoch_score[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized
            self.epoch_fisher[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized
        self.alpha = 0.5
        self.task_count = 0
        self.score = []
        self.fisher = []
        
        self.reg_coef = 100
        self.online_reg = True
        
        self.saved_backbone = copy.deepcopy(self.backbone)
    
    def regularization_loss(self):
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            for _, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term["importance"]
                task_param = reg_term["task_param"]

                for n, p in self.parameters_dict.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                max_importance = 0
                max_param_change = 0
                for n, p in self.parameters_dict.items():
                    max_importance = max(max_importance, importance[n].max())
                    max_param_change = max(
                        max_param_change, ((p - task_param[n]) ** 2).max()
                    )
                if reg_loss > 1000:
                    logger.warning(
                        f"max_importance:{max_importance}, max_param_change:{max_param_change}"
                    )
                reg_loss += task_reg_loss
            reg_loss = self.reg_coef * reg_loss

        return reg_loss

    def observe_novel_class(self, num_learned_class):
        prev_weight = copy.deepcopy(self.backbone.fc.weight.data)
        prev_bias = copy.deepcopy(self.backbone.fc.bias.data)
        self.backbone.fc = nn.Linear(self.backbone.num_features, num_learned_class).cuda()
        with torch.no_grad():
            if num_learned_class > 1:
                self.backbone.fc.weight[:prev_weight.shape[0]] = prev_weight
                self.backbone.fc.bias[:prev_weight.shape[0]] = prev_bias
        
        saved_prev_weight = copy.deepcopy(self.saved_backbone.fc.weight.data)
        saved_prev_bias = copy.deepcopy(self.saved_backbone.fc.bias.data)
        self.saved_backbone.fc = nn.Linear(self.saved_backbone.num_features, num_learned_class).cuda()
        with torch.no_grad():
            if num_learned_class > 1:
                self.saved_backbone.fc.weight[:saved_prev_weight.shape[0]] = saved_prev_weight
                self.saved_backbone.fc.bias[:saved_prev_weight.shape[0]] = saved_prev_bias


    def forward(self, batch, return_feature=False, feature=None, distill_coeff = 0.01, **kwargs):
        if self.training:
            x, y = batch
            self.old_params = {n: p.clone().detach() for n, p in self.parameters_dict.items()}
            self.old_grads = {n: p.grad.clone().detach() for n, p in self.parameters_dict.items() if p.grad is not None}

            preds = self.backbone(x)
            loss = self.criterion(preds, y)
            loss = loss.mean()
                
            with torch.cuda.amp.autocast(True):
                reg_loss = self.regularization_loss()
                loss += reg_loss
            
            return loss

        else:
            x, y = batch
            preds = self.backbone(x[0]).detach()
            return preds.argmax(dim=1) == y

    def update_fisher_and_score(self, new_params, old_params, new_grads, old_grads, epsilon=0.001):
        for n, _ in self.parameters_dict.items():
            if n in old_grads:
                new_p = new_params[n]
                old_p = old_params[n]
                new_grad = new_grads[n]
                old_grad = old_grads[n]
                
                if torch.isinf(new_p).sum()+torch.isinf(old_p).sum()+torch.isinf(new_grad).sum()+torch.isinf(old_grad).sum():
                    continue
                if torch.isnan(new_p).sum()+torch.isnan(old_p).sum()+torch.isnan(new_grad).sum()+torch.isnan(old_grad).sum():
                    continue
                self.epoch_score[n] += (old_grad-new_grad) * (new_p - old_p) / (
                    0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon
                )
                
                if self.epoch_score[n].max() > 1000:
                    logger.debug(
                        "Too large score {} / {}".format(
                            (old_grad-new_grad) * (new_p - old_p),
                            0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon,
                        )
                    )
                
                if (self.epoch_fisher[n] == 0).all():  # First time
                    self.epoch_fisher[n] = new_grad ** 2
                else:
                    self.epoch_fisher[n] = (1 - self.alpha) * self.epoch_fisher[
                        n
                    ] + self.alpha * new_grad ** 2
        
    @torch.no_grad()
    def calculate_importance(self):
        importance = {}
        self.fisher.append(self.epoch_fisher)
        if self.task_count == 0:
            self.score.append(self.epoch_score)
        else:
            score = {}
            for n, p in self.parameters_dict.items():
                score[n] = 0.5 * self.score[-1][n] + 0.5 * self.epoch_score[n]
            self.score.append(score)

        for n, p in self.parameters_dict.items():
            importance[n] = self.fisher[-1][n]
            self.epoch_score[n] = self.parameters_dict[n].clone().detach().fill_(0)
        return importance
    
    @torch.no_grad()
    def after_task_(self, **kwargs):
        # 2.Backup the weight of current task
        
        task_param = {}
        for n, p in self.parameters_dict.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance()

        # Save the weight and importance of weights of current task
        self.task_count += 1

        # Use a new slot to store the task-specific information
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {
                "importance": importance,
                "task_param": task_param,
            }
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {
                "importance": importance,
                "task_param": task_param,
            }
        logger.debug(f"# of reg_terms: {len(self.regularization_terms)}")
        
    def after_model_update_(self):
        new_params = {n: p.clone().detach() for n, p in self.parameters_dict.items()}
        new_grads = {
            n: p.grad.clone().detach() for n, p in self.parameters_dict.items() if p.grad is not None
        }
        self.update_fisher_and_score(new_params, self.old_params, new_grads, self.old_grads)
        
    def stream_train_acc(self, batch, observed_classes=None):
        x, y = batch
        preds = self.saved_backbone(x)
        train_acc = torch.sum(preds.argmax(dim=1).detach() == y)
        
        self.saved_backbone = copy.deepcopy(self.backbone)
        
        return train_acc