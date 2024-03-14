import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from main import parse_args
from methods.DERModel import DERLogit
from losses import SupConLoss
from utils import get_statistics, strong_aug
import copy

class XDERModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        device,
        dataset
    ):
        super().__init__()
        self.args = parse_args()
        self.logit_container = DERLogit()
        self.xder_info = XDERInfo()
        self.backbone = backbone
        self.device = device
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.simclr_temp = 5
        self.simclr_batch_size = 64
        self.alpha = 0.9
        self.beta = 0.6
        self.gamma = 0.85
        self.eta = 0.01
        self.lambd = 0.04
        self.m = 0.2
        self.simclr_num_aug = 2
        self.cur_task = 0
        if 'clear' in self.dataset:
            self.tasks = 10
        else:
            self.tasks = 5
        mean, std, num_class, inp_size, _ = get_statistics(dataset=self.dataset)
        self.cpt = int(num_class / self.tasks)
        self.classifier = nn.Linear(self.backbone.num_features, num_class)
        self.gpu_augmentation = strong_aug(inp_size, mean, std)
        
        self.update_counter = torch.zeros(self.args.memory_size).to(self.device)
        self.simclr_lss = SupConLoss(temperature=5, reduction='sum')
        
        self.saved_backbone = copy.deepcopy(self.backbone)
        self.saved_classifier = copy.deepcopy(self.classifier)
        
    def observe_novel_class(self, num_learned_class):
        pass

    def forward(self, batch, return_feature=False, feature=None, distill_coeff = 0.01, **kwargs):
        if self.training:
            stream_batch_size = kwargs['batch_sampler'].temp_batch_size
            memory = kwargs['memory']
            batch_sampler = kwargs['batch_sampler']
            
            bs = batch_sampler.temp_batch_size
            stream_idx, memory_idx = batch_sampler.return_idx()
            buf_idx = batch_sampler.return_buf()
            [x, not_aug_x], y = batch

            features = self.backbone(x)
            if return_feature:
                return features.detach()

            logit = self.classifier(features)
            loss = self.criterion(logit[:bs*2], y[:bs*2])
            self.logit_container.add_logit(stream_idx, logit[:bs])
            
            if feature is not None:
                distill_loss = ((features - feature.detach()) ** 2).sum(dim=1)
                loss += distill_coeff * distill_loss
            if len(x[bs:]) <= 0:
                loss = loss[:bs].mean()
            else:
                loss = loss[:bs].mean() + self.alpha * loss[bs:bs*2].mean()
            if len(x[bs*2:]) > 0:
                features = logit[bs*2:]
                last_features = self.logit_container.return_logit(memory_idx, len(features[0]))
                loss += self.beta * F.mse_loss(features, last_features[-features.size(0):])
            # consistency loss
            loss_cons = self.get_consistency_loss(loss, y, not_aug_x)
            loss += loss_cons
            # constraint loss
            if self.cur_task < self.tasks:
                loss_constr_past, loss_constr_futu = self.get_logit_constraint_loss(loss, logit[:bs], logit[bs:], y[bs:], len(memory)>0)       
                loss += loss_constr_futu + loss_constr_past   
            if self.cur_task > 0 and self.cur_task < self.tasks:
                self.xder_info.save(logit, bs, y, buf_idx, last_features)
                logit[:bs] = self.update_memory_logits(y[:bs], logit[:bs].detach(), logit[:bs].detach(), 0, n_tasks=self.cur_task)
                self.logit_container.add_logit(stream_idx, logit[:bs])
            return loss

        else:
            x, y = batch
            preds = self.classifier(self.backbone(x[0])).detach()
            return preds.argmax(dim=1) == y


    def before_model_update_(self, iter_num, num_iter, float_iter, batch_sampler):
        self.cur_task = int(iter_num / (self.args.samples_per_task * num_iter)) 
        if num_iter != 1 and float_iter != 1:
            for _ in range(int(1 / float_iter) - 1):
                batch_sampler.return_idx()


    def after_model_update_(self):
        if self.cur_task > 0:
            logit = self.xder_info.logit
            bs = self.xder_info.bs
            y = self.xder_info.y
            buf_idx = self.xder_info.buf_idx.to(self.device)
            last_features = self.xder_info.last_features
            
            with torch.no_grad():
                chosen = (y[bs:] // self.cpt) < self.cur_task
                self.update_counter[buf_idx[chosen]] += 1
                c = chosen.clone()
                chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1
                if chosen.any():
                    ## change
                    to_transplant = self.update_memory_logits(y[bs:], last_features, logit[bs:].detach(), self.cur_task, n_tasks=self.tasks - self.cur_task)
                    self.logit_container.update_logits(buf_idx[chosen], to_transplant[chosen])
                    self.logit_container.update_task_ids(buf_idx[chosen], self.cur_task)
                   
    @torch.no_grad()
    def after_task_(self, batchsize=512, **kwargs):
        memory = kwargs['memory']
        self.backbone.eval()
        self.classifier.eval()
        if self.cur_task > 0 and self.cur_task < self.tasks and len(memory) > 0:
            for sample in memory:
                x, y = sample
                x = x[1][0].cuda()
                y = y.cuda()
                logits = self.logit_container.return_logit(memory.dataset.indices).cuda()
                buf_idxs = torch.arange(len(x)).cuda()
                
                for i in range(-(-len(x) // batchsize)):
                    past_logit = logits[i * batchsize:min((i + 1) * batchsize, len(x))]
                    buf_idx = buf_idxs[i * batchsize:min((i + 1) * batchsize, len(x))]
                    logit = self.classifier(self.backbone(x[i * batchsize:min((i + 1) * batchsize, len(x))].to(self.device)))
                    chosen = (y[i * batchsize:min((i + 1) * batchsize, len(x))] // self.cpt) < self.cur_task
                    if chosen.any():
                        to_transplant = self.update_memory_logits(y[buf_idx][chosen], past_logit[chosen], logit[chosen], self.cur_task, self.tasks - self.cur_task)
                        self.logit_container.update_logits(buf_idx[chosen], to_transplant)
                        self.logit_container.update_task_ids(buf_idx[chosen], self.cur_task)
        
        self.update_counter = torch.zeros(self.args.memory_size).to(self.device)

    def update_memory_logits(self, gt, old, new, cur_task, n_tasks = 1):
        #transplant = new[task_mask][torch.arange(new[task_mask]), self.cur_task]
        transplant = new[:, cur_task * self.cpt : (cur_task + n_tasks) * self.cpt]
        gt_values = old[torch.arange(len(gt)), gt]
        max_values = transplant.max(1).values
        coeff = self.gamma * gt_values / max_values
        coeff = coeff.unsqueeze(1).repeat(1, self.cpt * n_tasks)
        mask = (max_values > gt_values).unsqueeze(1).repeat(1, self.cpt * n_tasks)
        transplant[mask] *= coeff[mask]
        old[:, cur_task * self.cpt:(cur_task + n_tasks) * self.cpt] = transplant
        return old

    def get_consistency_loss(self, loss, y, not_aug_inputs):
        # Consistency Loss (future heads)
        loss_cons = torch.tensor(0.)
        loss_cons = loss_cons.type(loss.dtype)
        if self.cur_task < self.tasks - 1:
            scl_labels = y[:min(self.simclr_batch_size, len(y))]
            scl_na_inputs = not_aug_inputs[:min(self.simclr_batch_size, len(y))]
            
            with torch.no_grad():
                scl_inputs = self.gpu_augmentation(scl_na_inputs.repeat_interleave(self.simclr_num_aug, 0)).to(self.device)
            #with bn_track_stats(self, False):
            scl_outputs = self.classifier(self.backbone(scl_inputs)).float()
            scl_featuresFull = scl_outputs.reshape(-1, self.simclr_num_aug, scl_outputs.shape[-1]) 

            scl_features = scl_featuresFull[:, :, (self.cur_task + 1) * self.cpt:] 
            scl_n_heads = self.tasks - self.cur_task - 1

            scl_features = torch.stack(scl_features.split(self.cpt, 2), 1) 
            loss_cons = torch.stack([self.simclr_lss(features=F.normalize(scl_features[:, h], dim=2), labels=scl_labels) for h in range(scl_n_heads)]).sum()
            loss_cons /= scl_n_heads * scl_features.shape[0]
            loss_cons *= self.lambd
        return loss_cons
    
    def get_logit_constraint_loss(self, loss_stream, outputs, buf_outputs, buf_labels, has_memory):        
        # Past Logits Constraint
        loss_constr_past = torch.tensor(0.).type(loss_stream.dtype)
        if self.cur_task > 0:
            chead = F.softmax(outputs[:, :(self.cur_task + 1) * self.cpt], 1)
            good_head = chead[:, self.cur_task * self.cpt:(self.cur_task + 1) * self.cpt]
            bad_head = chead[:, :self.cpt * self.cur_task]
            loss_constr = bad_head.max(1)[0].detach() + self.m - good_head.max(1)[0]
            mask = loss_constr > 0

            if (mask).any():
                loss_constr_past = self.eta * loss_constr[mask].mean()

        # Future Logits Constraint
        loss_constr_futu = torch.tensor(0.)
        if self.cur_task < self.tasks - 1:
            bad_head = outputs[:, (self.cur_task + 1) * self.cpt:]
            good_head = outputs[:, self.cur_task * self.cpt:(self.cur_task + 1) * self.cpt]

            if has_memory:
                buf_tlgt = buf_labels // self.cpt
                bad_head = torch.cat([bad_head, buf_outputs[:, (self.cur_task + 1) * self.cpt:]])
                good_head = torch.cat([good_head, torch.stack(buf_outputs.split(self.cpt, 1), 1)[torch.arange(len(buf_tlgt)), buf_tlgt]])

            loss_constr = bad_head.max(1)[0] + self.m - good_head.max(1)[0]
            mask = loss_constr > 0
            if (mask).any():
                loss_constr_futu = self.eta * loss_constr[mask].mean()
        
        return loss_constr_past, loss_constr_futu

    def stream_train_acc(self, batch, observed_classes=None):
        
        x, y = batch
        features = self.saved_backbone(x)
        preds = self.saved_classifier(features)
        train_acc = torch.sum(preds.argmax(dim=1).detach() == y)
        
        self.saved_backbone = copy.deepcopy(self.backbone)
        self.saved_classifier = copy.deepcopy(self.classifier)
        
        return train_acc


class XDERInfo:
    def __init__(self):
        super().__init__()
        self.logit = torch.tensor([])
        self.bs = 0
        self.y = torch.tensor([]) 
        self.buf_idx = torch.tensor([])
        self.last_features = torch.tensor([])
    
    def save(self, logit, bs, y, buf_idx, last_features):
        self.logit = logit
        self.bs = bs
        self.y = y
        self.buf_idx = buf_idx
        self.last_features = last_features
        
