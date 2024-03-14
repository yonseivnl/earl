import torch
import torch.nn as nn
import torch.nn.functional as F

class DotRegressionReverseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                feat,
                target,
                num_classes
                ):
        print("feat shape", feat.shape)
        print("target shape", target.shape)
        dot = torch.sum(feat @ target, dim=1)
        base = torch.ones_like(dot)
        loss = 0.5 *((dot + (1/num_classes-1) * base) ** 2)
        return loss


class DotRegressionLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(self,
                feat,
                target,
                pure_num=None,
                augmented_num=None,
                h_norm2=None,
                m_norm2=None,
                avg_factor=None,
                ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        if self.reduction == "mean":
            if augmented_num is None:
                loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)
            else:
                loss = ((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2
                loss = 0.5 * ((torch.mean(loss[:pure_num]) + torch.mean(loss[pure_num:])) / 2)

        elif self.reduction == "none":
            loss = 0.5 * (((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight

    
class CollapseLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self,
                feature_clusters
                ):
        feature_clusters = F.normalize(feature_clusters, dim=1)
        dot_matrix = (torch.matmul(feature_clusters, feature_clusters.T) + (1 / (self.num_classes - 1))) ** 2
        return torch.diagonal(dot_matrix, offset = -1)
    
    
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean(0)

        return loss.mean() if self.reduction == 'mean' else loss.sum()
