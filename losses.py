import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        logits_mask[:batch_size, :batch_size] = 0
        logits_mask[batch_size:, batch_size:] = 0
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        with torch.no_grad():
            logits_mask_x = torch.ones_like(mask)
            logits_mask_x[:batch_size, batch_size:] = 0
            logits_mask_x[batch_size:, :batch_size] = 0
            exp_logits_x = torch.exp(logits) * logits_mask_x
            log_prob_x = logits - torch.log(exp_logits_x.sum(1, keepdim=True))
            mask_x = torch.zeros_like(mask)
            mask_x.diagonal().fill_(1)
            mean_log_prob_pos_x = (mask_x * log_prob_x).sum(1) / mask_x.sum(1)
            loss_x = - (self.temperature / self.base_temperature) * mean_log_prob_pos_x
            loss_x, loss_y = loss_x.view(anchor_count, batch_size).mean(1)

        return loss, loss_x, loss_y


def ortho_loss(z1, zs, norm=True, temp=0.1):
    z1 = F.normalize(z1, dim=-1)
    zs = F.normalize(zs, dim=-1)
    if norm:
        return torch.norm(torch.matmul(z1.T, zs)) # yes (type1)
    else:
        raise NotImplementedError('Please set norm=True')






def edl_loss(func, y, alpha, annealing_step, num_classes, annealing_start, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)

    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(annealing_step / annealing_start, dtype=torch.float32, device=device),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def get_dc_loss(evidences, device):
    num_views = len(evidences[0])
    batch_size, num_classes = evidences.shape[0], evidences.shape[-1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[:,v,:] + 1
        print('alpha_shape:',alpha.shape)
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum

def get_dc_loss_vectorized(evidences: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Vectorized disagreement-consistency loss over views (no permute).
    evidences: [B, V, C] non-negative evidences per view.
    Returns a scalar loss.
    """
    assert evidences.dim() == 3, "evidences must be [B, V, C]"
    B, V, C = evidences.shape
    device, dtype = evidences.device, evidences.dtype

    alpha = evidences + 1.0                       # [B, V, C]
    S = alpha.sum(dim=-1, keepdim=True)           # [B, V, 1]
    p = alpha / (S + eps)                         # [B, V, C]
    u = (C / (S + eps)).squeeze(-1)               # [B, V]

    # Pairwise L1/2 distance over classes: pd[b,i,j] = 0.5 * sum_c |p[b,i,c] - p[b,j,c]|
    pd = (p.unsqueeze(2) - p.unsqueeze(1)).abs().sum(dim=-1) * 0.5   # [B, V, V]

    # Certainty coupling: cc[b,i,j] = (1 - u[b,i]) * (1 - u[b,j])
    one_minus_u = (1.0 - u)
    cc = one_minus_u.unsqueeze(2) * one_minus_u.unsqueeze(1)         # [B, V, V]

    # Combine, average over j != i (self-term is zero anyway), then sum over i, then mean over batch
    dc = pd * cc                                                      # [B, V, V]
    dc_per_i = dc.sum(dim=2) / max(1, V - 1)                         # [B, V]
    dc_sum_batch = dc_per_i.sum(dim=1)                                # [B]
    return dc_sum_batch.mean()                                        # scalar

def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


        

class AvgTrustedLoss(nn.Module):
    def __init__(self, num_views: int, annealing_start=50, gamma=1):
        super(AvgTrustedLoss, self).__init__()
        self.num_views = num_views
        self.annealing_step = 0
        self.annealing_start = annealing_start
        self.gamma = gamma

    def forward(self, evidences, target, evidence_a, fused=1,**kwargs):

        # evidences: [B, V, C], evidence_a: [B, C], target: [B]
        B, V, C = evidences.shape
        device = evidences.device

        target_1h = F.one_hot(target, C).to(device)

        # 1) fused branch (unchanged)
        loss_fused = edl_digamma_loss(evidence_a + 1, target_1h,
                                    self.annealing_step, C, self.annealing_start, device)
        loss_fused *=  fused

        # 2) vectorized per-view branch
        alpha_flat   = (evidences + 1).reshape(B * V, C)              # [B*V, C]
        target_flat  = target_1h.repeat_interleave(V, dim=0)          # [B*V, C]

        # mean over B*V  == average over views of per-batch means
        loss_views_mean = edl_digamma_loss(alpha_flat, target_flat,
                                        self.annealing_step, C, self.annealing_start, device)

        # Reconstruct: loss_fused + sum_v edl(view_v), then average over (V+1)
        # loss_acc = (loss_fused + V * loss_views_mean) / (V + 1)
        loss_acc = loss_views_mean / V


        t = min(1.0, self.annealing_step / max(1, self.annealing_start))
        gamma_t = 0.2 * (1 - t) + self.gamma * t   # 0.2 → gamma by epoch ~annealing_start
        dc_loss = get_dc_loss_vectorized(evidences)

        loss = loss_acc  + (gamma_t * dc_loss * fused)
        return loss
    
class SingleEvidentialLoss(nn.Module):
    """
    Evidential loss for an intermediate-fusion model with **one Dirichlet head**.
    Uses the digamma EDL variant + KL annealed after `annealing_start`.
    """
    def __init__(self, annealing_start: int = 50):
        super().__init__()
        self.annealing_step  = 0      # will be incremented every training step
        self.annealing_start = annealing_start

    def forward(self, evidence, target):
        """
        evidence : (B, C)   -- raw evidence (α-1) coming from the head
        target   : (B,)     -- integer class labels 0 … C-1
        """
        num_classes = evidence.shape[-1]
        target_onehot = F.one_hot(target, num_classes)

        # α = evidence + 1  to convert to Dirichlet parameters
        alpha = evidence + 1.0
        loss = edl_digamma_loss(alpha, target_onehot, self.annealing_step, num_classes, self.annealing_start, 
                                evidence.device)
        return loss