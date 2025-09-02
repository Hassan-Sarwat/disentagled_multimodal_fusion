import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
from typing import Dict, Any, Optional, Union



class Scheduler:
    def __call__(self, **kwargs):
        raise NotImplemented()
    

class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.m + self.start_value

class ExponentialScheduler(LinearScheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0, base=10):
        self.base = base

        super(ExponentialScheduler, self).__init__(start_value=math.log(start_value, base),
                                                   end_value=math.log(end_value, base),
                                                   n_iterations=n_iterations,
                                                   start_iteration=start_iteration)

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return self.base ** linear_value
    


def activation_function(h, activation='exp'):
    if activation == 'softplus':
        return nn.functional.softplus(h)
    # Compute log(1e13) accurately
    h = h.clamp(-10, 10)
    log1e13 = 13 * torch.log(torch.tensor(10.0, dtype=h.dtype, device=h.device))

    # Numerator in log-space
    numerator = h + log1e13

    # Denominator in log-space using logaddexp for numerical stability
    denominator = torch.logaddexp(h, log1e13)

    # Compute the log of the function
    log_f = numerator - denominator

    # Exponentiate to get the final result
    return torch.exp(log_f)


def get_cml_fusion(all_evidences):
    fused_evidence = all_evidences.sum(dim=1)
    return fused_evidence


def get_avg_fusion(all_evidences):
    fused_evidence = all_evidences.mean(dim=1)
    return fused_evidence

### Only use disentangled modalities
def get_disentangled_fusion(all_evidences, shared_index=0):
    disentangled = all_evidences[:, [i for i in range(all_evidences.size(1)) if i != shared_index], :]
    return disentangled.sum(dim=1)

### Get sum of disentangled modalities, get average of sum and shared view
def get_joint_fusion(all_evidences, shared_index=0, shared_weight=0.5):
    shared = all_evidences[:, shared_index, :]
    disentangled = all_evidences[:, [i for i in range(all_evidences.size(1)) if i != shared_index], :]
    disentangled = disentangled.sum(dim=1)
    fused = shared_weight * shared + (1 - shared_weight) * disentangled
    return fused

def discounted_belief_fusion( all_evidences, flambda=3):
    """
    Perform belief fusion across unimodal and inter-modality experts.
    """
    num_classes = all_evidences.shape[-1]
    # Compute belief & uncertainty
    denominator = (all_evidences + 1).sum(dim=-1, keepdim=True)
    prob_tensor = (all_evidences + 1) / denominator
    belief_tensor = all_evidences / denominator
    uncertainty = num_classes / denominator
    # Compute discounting factor based on agreement
    discount = torch.ones(belief_tensor.shape[:-1]).to(belief_tensor.device)
    for i in range(len(all_evidences[0])):
        cp = torch.abs(prob_tensor[:, i].unsqueeze(1) - prob_tensor).sum(-1) / 2
        cc = ((1 - uncertainty[:, i].unsqueeze(1)) * (1 - uncertainty)).squeeze(1)
        dc = cp * cc.squeeze(-1)
        agreement = torch.prod((1 - (dc) ** flambda) ** (1 / flambda), dim=1)
        discount[:, i] *= agreement

    # Apply discounting
    discount = discount.unsqueeze(-1)
    belief_tensor *= discount
    uncertainty = uncertainty * discount + 1 - discount
    assert torch.allclose(belief_tensor.sum(dim=-1) + uncertainty.squeeze(-1), torch.ones_like(belief_tensor.sum(dim=-1))), f"{(belief_tensor.sum(dim=-1) + uncertainty.squeeze(-1) - torch.ones_like(belief_tensor.sum(dim=-1))).max()}"

    # Normalize beliefs
    discounted_evidence = num_classes * belief_tensor / (uncertainty + 1e-6)
    
    return discounted_evidence.mean(dim=1)

def noise(x, scale=0.01):
    noise = torch.randn(x.shape) * scale
    return x + noise.cuda()

def swap(x):
    mid = x.shape[0] // 2
    return torch.cat([x[mid:], x[:mid]])

def random_drop(x, drop_scale=10):
    drop_num = x.shape[0] // drop_scale
    drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
    x_aug = torch.clone(x)
    x_aug[drop_idxs] = 0.0
    return x_aug

def identity(x):
    return x

def augment_data(x_batch, noise_scale=0.01, drop_scale=10):
    v1 = x_batch
    v2 = torch.clone(v1)
    transforms = ['n', 'r', 'i']

    for i in range(x_batch.shape[0]):
        t_idxs = np.random.choice(3, 1, replace=False)
        t2 = transforms[t_idxs[0]]
        if t2 == 'n':
            v2[i] = noise(v2[i], scale=noise_scale)
        elif t2 == 'r':
            v2[i] = random_drop(v2[i], drop_scale=drop_scale)
        elif t2 == 'i':
            v2[i] = identity(v2[i])
    
    return v2

def initialize_weights(model, initialization='xavier'):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if initialization == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif initialization == 'zeros':
                nn.init.zeros_(m.weight)
            elif initialization == 'normal':
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif initialization == 'uniform':
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
            else:
                raise NotImplementedError()
    return model

