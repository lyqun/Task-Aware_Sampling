import torch
import torch.nn as nn
import knn_cuda
from pointnet2 import pointnet2_utils

import utils.chamfer_distance as cd
import utils.auction_match as emd


'''
    emd, cd, nearest matching
    - source: (B, N, 3)
    - target: (B, N, 3) for emd
        (B, M, 3) for cd and nearest matching
    - hint: (B, N, 3)
'''

def _emd_loss_with_hint(source, target, hint=None):
    if hint is None:
        hint = source
    
    idx, _ = emd.auction_match(hint, target)
    matched_out = pointnet2_utils.gather_operation(target.transpose(1, 2).contiguous(), idx)
    matched_out = matched_out.transpose(1, 2).contiguous() # B, N, 3
    
    dist2 = (source - matched_out) ** 2 # B, N, 3
    dist2 = torch.mean(dist2.view(-1))
    return dist2

def _emd_loss(source, target):
    return _emd_loss_with_hint(source, target, hint=None)


def _cd_loss_with_hint(source, target, hint=None, alpha=0.5):
    if alpha > 1.0:
        alpha = 1.0
    elif alpha < 0.0:
        alpha = 0.0
    
    if hint is None:
        return _cd_loss(source, target, alpha=alpha)

    _, _, idx1, idx2 = cd.chamfer_distance(hint, target)
    
    # source to target (forward)
    matched_out1 = pointnet2_utils.gather_operation(
        target.transpose(1, 2).contiguous(), 
        idx1).transpose(1, 2).contiguous()
    dist1 = (source - matched_out1) ** 2
    dist1 = torch.sum(dist1, dim=-1)

    # target to source (backward)
    matched_out2 = pointnet2_utils.gather_operation(
        source.transpose(1, 2).contiguous(), 
        idx2).transpose(1, 2).contiguous()
    dist2 = (target - matched_out2) ** 2
    dist2 = torch.sum(dist2, dim=-1)

    return alpha * torch.mean(dist1) + (1 - alpha) * torch.mean(dist2)


def _cd_loss(source, target, alpha=0.5):
    if alpha > 1.0:
        alpha = 1.0
    elif alpha < 0.0:
        alpha = 0.0
    
    dist1, dist2, _, _ = cd.chamfer_distance(source, target)
    return alpha * torch.mean(dist1) + (1 - alpha) * torch.mean(dist2)


def _nearest_matching_loss(source, target):
    dist, _, _, _ = cd.chamfer_distance(source, target)
    return torch.mean(dist)
    # return _nearest_matching_loss_with_hint(source, target, hint=None)

def _nearest_matching_loss_with_hint(source, target, hint=None):
    if hint is None:
        return _nearest_matching_loss(source, target)
        # hint = source
    
    _, _, idx, _ = cd.chamfer_distance(hint, target)
    # idx: B, N'
    matched_out = pointnet2_utils.gather_operation(
        target.transpose(1, 2).contiguous(), 
        idx).transpose(1, 2).contiguous()
    
    dist2 = (source - matched_out) ** 2
    dist2 = torch.sum(dist2, dim=-1)
    return torch.mean(dist2)


'''
    density loss
    - source: (B, N, 3), usually indicates predicted point cloud
    - target: (B, N, 3)
'''

def _knn_dist(source, query, knn_k=8, eps=1e-12):
    knn_func = knn_cuda.KNN(k=knn_k, transpose_mode=True)

    _, knn_idx = knn_func(source, query) # B, N, k
    knn_points = pointnet2_utils.grouping_operation(
        source.transpose(1, 2).contiguous(), knn_idx.int()) # B, 3, N, k
    
    dist = knn_points - query.transpose(1, 2).unsqueeze(-1)
    dist = torch.sum(dist ** 2, dim=1) # B, N, k
    dist = torch.sqrt(dist + eps)
    
    return dist # B, N, k

def _density_loss(source, target, knn_k=8, use_l1=True):
    dist_source = _knn_dist(source=source, query=target, knn_k=knn_k)
    dist_target = _knn_dist(source=target, query=target, knn_k=knn_k)

    if use_l1:
        return torch.mean(torch.abs(dist_target - dist_source))
    else:
        diff_dist2 = (dist_target - dist_source) ** 2
        return torch.mean(diff_dist2)


'''
    repulsion loss
    - source (B, N, 3)
'''

def _repulsion_loss(source, knn_k=5, radius=0.1, h=0.03, eps=1e-12):
    knn_func = knn_cuda.KNN(k=knn_k, transpose_mode=True)

    _, idx = knn_func(source, source) # B, N, k
    idx = idx[:, :, 1:].to(torch.int32).contiguous() # remove first one

    source = source.transpose(1, 2).contiguous() # B, 3, N
    neb_points = pointnet2_utils.grouping_operation(source, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

    neb_offset = neb_points - source.unsqueeze(-1)
    dist2 = torch.sum(neb_offset ** 2, dim=1)
    dist2 = torch.max(dist2, torch.tensor(eps).cuda())
    dist = torch.sqrt(dist2)
    weight = torch.exp(- dist2 / h ** 2)

    uniform_loss = torch.mean((radius - dist) * weight)
    return uniform_loss
