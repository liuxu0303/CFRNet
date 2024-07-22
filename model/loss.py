import torch.nn.functional as F
import torch

def scale_invariant_loss(y_input, y_target, weight = 1.0, n_lambda = 1.0):
    mask = y_target > 0
    y_input  = y_input[mask]
    y_target = y_target[mask]
    log_diff = y_input - y_target

    return weight * (((log_diff**2).mean()-(n_lambda*(log_diff.mean())**2))**0.5)

def scale_invariant_log_loss(depth, gt, n_lambda=1.0, min_depth=2.0, max_depth=80.0):
    # mask = (gt > min_depth) & (gt < max_depth)  # for log
    mask = gt > min_depth
    depth = depth[mask]
    gt = gt[mask]
    log_diff = torch.log(depth) - torch.log(gt)
    return (log_diff ** 2).mean() - (n_lambda * (log_diff.mean()) ** 2)

def l1_norm_loss(y_input, y_target):
    mask = torch.nonzero(y_target, as_tuple=True)
    return F.l1_loss(y_input[mask], y_target[mask])

