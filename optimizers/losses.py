import torch
import torch.nn.functional as F
from igfold.training.utils import do_kabsch


def RMSD_loss_fn(preds, target, mask):
    aligned_target = do_kabsch(
        mobile=target,
        stationary=preds.detach(),
        align_mask=None,
    )
    mse = F.mse_loss(
        preds,
        aligned_target,
        reduction="none",
    ).sum((-2, -1))
    rmsd_sq = torch.sum(mse * mask, dim=-1) / mask.sum()
    return rmsd_sq


def inner_dist_matrices_mse(
        preds,
        target,
        mask,
        atom_types_mask=torch.Tensor([0, 1, 0, 0, 1])):
    preds = preds[0, mask, atom_types_mask, :].reshape((-1, 3))
    preds_dist_matrix = torch.sqrt(
        torch.sum((preds.unsqueeze(1) - preds) ** 2, dim=-1)
    )

    target = target[0, mask, atom_types_mask, :].reshape((-1, 3))
    target_dist_matrix = torch.sqrt(
        torch.sum((target.unsqueeze(1) - target) ** 2, dim=-1)
    )

    mse = F.mse_loss(preds_dist_matrix, target_dist_matrix)

    return mse
