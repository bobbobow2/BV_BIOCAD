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
        atom_types_mask=torch.tensor([0, 1, 0, 0, 1], dtype=torch.bool)):
    preds = preds.squeeze()
    preds = preds[mask]
    preds = preds[:, atom_types_mask].reshape((-1, 3))
    preds_dist_matrix = torch.cdist(preds, preds)

    target = target.squeeze()
    target = target[mask]
    target = target[:, atom_types_mask].reshape((-1, 3)).detach()
    target_dist_matrix = torch.cdist(target, target)

    mse = F.mse_loss(preds_dist_matrix, target_dist_matrix)
    return mse


def smooth_v_gene(logits, tokenized_germlines):
    ce = F.cross_entropy(
        logits.transpose(1, 2),
        tokenized_germlines.input_ids,
        ignore_index=1
    )
    return ce