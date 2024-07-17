from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn.functional as F
import torch
from einops import rearrange
import warnings
warnings.filterwarnings("ignore")
################################################ CONSTANT
EPS = 1e-6
# Backbone bond lengths
BL_N_CA = 1.459
BL_CA_C = 1.525
BL_C_N = 1.336
BL_C_O = 1.229

# Backbone bond angles
BA_N_CA_C = 111.0
BA_N_CA_CB = 110.6
BA_CA_C_N = 117.2
BA_CA_C_O = 120.1
BA_O_C_N = 122.7
BA_C_CA_CB = 110.6
BA_C_N_CA = 121.7

# Van der Waals radii
VDW_N = 1.55
VDW_C = 1.7
############################################# CONSTANT


############################################# USING FUNC
def normed_vec(vec, eps=EPS):
    mag_sq = torch.sum(vec**2, dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + eps)
    vec = vec / mag

    return vec


def normed_cross(vec1, vec2, eps=EPS):
    vec1 = normed_vec(vec1, eps=eps)
    vec2 = normed_vec(vec2, eps=eps)
    cross = torch.cross(vec1, vec2, dim=-1)

    return cross


def dist(x_1, x_2, eps=EPS):
    d_sq = (x_1 - x_2)**2
    d = torch.sqrt(d_sq.sum(-1) + eps)

    return d


def angle(x_1, x_2, x_3, eps=EPS):
    a = normed_vec(x_1 - x_2, eps=eps)
    b = normed_vec(x_3 - x_2, eps=eps)
    ang = torch.arccos((a * b).sum(-1))

    return ang


def dihedral(x_1, x_2, x_3, x_4, eps=EPS):
    b1 = normed_vec(x_1 - x_2, eps=eps)
    b2 = normed_vec(x_2 - x_3, eps=eps)
    b3 = normed_vec(x_3 - x_4, eps=eps)
    n1 = normed_cross(b1, b2, eps=eps)
    n2 = normed_cross(b2, b3, eps=eps)
    m1 = normed_cross(n1, b2, eps=eps)
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)

    dih = torch.atan2(y, x)

    return dih


def coords_to_frame(coords, eps=EPS):
    if len(coords.shape) == 3:
        coords = rearrange(
            coords,
            "b (l a) d -> b l a d",
            l=coords.shape[-2] // 4,
        )

    N, CA, C, _ = coords.unbind(-2)
    CA_N = normed_vec(N - CA, eps=eps)
    CA_C = normed_vec(C - CA, eps=eps)
    n1 = CA_N
    n2 = normed_cross(n1, CA_C, eps=eps)
    n3 = normed_cross(n1, n2, eps=eps)
    rot = torch.stack([n1, n2, n3], -1)

    return CA, rot


from igfold.utils.constants import *
def exists(x):
    return x is not None



############################################# USING FUNC





############################################# FUNC FOR MSE 


def kabsch(
    mobile,
    stationary,
    return_translation_rotation=False,
):
    X = rearrange(
        mobile,
        "... l d -> ... d l",
    )
    Y = rearrange(
        stationary,
        "... l d -> ... d l",
    )

    #  center X and Y to the origin
    XT, YT = X.mean(dim=-1, keepdim=True), Y.mean(dim=-1, keepdim=True)
    X_ = X - XT
    Y_ = Y - YT

    # calculate convariance matrix
    C = torch.einsum("... x l, ... y l -> ... x y", X_, Y_)

    # Optimal rotation matrix via SVD
    if int(torch.__version__.split(".")[1]) < 8:
        # warning! int torch 1.<8 : W must be transposed
        V, S, W = torch.svd(C)
        W = rearrange(W, "... a b -> ... b a")
    else:
        V, S, W = torch.linalg.svd(C)

    # determinant sign for direction correction
    v_det = torch.det(V.to("cpu")).to(X.device)
    w_det = torch.det(W.to("cpu")).to(X.device)
    d = (v_det * w_det) < 0.0
    if d.any():
        S[d] = S[d] * (-1)
        V[d, :] = V[d, :] * (-1)

    # Create Rotation matrix U
    U = torch.matmul(V, W)  #.to(device)

    U = rearrange(
        U,
        "... d x -> ... x d",
    )
    XT = rearrange(
        XT,
        "... d x -> ... x d",
    )
    YT = rearrange(
        YT,
        "... d x -> ... x d",
    )

    if return_translation_rotation:
        return XT, U, YT

    transform = lambda coords: torch.einsum(
        "... l d, ... x d -> ... l x",
        coords - XT,
        U,
    ) + YT
    mobile = transform(mobile)

    return mobile, transform


def do_kabsch(
    mobile,
    stationary,
    align_mask=None,
):
    mobile_, stationary_ = mobile.clone(), stationary.clone()
    if exists(align_mask):
        mobile_[~align_mask] = mobile_[align_mask].mean(dim=-2)
        stationary_[~align_mask] = stationary_[align_mask].mean(dim=-2)
        _, kabsch_xform = kabsch(
            mobile_,
            stationary_,
        )
    else:
        _, kabsch_xform = kabsch(
            mobile_,
            stationary_,
        )

    return kabsch_xform(mobile)


def kabsch_mse(
    pred,
    target,
    align_mask=None,
    mask=None,
    clamp=0.,
    sqrt=False,
):
    aligned_target = do_kabsch(
        mobile=target,
        stationary=pred.detach(),
        align_mask=align_mask,
    )
    mse = F.mse_loss(
        pred,
        aligned_target,
        reduction='none',
    ).mean(-1)

    if clamp > 0:
        mse = torch.clamp(mse, max=clamp**2)

    if exists(mask):
        mse = torch.sum(
            mse * mask,
            dim=-1,
        ) / torch.sum(
            mask,
            dim=-1,
        )
    else:
        mse = mse.mean(-1)

    if sqrt:
        mse = mse.sqrt()

    return mse


############################################ RMSD
def RMSD_LOSS(binder1, binder2, sqrt=True):
    mask = torch.concat((binder1.vh_cdr_mask, binder1.vl_cdr_mask))
    kabsch_loss = kabsch_mse(binder1.coords, binder2.coords, mask=mask.unsqueeze(0))
    rmsd = kabsch_loss.nansum() / (~torch.isnan(kabsch_loss)).sum()
    return float(rmsd.item())
########################################### V_GENE
import subprocess
def v_gene(seq):
    command = f'ANARCI -i {seq} --assign_germline'
    result = subprocess.check_output(command, shell=True, text=True)
    count_r = 0
    count_il = 0
    # print(result)
    for i in range(len(result)):
        if result[i] == '#':
            count_r += 1  
        if count_r == 9:
            # print(count_r)
            niz = result[i:].find('\n')+1
            # print(niz)
            result_str = result[i:]
            # print(result_str)
            result_str = result_str[:niz]
            # print(result_str)
            break
    for j in range(len(result_str)):
        if result_str[j] == '|':
            count_il += 1
        if count_il == 3:
            result_str = result_str[j+1:]
            break
    last = result_str.find('|')
    result_str = result_str[:last]
    v_gene = result_str
    # print(result)
    return float(v_gene)


def loss(binder1, binder2):
    try:
        # return (0.2*v_gene(binder1.vl)+ 0.3*v_gene(binder1.vh) + 0.5*(1/(1+(RMSD_LOSS(binder1, binder2)/0.1)**2)))
        vl, vh, rmsd_l = get_losses(binder1, binder2)
        return 0.2 * vl + 0.3 * vh + 0.5 * rmsd_l
    except Exception as e:
        print(e)
        return -1


def get_losses(binder1, binder2):
    try:
        return v_gene(binder1.vl), v_gene(binder1.vh), 1/(1+(RMSD_LOSS(binder1, binder2)/0.1)**2)
    except Exception as e:
        print(e)
        return -1