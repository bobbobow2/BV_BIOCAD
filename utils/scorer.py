import warnings

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

warnings.filterwarnings("ignore")
from igfold.training.utils import do_kabsch, kabsch_mse


############################################ RMSD
def RMSD_LOSS(binder1, binder2, sqrt=True):
    mask = torch.concat((binder1.vh_cdr_mask, binder1.vl_cdr_mask))
    kabsch_loss = kabsch_mse(binder1.coords, binder2.coords, mask=mask.unsqueeze(0))
    rmsd = kabsch_loss.nansum() / (~torch.isnan(kabsch_loss)).sum()
    return float(rmsd.item())


########################################### V_GENE
import subprocess


def v_gene(seq):
    command = f"ANARCI -i {seq} --assign_germline"
    result = subprocess.check_output(command, shell=True, text=True)
    count_r = 0
    count_il = 0
    # print(result)
    for i in range(len(result)):
        if result[i] == "#":
            count_r += 1
        if count_r == 9:
            # print(count_r)
            niz = result[i:].find("\n") + 1
            # print(niz)
            result_str = result[i:]
            # print(result_str)
            result_str = result_str[:niz]
            # print(result_str)
            break
    for j in range(len(result_str)):
        if result_str[j] == "|":
            count_il += 1
        if count_il == 3:
            result_str = result_str[j + 1 :]
            break
    last = result_str.find("|")
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
        return (
            v_gene(binder1.vl),
            v_gene(binder1.vh),
            1 / (1 + (RMSD_LOSS(binder1, binder2) / 0.1) ** 2),
        )
    except Exception as e:
        print(e)
        return -1
