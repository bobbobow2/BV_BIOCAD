import os
from glob import glob
from typing import List, Tuple

import antiberty
import igfold
import numpy as np
import torch
import torch.nn as nn
import transformers
from abnumber import Chain
from antiberty import AntiBERTy
from igfold.model.IgFold import IgFold
from igfold.model.interface import IgFoldInput
from igfold.utils.embed import embed
from igfold.utils.fasta import get_fasta_chain_dict
from igfold.utils.folding import fold
from igfold.utils.general import exists
from igfold.utils.pdb import cdr_indices, get_atom_coords, save_PDB, write_pdb_bfactor

project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
trained_models_dir = os.path.join(project_path, "trained_models")
ANTIBERTY_CHECKPOINT_PATH = os.path.join(trained_models_dir, "AntiBERTy_md_smooth")
ANTIBERTY_VOCAB_FILE = os.path.join(trained_models_dir, "vocab.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exists(x):
    return x is not None


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


class FVbinder:
    def __init__(
        self,
        vh: str,
        vl: str,
        coords: torch.Tensor = None,
        vh_cdr_mask: torch.Tensor = None,
        vl_cdr_mask: torch.Tensor = None,
    ):
        self.vh = Chain(vh, scheme="chothia")
        self.vl = Chain(vl, scheme="chothia")
        self.coords = coords
        self.vh_cdr_mask = vh_cdr_mask
        if self.vh_cdr_mask is None:
            self.vh_cdr_mask = self.get_mask_from_sequence(self.vh).to(device)
        self.vl_cdr_mask = vl_cdr_mask
        if self.vl_cdr_mask is None:
            self.vl_cdr_mask = self.get_mask_from_sequence(self.vl).to(device)

    @staticmethod
    def get_mask_from_sequence(chain: Chain) -> torch.Tensor:
        mask = list()
        for pos in chain.positions:
            mask.append(torch.full((5,), fill_value=int(pos.is_in_cdr())))
        mask = torch.stack(mask).float()
        return mask

    def to_pdb(self, filename: str):
        if self.coords is None:
            raise ValueError("No coordinates specified. Provide coords or use IgFold")

        seqs = [self.vh.seq, self.vl.seq]
        full_seq = "".join(seqs)
        chains = ["H", "L"]
        delims = np.cumsum([len(s) for s in seqs]).tolist()
        pdb_string = save_PDB(
            filename,
            self.coords.squeeze(0),
            full_seq,
            chains=chains,
            atoms=["N", "CA", "C", "CB", "O"],
            # error=res_rmsd,
            delim=delims,
            write_pdb=True,
        )

    def __str__(self):
        return f"FVbinder(\nvh='{self.vh.seq}',\nvl='{self.vl.seq}'\n)"

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return str(self) < str(other)


ckpt_path = os.path.join(
    os.path.dirname(os.path.realpath(igfold.__file__)),
    "trained_models/IgFold/*.ckpt",
)
model_ckpts = list(glob(ckpt_path))


class CustomIgFold:
    def __init__(self):
        self.antiberty = AntiBERTy.from_pretrained(ANTIBERTY_CHECKPOINT_PATH).to(device)
        self.antiberty.eval()

        self.antiberty_tokenizer = transformers.BertTokenizer(
            vocab_file=ANTIBERTY_VOCAB_FILE, do_lower_case=False
        )

        self.models = []
        for ckpt_file in model_ckpts:
            print(f"Loading {ckpt_file}...")
            self.models.append(IgFold.load_from_checkpoint(ckpt_file).eval().to(device))

    def fold(self, binder: FVbinder):
        if exists(binder.coords):
            return

        seqs = [" ".join(list(i)) for i in [binder.vh.seq, binder.vl.seq]]
        tokenizer_out = self.antiberty_tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
        ).to(device)
        attention_mask = tokenizer_out["attention_mask"].to(device)

        emb = self.antiberty.bert.embeddings(
            input_ids=tokenizer_out.input_ids,
            token_type_ids=tokenizer_out.token_type_ids,
        )

        with torch.no_grad():
            outputs = self.antiberty(
                # input_ids=tokenizer_out.input_ids,
                inputs_embeds=emb,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        embeddings = outputs.hidden_states
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = list(embeddings.detach())

        for i, a in enumerate(attention_mask):
            embeddings[i] = embeddings[i][:, a == 1]

        hidden_layer = -1
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i][hidden_layer]

        attentions = outputs.attentions
        attentions = torch.stack(attentions, dim=1)
        attentions = list(attentions.detach())

        for i, a in enumerate(attention_mask):
            attentions[i] = attentions[i][:, :, a == 1]
            attentions[i] = attentions[i][:, :, :, a == 1]

        embeddings = [e[1:-1].unsqueeze(0) for e in embeddings]
        attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0) for a in attentions]

        model_in = IgFoldInput(
            embeddings=embeddings,
            attentions=attentions,
            template_coords=None,
            template_mask=None,
            return_embeddings=True,
        )
        with torch.no_grad():
            model_outs, scores = [], []
            for i, model in enumerate(self.models):
                model_out = model(model_in)
                model_out = model.gradient_refine(model_in, model_out)
                scores.append(model_out.prmsd.quantile(0.9))
                model_outs.append(model_out)

        best_model_i = scores.index(min(scores))
        model_out = model_outs[best_model_i]

        binder.coords = model_out.coords
