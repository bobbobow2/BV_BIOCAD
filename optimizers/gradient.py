from glob import glob

import antiberty
import igfold
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from antiberty import AntiBERTy
from igfold.model.IgFold import IgFold
from igfold.model.interface import IgFoldInput
from igfold.training.utils import do_kabsch, kabsch_mse
from igfold.utils.embed import embed
from igfold.utils.fasta import get_fasta_chain_dict
from igfold.utils.folding import fold
from igfold.utils.general import exists
from igfold.utils.pdb import cdr_indices, get_atom_coords, save_PDB, write_pdb_bfactor
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from utils.general import *
from .losses import RMSD_loss_fn, inner_dist_matrices_mse

# from utils.scorer import kabsch_mse, do_kabsch, kabsch


species_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.15)


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

  
def v_gene_identity(sequence):
    c = Chain(sequence, scheme="imgt")
    v_germline_chains, _ = c.find_human_germlines(limit=1)
    top_v = v_germline_chains[0]

    num_v_matches = sum(top_v.positions.get(pos) == aa for pos, aa in c)

    v_identity = num_v_matches / len(top_v)
    return v_identity


device = torch.device("cuda")


class GenAB(nn.Module):
    def __init__(self, b: FVbinder, sub_infinity=10):
        super().__init__()
        seqs = [b.vh.seq, b.vl.seq]
        seqs = [" ".join(list(j)) for j in seqs]

        self.antiberty = AntiBERTy.from_pretrained(ANTIBERTY_CHECKPOINT_PATH).to(device)
        self.antiberty.eval()

        self.antiberty_tokenizer = transformers.BertTokenizer(
            vocab_file=ANTIBERTY_VOCAB_FILE, do_lower_case=False
        )

        self.tokenizer_out = self.antiberty_tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
        ).to(device)
        self.attention_mask = self.tokenizer_out["attention_mask"].to(device)

        self.logits = (
            F.one_hot(self.tokenizer_out.input_ids, num_classes=25).float()
            * sub_infinity
        )

        # Preprocess logits
        vh_pad_length = (1, self.logits.size(dim=1) - b.vh_cdr_mask.size(dim=0) - 1)
        vh_cdr_mask = F.pad(b.vh_cdr_mask, pad=vh_pad_length, value=0)

        # self.logits[0, vh_cdr_mask][self.logits[0, vh_cdr_mask]!=0] = -np.inf
        # self.logits[0, vh_cdr_mask, self.tokenizer_out.input_ids]

        vl_pad_length = (1, self.logits.size(dim=1) - b.vl_cdr_mask.size(dim=0) - 1)
        vl_cdr_mask = F.pad(b.vl_cdr_mask, pad=vl_pad_length, value=0)
        # self.logits[1, vh_cdr_mask, :] = -np.inf

        # vh_cdr_mask = F.pad(
        #     b.vh_cdr_mask.max(dim=1).values,
        #     pad=(1, self.logits.size(dim=1) - b.vh_cdr_mask.size(dim=0) - 1),
        #     mode="constant",
        #     value=0,
        # )
        self.logits[0][vh_cdr_mask] = self.logits[0][vh_cdr_mask].where(
            self.logits[0][vh_cdr_mask] != 0,
            torch.tensor([-np.inf], device=device),
        )

        # vl_cdr_mask = F.pad(
        #     b.vl_cdr_mask.max(dim=1).values,
        #     pad=(1, self.logits.size(dim=1) - b.vl_cdr_mask.size(dim=0) - 1),
        #     mode="constant",
        #     value=0,
        # )
        self.logits[1][vl_cdr_mask] = self.logits[1][vl_cdr_mask].where(
            self.logits[1][vl_cdr_mask] != 0,
            torch.tensor([-np.inf], device=device),
        )

        self.logits[:, :, 0:5][self.logits[:, :, 0:5] == 0] = -np.inf
        for b in range(self.logits.size(dim=0)):
            for i in range(self.logits.size(dim=1)):
                if 0 <= self.logits[b, i].argmax() <= 4:
                    self.logits[b, i, self.logits[b, i, :] == 0] = -np.inf

        self.logits = nn.Parameter(self.logits)
        self.logits.requires_grad = True

        self.W_matrix = self.antiberty.bert.embeddings.word_embeddings.weight
        self.antiberty.bert.embeddings.word_embeddings = nn.Identity()

        self.models = []
        for ckpt_file in model_ckpts:
            # print(f"Loading {ckpt_file}...")
            m = IgFold.load_from_checkpoint(ckpt_file).eval().to(device)
            self.models.append(m)

    def forward(self):
        w_embeddings = F.softmax(self.logits, dim=-1) @ self.W_matrix
        embeddings = self.antiberty.bert.embeddings(
            input_ids=w_embeddings, token_type_ids=self.tokenizer_out.token_type_ids
        )

        outputs = self.antiberty(
            inputs_embeds=embeddings,
            attention_mask=self.attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        # Pseudo Log-Likelyhood

        log_likelyhood = -F.cross_entropy(
            outputs.prediction_logits.transpose(1, 2),
            F.softmax(self.logits.detach(), dim=-1).transpose(1, 2),
            reduction="mean",
        )

        # Coords Preds
        embeddings = outputs.hidden_states
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = list(embeddings)

        for i, a in enumerate(self.attention_mask):
            embeddings[i] = embeddings[i][:, a == 1]

        hidden_layer = -1
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i][hidden_layer]

        attentions = outputs.attentions
        attentions = torch.stack(attentions, dim=1)
        attentions = list(attentions)

        for i, a in enumerate(self.attention_mask):
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

        model_outs, scores = [], []
        for i, model in enumerate(self.models):
            model_out = model(model_in)
            # model_out = model.gradient_refine(model_in, model_out)
            scores.append(model_out.prmsd.quantile(0.9))
            model_outs.append(model_out)

        best_model_i = scores.index(min(scores))
        model_out = model_outs[best_model_i]

        return model_out.coords, outputs.species_logits, log_likelyhood


class GradientOptimizer:
    def __init__(
        self,
        start: FVbinder,
        target: FVbinder,
        target_class=1,
        lr=0.05,
        sub_infinity=10,
        validation=False,
        verbose=False,
        tensorboard=False,
    ):
        self.start = start
        self.target = target
        assert self.target.coords is not None

        self.target_class = target_class

        self.verbose = verbose
        self.tensorboard = tensorboard
        self.validation = validation

        self.generator = GenAB(b=self.start, sub_infinity=sub_infinity)
        self.optimizer = torch.optim.Adam([self.generator.logits], lr=lr)

        if self.tensorboard:
            self.writer = SummaryWriter()

        self.loss_idx_value = 0
        self.best_valid_binder = None
        self.best_valid_binder_score = 1e12

        self.log_likelyhood = None

    def step(self, steps=1):
        for i in (pbar := tqdm(range(steps))):
            self.optimizer.zero_grad()
            coords, species, log_likelyhood = self.generator.forward()

            # rmsd_loss = RMSD_loss_fn(
            #     target=self.target.coords,
            #     preds=coords,
            #     mask=torch.concat((self.target.vh_cdr_mask, self.target.vl_cdr_mask)),
            # )

            rmsd_loss = inner_dist_matrices_mse(
                coords,
                self.target.coords,
                torch.concat((self.target.vh_cdr_mask, self.target.vl_cdr_mask))
            )

            species_loss = species_loss_fn(
                species,
                torch.full(
                    size=(species.size(dim=0),), fill_value=self.target_class
                ).to(device),
            )
            log_likelyhood_loss = -log_likelyhood

            # loss = log_likelyhood_loss
            if self.log_likelyhood is None:
                self.log_likelyhood = log_likelyhood_loss.detach()
            loss = 10 * rmsd_loss + 0.2 * species_loss + 10 * abs(self.log_likelyhood - log_likelyhood_loss)

            # writer.add_scalars("Loss", {
            #     "RMSD Loss": rmsd_loss,
            #     "Species Loss": species_loss,
            #     "Log-Likelyhood Loss": log_likelyhood_loss,
            #     "Total Loss": loss
            # }, loss_idx_value)
            try:
                gen_vh, gen_vl = [
                    i.replace(" ", "")
                    for i in self.generator.antiberty_tokenizer.batch_decode(
                        self.generator.logits.argmax(dim=2), skip_special_tokens=True
                    )
                ]
                a = FVbinder(vh=gen_vh, vl=gen_vl)
                valid = True
                if loss < self.best_valid_binder_score:
                    self.best_valid_binder = a

                if self.tensorboard and self.validation:
                    self.writer.add_scalar(
                        "Validation/VH_v-gen", v_gene_identity(gen_vh), self.loss_idx_value
                    )
                    self.writer.add_scalar(
                        "Validation/VL_v-gen", v_gene_identity(gen_vl), self.loss_idx_value
                    )
            except Exception as e:
                valid = False

            pbar.set_description(f"Valid: {valid}, Loss {loss}")
            self.writer.add_text("Valid", str(valid))

            if self.verbose:
                print(f"RMSD Loss: {rmsd_loss}")
                print(f"Species Loss: {species_loss}")
                print(f"Log-likelyhood Loss: {log_likelyhood_loss}")
                print(f"Full Loss: {loss}")
                print()
                print("---")

            if self.tensorboard:
                self.writer.add_scalar("Loss/RMSD Loss", rmsd_loss, self.loss_idx_value)
                self.writer.add_scalar(
                    "Loss/Species Loss", species_loss, self.loss_idx_value
                )
                self.writer.add_scalar(
                    "Loss/Log-Likelyhood Loss", log_likelyhood_loss, self.loss_idx_value
                )
                self.writer.add_scalar("Loss/Total Loss", loss, self.loss_idx_value)

            loss.backward()
            self.optimizer.step()
            self.loss_idx_value += 1
            # print(m.logits)
