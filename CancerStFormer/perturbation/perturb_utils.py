from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
import torch.nn.functional as F

from collections import defaultdict
import pickle
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset helpers
# ============================================================================

def load_and_filter_dataset(filter_data: Optional[dict], nproc: int, input_data_file: str) -> Dataset:
    """
    Load a huggingface `Dataset` from disk and apply optional metadata filtering.
    """
    data: Dataset = load_from_disk(input_data_file)
    if filter_data:
        data = filter_by_metadata(data, filter_data, nproc)
    return data


def filter_by_metadata(data: Dataset, filter_data: dict, nproc: int) -> Dataset:
    """
    Keep rows where each key's value is in the provided list.
    """
    for key, vals in filter_data.items():
        if not isinstance(vals, list):
            vals = [vals]
        data = data.filter(lambda ex: ex.get(key) in vals, num_proc=nproc)
    if len(data) == 0:
        raise ValueError("No rows remain after metadata filtering; check `filter_data`.")
    return data


def filter_by_start_state(data: Dataset, state_dict: dict, nproc: int) -> Dataset:
    """
    Keep rows whose `state_key` (specified in `state_dict['state_key']`) equals desired values.
    """
    key = state_dict.get("state_key")
    if key is None:
        return data
    wants = []
    for k, v in state_dict.items():
        if k == "state_key":
            continue
        if isinstance(v, list):
            wants.extend(v)
        else:
            wants.append(v)

    return data.filter(lambda ex: ex.get(key) in wants, num_proc=nproc)


def slice_by_indices_to_perturb(data: Dataset, inds: dict) -> Dataset:
    """
    Return a contiguous slice from 'start' (inclusive) to 'end' (exclusive).
    """
    start = int(inds.get("start", 0))
    end = int(inds.get("end", len(data)))
    if start < 0 or end <= start or start >= len(data):
        raise ValueError(f"Invalid slice range: start={start}, end={end}, len={len(data)}")
    end = min(end, len(data))
    return data.select(range(start, end))


def downsample_and_sort(data: Dataset, max_ncells: int) -> Dataset:
    """
    Trim to at most `max_ncells`, preserving original order (stable head).
    """
    if max_ncells is None or len(data) <= max_ncells:
        return data
    return data.select(range(int(max_ncells)))


# ============================================================================
# Model helpers
# ============================================================================

def load_model_to_device(model_type: str, num_classes: int, model_directory: str, mode: str = "eval"):
    """
    Load a model from directory according to `model_type`.
    Expects the model directory to be a HF checkpoint with config.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "Pretrained":
        model = AutoModel.from_pretrained(model_directory, output_hidden_states=True)
    elif model_type == "GeneClassifier" or model_type == "CellClassifier":
        # sequence/classification heads with hidden states
        model = AutoModelForSequenceClassification.from_pretrained(
            model_directory, num_labels=int(num_classes), output_hidden_states=True
        )
    else:
        raise ValueError(f"Unsupported model_type={model_type!r}")
    model.to(device)
    if mode == "eval":
        model.eval()
    return model


def quant_layers(model) -> int:
    """
    Return number of hidden layers to allow negative indexing (-1 = last layer).
    """
    n = getattr(model.config, "num_hidden_layers", None)
    if n is None and hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        n = getattr(model.base_model.config, "num_hidden_layers", None)
    if n is None:
        # fallback: try reading from a typical transformer stack attribute
        n = len(getattr(model, "encoder", getattr(model, "transformer", [])).layer)
    return int(n)


def get_model_hidden_size(model) -> int:
    """
    Hidden size (embedding dimension).
    """
    h = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    if h is None and hasattr(model, "base_model"):
        h = getattr(model.base_model.config, "hidden_size", None)
    if h is None:
        raise AttributeError("Unable to determine model hidden size from config.")
    return int(h)


def get_model_input_size(model) -> int:
    """
    Maximum input sequence length from config (or a conservative default).
    """
    max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is None and hasattr(model, "base_model"):
        max_len = getattr(model.base_model.config, "max_position_embeddings", None)
    return int(max_len or 512)


# Alias expected by other modules
get_model_emb_dims = get_model_hidden_size
load_model = load_model_to_device


def quant_cos_sims_tokenwise(hid_orig: torch.Tensor, hid_pert: torch.Tensor) -> torch.Tensor:
    # normalize on the embedding dimension (last)
    ho = F.normalize(hid_orig, p=2, dim=-1)
    hp = F.normalize(hid_pert, p=2, dim=-1)
    return (ho * hp).sum(dim=-1)

# Cosine similarity utilities
def quant_cos_sims(A: torch.Tensor, B: torch.Tensor,
                   cell_states_to_model=None, state_embs_dict=None, emb_mode="gene") -> torch.Tensor:
    if emb_mode == "gene":
        # A,B: [B, L, H] -> per-gene cosine along H
        a = torch.nn.functional.normalize(A, dim=-1)
        b = torch.nn.functional.normalize(B, dim=-1)
        return (a * b).sum(dim=-1)  # [B, L]
    elif emb_mode == "cell":
        # A,B: [B, H]
        return torch.nn.functional.cosine_similarity(A, B, dim=-1)
    else:
        raise ValueError("emb_mode must be 'gene' or 'cell'")
    
def remove_front_per_example(hid: torch.Tensor, k_vec: torch.Tensor) -> torch.Tensor:
    """
    Remove the first k[b] tokens from each row of a [B, L, D] tensor and
    left-pad with zeros to the max remaining length in the batch.
    """
    B, L, D = hid.shape
    device = hid.device
    rows = []
    maxL = 1
    for b in range(B):
        k = int(k_vec[b].item())
        k = max(0, min(k, L)) 
        row = hid[b, k:, :] 
        maxL = max(maxL, row.shape[0])
        rows.append(row)
    out = []
    for row in rows:
        if row.shape[0] < maxL:
            pad = torch.zeros(maxL - row.shape[0], row.shape[1], device=device, dtype=hid.dtype)
            row = torch.cat([row, pad], dim=0)
        out.append(row.unsqueeze(0))
    return torch.cat(out, dim=0) 

def remove_front_per_example_2d(ids: torch.Tensor, k_vec: torch.Tensor) -> torch.Tensor:
    """
    Remove the first k[b] tokens from each row of a [B, L] tensor and
    left-pad with zeros to the max remaining length in the batch.
    """
    B, L = ids.shape
    device = ids.device
    rows = []
    maxL = 1
    for b in range(B):
        k = int(k_vec[b].item())
        k = max(0, min(k, L))
        row = ids[b, k:]      
        maxL = max(maxL, row.shape[0])
        rows.append(row)
    out = []
    for row in rows:
        if row.shape[0] < maxL:
            pad = torch.zeros(maxL - row.shape[0], device=device, dtype=ids.dtype)
            row = torch.cat([row, pad], dim=0)
        out.append(row.unsqueeze(0))
    return torch.cat(out, dim=0)  
# ============================================================================
# Padding / masking utilities
# ============================================================================

def pad_tensor_list(
    tensors: List[torch.Tensor],
    max_len: int,
    pad_token_id: int,
    model_input_size: int,
    dim_to_pad: int = 1,
    pad_fn=None,
) -> torch.Tensor:
    """
    Pad a list of 2D/3D tensors along the token-length dimension and stack.
    """
    if pad_fn is None:
        # default: 2D [B, L] pad then stack
        padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=pad_token_id) for t in tensors]
        return torch.stack(padded, dim=0)
    return pad_fn(tensors, max_len, pad_token_id, model_input_size, dim_to_pad)


def pad_3d_tensor(
    tensors: List[torch.Tensor],
    max_len: int,
    pad_token_id: int,
    model_input_size: int,
    dim_to_pad: int = 1,
) -> torch.Tensor:
    """
    Pad a list of [L, H] or [B, L, H] tensors into a single [B, L, H] tensor.
    """
    out: List[torch.Tensor] = []
    for t in tensors:
        if t.dim() == 2:
            t = t.unsqueeze(0)
        L = t.shape[1]
        if L < max_len:
            pad_amount = max_len - L
            pad = torch.zeros((t.shape[0], pad_amount, t.shape[2]), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        out.append(t)
    return torch.cat(out, dim=0)


def gen_attention_mask(batch: Dataset) -> torch.Tensor:
    """
    Build attention mask from a dataset batch that has 'length' and 'input_ids'.
    """
    lengths = torch.as_tensor(batch["length"])
    max_len = int(max(lengths))
    return gen_attention_mask_from_lengths(max_len, len(lengths))


def gen_attention_mask_from_lengths(lengths: int, batch_size: int) -> torch.Tensor:
    # create on CPU; ModelAdapter.forward will move to model device
    return torch.ones((batch_size, int(lengths)), dtype=torch.long)

def mean_nonpadding_embs(embs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool embeddings over the first `lengths` tokens (per row).
    Expects `embs` shape [B, L, H] and `lengths` as [B].
    """
    B, L, H = embs.shape
    device = embs.device
    rng = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    mask = (rng < lengths.unsqueeze(1)).float().unsqueeze(-1)  # [B, L, 1]
    summed = (embs * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
    return summed / denom


# ============================================================================
# HF-style adapters for batching & forward pass
# ============================================================================

@dataclass
class BatchMaker:
    pad_token_id: int
    model_input_size: int
    batch_size: int = 64

    def iter(self, data: Dataset, with_indices: bool = False, progress_desc: Optional[str] = None):
        """
        Yield dict batches with 'input_ids', 'attention_mask', and 'lengths'.
        """
        total = len(data)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = data.select(range(start, end))
            # infer max_len
            max_len = max(int(x) for x in batch["length"])
            # format to tensors
            batch.set_format(type="torch")
            ids = batch["input_ids"]
            ids = pad_tensor_list(ids, max_len, self.pad_token_id, self.model_input_size)
            attn = gen_attention_mask_from_lengths(max_len, ids.shape[0])
            lens = torch.as_tensor(batch["length"], device=ids.device)
            yield {"input_ids": ids, "attention_mask": attn, "lengths": lens}


class ModelAdapter:
    """
    Small collection of static helpers to interact with models consistently.
    """

    # discovery
    @staticmethod
    def get_pad_token_id(model) -> int:
        return int(getattr(getattr(model, "config", None), "pad_token_id", 0) or 0)

    @staticmethod
    def get_model_input_size(model) -> int:
        return get_model_input_size(model)

    @staticmethod
    def quant_layers(model) -> int:
        return quant_layers(model)

    # forward/layer selection
    @staticmethod
    def forward(model, batch: Dict[str, torch.Tensor]):
        # NEW: ensure inputs live on same device as model
        device = next(model.parameters()).device
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        return model(input_ids=ids, attention_mask=attn) 

    @staticmethod
    def pick_layer(outputs, layer_index: int) -> torch.Tensor:
        return outputs.hidden_states[layer_index]

    # pooling
    @staticmethod
    def pool_mean(embs: torch.Tensor, lengths: torch.Tensor,
                  exclude_cls: bool = False, exclude_eos: bool = False) -> torch.Tensor:
        # NEW: keep lengths on same device as embs
        lengths = lengths.to(embs.device)

        if exclude_cls:
            embs = embs[:, 1:, :]
            lengths = lengths - 1
        if exclude_eos:
            embs = embs[:, :-1, :]
            lengths = lengths - 1
        return mean_nonpadding_embs(embs, lengths)

# ============================================================================
# Perturbation operators (group/single)
# ============================================================================

@dataclass
class PerturbOps:
    genes_to_keep: Optional[List[Union[int, str]]] = None
    genes_to_perturb: Optional[List[Union[int, str]]] = None
    perturb_fraction: Optional[float] = None
    rank_shift: Optional[str] = None
    top_k: Optional[int] = None
    pad_token_id: int = 0

    def _as_set(self, ids: Optional[Iterable[Union[int, str]]]) -> set:
        return set([] if ids is None else [int(x) for x in ids])
    
    def iter_single_tokens(self, input_ids: torch.Tensor) -> Iterable[int]:
        """Yield only the requested tokens if provided; else all (minus keep)."""
        keep = self._as_set(self.genes_to_keep)
        present = set(map(int, torch.unique(input_ids).tolist()))

        if self.genes_to_perturb:
            requested = self._as_set(self.genes_to_perturb)
            for t in requested & present:
                if t not in keep:
                    yield t
            return

        # fallback: iterate all present (minus keep)
        for t in present:
            if t not in keep:
                yield t

    def _mask_keep(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask: True where token is allowed to be perturbed.
        """
        keep = self._as_set(self.genes_to_keep)
        if not keep:
            return torch.ones_like(ids, dtype=torch.bool)
        mask = torch.ones_like(ids, dtype=torch.bool)
        for k in keep:
            mask = mask & (ids != int(k))
        return mask

    # --- group mode ---
    def apply_group(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Replace a fraction/top_k of tokens (that are not in `genes_to_keep`)
        with the provided `genes_to_perturb` tokens (cyclic fill).
        """
        ids = input_ids.clone()
        allowed = self._mask_keep(ids)
        B, L = ids.shape
        flat = ids[allowed]
        n = flat.numel()
        if n == 0:
            return ids
        # how many to perturb
        k = n
        if self.top_k is not None:
            k = min(k, int(self.top_k))
        if self.perturb_fraction is not None:
            k = min(k, max(1, int(math.ceil(self.perturb_fraction * n))))
        # choose first k positions deterministically (stable)
        idx = torch.nonzero(allowed, as_tuple=False).view(-1, 2)[:k]
        tok_list = [int(t) for t in (self.genes_to_perturb or [])]
        if not tok_list:
            return ids
        fill = torch.as_tensor([tok_list[i % len(tok_list)] for i in range(k)], device=ids.device, dtype=ids.dtype)
        ids[idx[:, 0], idx[:, 1]] = fill
        return ids

    # --- single mode ---
    def iter_single_tokens(self, input_ids: torch.Tensor) -> Iterable[int]:
        """
        Iterate distinct tokens (excluding keep set) present in the batch.
        """
        keep = self._as_set(self.genes_to_keep)
        toks = torch.unique(input_ids).tolist()
        for t in toks:
            if int(t) not in keep:
                yield int(t)

    def apply_single(self, input_ids: torch.Tensor, gene_tok: int) -> torch.Tensor:
        """Replace occurrences of `gene_tok` with PAD if rank_shift == 'delete'."""
        ids = input_ids.clone()
        if self.rank_shift == "delete":
            return torch.where(
                ids == int(gene_tok),
                torch.as_tensor(self.pad_token_id, device=ids.device, dtype=ids.dtype),
                ids,
            )
        return ids


def validate_gene_token_mapping(ens2tok: dict) -> dict:
    """
    Normalize and validate a gene->token dict. Returns cleaned dict.
    Logs collisions and suspicious entries (non-int tokens).
    """
    import logging
    logger = logging.getLogger(__name__)
    cleaned = {}
    bad = 0
    for k, v in ens2tok.items():
        ks = str(k)
        try:
            vi = int(v)
        except Exception:
            bad += 1
            continue
        cleaned[ks] = vi
    if bad:
        logger.warning("Dropped %d entries with non-integer token IDs.", bad)

    # collision check (token->multiple ensembl)
    rev = {}
    col = 0
    for e, t in cleaned.items():
        if t in rev and rev[t] != e:
            col += 1
        else:
            rev[t] = e
    if col:
        logger.warning("Detected %d token->ensembl collisions (keeping first occurrence).", col)
    return cleaned

# Delete tokens at indices in example["perturb_index"]
def delete_indices(example: dict) -> dict:
    idxs = example.get("perturb_index", [])
    if not idxs or idxs == [-100]:
        return example
    ids = example["input_ids"]
    keep = [tok for i, tok in enumerate(ids) if i not in idxs]
    example["input_ids"] = keep
    example["length"] = len(keep)
    return example

# Move tokens_to_perturb to the front, preserving order, clip overflow to max_len
def overexpress_tokens(example: dict, max_len: int) -> dict:
    ids = example["input_ids"]
    toks = example.get("tokens_to_perturb", [])
    front = [t for t in toks if t in ids]
    tail = [t for t in ids if t not in front]
    new = (front + tail)[:max_len]
    example["input_ids"] = new
    example["length"] = len(new)
    return example

def calc_n_overflow(max_len: int, length: int, tokens_to_perturb: list, indices_to_perturb: list) -> int:
    # how many items would be pushed off the end when overexpressing
    present = sum(1 for i in indices_to_perturb if i is not None)
    added = present
    overflow = max(0, (length + added) - max_len)
    return int(overflow)

def truncate_by_n_overflow(example: dict) -> dict:
    n = int(example.get("n_overflow", 0))
    if n <= 0:
        return example
    ids = example["input_ids"]
    example["input_ids"] = ids[:-n]
    example["length"] = len(example["input_ids"])
    return example


def remove_perturbed_indices_set(full_original_emb: torch.Tensor,
                                 perturb_type: str,
                                 indices_to_perturb: list,
                                 tokens_to_perturb: list,
                                 lengths: list) -> torch.Tensor:
    if perturb_type == "overexpress":
        # remove the front len(tokens_to_perturb) positions later (handled in caller)
        return full_original_emb
    # delete: drop perturbed positions
    slices = []
    for b, idxs in enumerate(indices_to_perturb):
        if not idxs or idxs == [-100]:
            slices.append(full_original_emb[b, :lengths[b], :])
        else:
            keep = [j for j in range(lengths[b]) if j not in set(idxs)]
            slices.append(full_original_emb[b, keep, :])
    # pad to max length in batch
    maxL = max(s.shape[0] for s in slices)
    out = []
    for s in slices:
        L = s.shape[0]
        if L < maxL:
            pad = torch.zeros((maxL - L, s.shape[1]), device=s.device, dtype=s.dtype)
            s = torch.cat([s, pad], dim=0)
        out.append(s.unsqueeze(0))
    return torch.cat(out, dim=0)

def compute_nonpadded_cell_embedding(full_emb: torch.Tensor, style: str = "mean_pool") -> torch.Tensor:
    # full_emb: [B, L, H]; mean over non-padding positions (caller must pass lengths if needed)
    # Here we rely on caller to already cut padding; otherwise use your lengths tensor with mean_nonpadding_embs.
    return full_emb.mean(dim=1)


def remove_indices_per_example(hid: torch.Tensor,
                               lengths: torch.Tensor,
                               indices_per_ex: list[list[int]]) -> torch.Tensor:
    """
    Remove a (possibly different) list of indices per example from a [B, L, D] tensor.
    Keeps order of the remaining tokens.
    """
    B, L, D = hid.shape
    kept_rows = []
    device = hid.device
    for b in range(B):
        idxs = sorted(set(int(i) for i in indices_per_ex[b] if i is not None))
        keep = [i for i in range(int(lengths[b])) if i not in idxs]
        # if empty, keep a single pad-like zero row to avoid empty tensors
        if len(keep) == 0:
            kept_rows.append(torch.zeros(1, D, device=device, dtype=hid.dtype))
        else:
            kept_rows.append(hid[b, keep, :])
    # pad to max length among rows
    maxL = max(row.shape[0] for row in kept_rows)
    out = []
    for row in kept_rows:
        if row.shape[0] < maxL:
            pad = torch.zeros(maxL - row.shape[0], D, device=device, dtype=hid.dtype)
            row = torch.cat([row, pad], dim=0)
        out.append(row.unsqueeze(0))
    return torch.cat(out, dim=0)

# ============================================================================
# IO helpers
# ============================================================================

def write_cosine_sim_dict(cos_sims: Dict[Tuple[Union[int, Tuple[int, ...]], str], List[float]], out_dir: str, prefix: str) -> str:
    """
    Write a nested dict as newline-delimited JSON for easy reload.
    """
    out_path = str(Path(out_dir, f"{prefix}_cosine_sims.jsonl"))
    with open(out_path, "w") as f:
        for (tok, metric), vals in cos_sims.items():
            rec = {"key": [tok if not isinstance(tok, tuple) else list(tok), metric], "values": list(map(float, vals))}
            f.write(json.dumps(rec) + "\n")
    return out_path

def read_cosine_sims(path):
    with open(path,'rb') as f:
        return pickle.load(f)



def gene_sims_to_dict(input_ids: torch.Tensor,
                      sims_tokenwise: torch.Tensor,
                      pad_token_id: int) -> Dict[Tuple[int, str], List[float]]:
    """
    Aggregate token-wise similarities into dict entries keyed as (token_id, 'gene_emb').
    input_ids: [B, L] int tensor
    sims_tokenwise: [B, L] float tensor
    """
    out: Dict[Tuple[int, str], List[float]] = {}
    ids_np = input_ids.detach().cpu().numpy()
    sims_np = sims_tokenwise.detach().cpu().numpy()
    B, L = ids_np.shape
    for b in range(B):
        for t in range(L):
            tok = int(ids_np[b, t])
            if tok == pad_token_id:
                continue
            out.setdefault((tok, "gene_emb"), []).append(float(sims_np[b, t]))
    return out

import torch



def remove_indices_per_example(full_emb: torch.Tensor,
                               lengths: torch.Tensor,
                               indices_to_remove: List[List[int]]) -> torch.Tensor:
    """
    Remove specified indices per example from a [B, L, H] tensor and repad to the
    max remaining length within the batch. Keeps device/dtype.
    """
    device, dtype = full_emb.device, full_emb.dtype
    B, L, H = full_emb.shape
    slices = []
    maxL = 1
    for b in range(B):
        keep = [j for j in range(int(lengths[b])) if j not in set(indices_to_remove[b])]
        if not keep:  # degenerate, keep at least a zero row
            s = torch.zeros((1, H), device=device, dtype=dtype)
        else:
            s = full_emb[b, keep, :]
        maxL = max(maxL, s.shape[0])
        slices.append(s)
    out = []
    for s in slices:
        pad_len = maxL - s.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, H), device=device, dtype=dtype)
            s = torch.cat([s, pad], dim=0)
        out.append(s.unsqueeze(0))
    return torch.cat(out, dim=0)  # [B, maxL, H]


def write_perturbation_dictionary(cos_sims_dict: defaultdict, output_path_prefix: str):
    with open(f"{output_path_prefix}_raw.pickle", "wb") as fp:
        pickle.dump(cos_sims_dict, fp)

# ====================================
#       Extended Model Support
# ====================================
def nonpad_len_1d(ids_row: list[int], pad_token_id: int) -> int:
    # count until first pad (assumes right padding)
    for i, t in enumerate(ids_row):
        if int(t) == pad_token_id:
            return i
    return len(ids_row)

def overexpress_tokens_extended(example, half_size: int, model_input_size: int):
    """
    Insert example['tokens_to_perturb'] at the FRONT of BOTH halves.
    Set:
      - example['input_ids'] new concatenated list
      - example['length']
      - example['n_overflow_halves'] = [k_spot, k_neigh]
    """
    pad = example.get("pad_token_id", None)  # optional if you carry pad id in example
    tokens = example["tokens_to_perturb"]
    ids = example["input_ids"]

    spot_ids  = ids[:half_size]
    neigh_ids = ids[half_size:]

    # compute non-pad lengths for each half (fallback if pad unknown: assume no pad)
    if pad is None:
        Ls = min(len(spot_ids),  half_size)
        Ln = min(len(neigh_ids), half_size)
    else:
        Ls = nonpad_len_1d(spot_ids,  pad)
        Ln = nonpad_len_1d(neigh_ids, pad)

    # how many tokens can we insert per half without exceeding half_size
    ins_spot  = min(len(tokens), max(0, half_size - Ls))
    ins_neigh = min(len(tokens), max(0, half_size - Ln))

    # overflow for each half (how many pre-existing tokens we must truncate off the end)
    k_spot  = max(0, len(tokens) - ins_spot)
    k_neigh = max(0, len(tokens) - ins_neigh)

    # build new halves: [tokens[:ins_*]] + original[:half_size - ins_*]
    new_spot  = list(tokens[:ins_spot])  + spot_ids[:half_size - ins_spot]
    new_neigh = list(tokens[:ins_neigh]) + neigh_ids[:half_size - ins_neigh]

    # concat halves back
    new_ids = new_spot + new_neigh
    example["input_ids"] = new_ids[:model_input_size]
    example["length"] = min(model_input_size, (ins_spot + min(Ls, half_size - ins_spot)) +
                                          (ins_neigh + min(Ln, half_size - ins_neigh)))
    example["n_overflow_halves"] = [int(k_spot), int(k_neigh)]
    return example

def truncate_by_n_overflow_extended(example, half_size: int):
    """
    Truncate ORIGINAL example to remove overflow at the END of each half:
    remove k_spot from end of first half and k_neigh from end of second half.
    """
    ids = example["input_ids"]
    k_spot, k_neigh = example.get("n_overflow_halves", [0, 0])
    # first half
    spot_ids  = ids[:half_size]
    neigh_ids = ids[half_size:]
    if k_spot > 0:
        spot_ids = spot_ids[:-k_spot] if k_spot < len(spot_ids) else []
    if k_neigh > 0:
        neigh_ids = neigh_ids[:-k_neigh] if k_neigh < len(neigh_ids) else []
    example["input_ids"] = spot_ids + neigh_ids
    example["length"] = len(spot_ids) + len(neigh_ids)
    return example

def remove_front_per_example_halves(hid: torch.Tensor,
                                    k_spot_vec: torch.Tensor,
                                    k_neig_vec: torch.Tensor,
                                    half_size: int) -> torch.Tensor:
    """
    Drop first k_spot tokens in first half, and first k_neig tokens in second half.
    """
    B, L, D = hid.shape
    device = hid.device
    rows, maxL = [], 1
    for b in range(B):
        k1 = int(k_spot_vec[b].item())
        k2 = int(k_neig_vec[b].item())
        # slice halves
        h1 = hid[b, :half_size, :]
        h2 = hid[b, half_size:, :]
        h1 = h1[k1:, :]
        h2 = h2[k2:, :]
        row = torch.cat([h1, h2], dim=0)   # [L' , D]
        maxL = max(maxL, row.shape[0])
        rows.append(row)
    out = []
    for row in rows:
        if row.shape[0] < maxL:
            pad = torch.zeros(maxL - row.shape[0], row.shape[1], device=device, dtype=hid.dtype)
            row = torch.cat([row, pad], dim=0)
        out.append(row.unsqueeze(0))
    return torch.cat(out, dim=0)

def remove_front_per_example_halves_2d(ids: torch.Tensor,
                                       k_spot_vec: torch.Tensor,
                                       k_neig_vec: torch.Tensor,
                                       half_size: int) -> torch.Tensor:
    """
    Same as above for 2D int token ids.
    """
    B, L = ids.shape
    device = ids.device
    rows, maxL = [], 1
    for b in range(B):
        k1 = int(k_spot_vec[b].item()); k2 = int(k_neig_vec[b].item())
        i1 = ids[b, :half_size]
        i2 = ids[b, half_size:]
        i1 = i1[k1:]
        i2 = i2[k2:]
        row = torch.cat([i1, i2], dim=0)
        maxL = max(maxL, row.shape[0])
        rows.append(row)
    out = []
    for row in rows:
        if row.shape[0] < maxL:
            pad = torch.zeros(maxL - row.shape[0], device=device, dtype=ids.dtype)
            row = torch.cat([row, pad], dim=0)
        out.append(row.unsqueeze(0))
    return torch.cat(out, dim=0)
