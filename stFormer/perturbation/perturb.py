
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
import pickle
from collections import defaultdict

from perturb_utils import (
    # dataset I/O + filtering
    load_and_filter_dataset as load_and_filter,
    filter_by_metadata as filter_by_dict,
    filter_by_start_state,
    slice_by_indices_to_perturb as slice_by_inds_to_perturb,
    # model helpers
    load_model_to_device as load_model,
    get_model_hidden_size as get_model_emb_dims,
    validate_gene_token_mapping,
    delete_indices,
    overexpress_tokens,
    calc_n_overflow,
    truncate_by_n_overflow,
    pad_tensor_list,
    remove_front_per_example_2d,
    remove_front_per_example,
    remove_indices_per_example,
    quant_cos_sims_tokenwise,
    # extended model options
    overexpress_tokens_extended,
    truncate_by_n_overflow_extended,
    remove_front_per_example_halves,
    remove_front_per_example_halves_2d,
    # batching + adapters
    BatchMaker,
    ModelAdapter,
    PerturbOps,
    # output
    write_perturbation_dictionary
)

logger = logging.getLogger(__name__)

import warnings
from requests.exceptions import RequestsDependencyWarning
warnings.simplefilter("ignore", RequestsDependencyWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="datasets.table")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


@dataclass
class Config:
    # perturbation setup
    perturb_type: str  # {"group","single"}
    model_variant: str
    perturb_rank_shift: Optional[str]
    genes_to_perturb: Optional[List[Union[int, str]]]
    genes_to_keep: Optional[List[Union[int, str]]]
    perturb_fraction: Optional[float]
    top_k: Optional[int]
    emb_mode: str
    # data/model
    num_samples: int
    forward_batch_size: int
    model_type: str  # {"Pretrained","GeneClassifier","CellClassifier"}
    num_classes: int
    emb_layer: int
    token_dictionary_file: Optional[str]

    # filtering
    start_state: Optional[Dict[str, Union[str, int, float]]]
    filter_data: Optional[Dict[str, List[Union[str, int, float]]]]
    cell_inds_to_perturb: Optional[Dict[str, int]]  # {"start": i, "end": j}
    nproc: int

    def __post_init__(self) -> None:
        _choice("perturb_type", self.perturb_type, {"group", "single"})
        _choice('model_variant',self.model_variant,{'spot','extended'})
        _choice("model_type", self.model_type, {"Pretrained", "GeneClassifier", "CellClassifier"})
        _choice("emb_mode", getattr(self, "emb_mode", "cell"), {"cell", "gene", "cell_and_gene"})
        _ge("num_samples", self.num_samples, 1)
        _ge("forward_batch_size", self.forward_batch_size, 1)
        _ge("nproc", self.nproc, 1)
        if self.perturb_fraction is not None and not (0.0 < float(self.perturb_fraction) <= 1.0):
            raise ValueError("`perturb_fraction` must be in (0, 1].")
        if self.cell_inds_to_perturb is not None:
            if not {"start", "end"} <= set(self.cell_inds_to_perturb):
                raise ValueError("`cell_inds_to_perturb` must include keys {'start','end'}.")
            if int(self.cell_inds_to_perturb["start"]) < 0:
                raise ValueError("`cell_inds_to_perturb['start']` must be >= 0.")
        if self.filter_data is not None and not isinstance(self.filter_data, dict):
            raise ValueError("`filter_data` must be a dict or None.")
        # normalize filter_data scalars to lists
        if isinstance(self.filter_data, dict):
            for k, v in list(self.filter_data.items()):
                if not isinstance(v, list):
                    self.filter_data[k] = [v]


def _choice(name: str, value, allowed: set) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid `{name}` = {value!r}; allowed: {sorted(allowed)}")


def _ge(name: str, value: int, lo: int) -> None:
    if int(value) < lo:
        raise ValueError(f"Invalid `{name}` = {value!r}; must be >= {lo}")


class Perturber:
    """
    Apply in-silico perturbations to tokenized sequences and measure changes
    in pooled cell embeddings. This facade keeps the original API while
    modernizing internals.
    """

    def __init__(
        self,
        perturb_type,
        model_variant,
        perturb_rank_shift,
        genes_to_perturb,
        genes_to_keep,
        perturb_fraction,
        num_samples,
        top_k,
        model_type,
        num_classes,
        start_state,
        filter_data,
        emb_layer,
        emb_mode,
        forward_batch_size,
        nproc,
        token_dictionary_file,
        cell_inds_to_perturb=None,
        **kwargs,  # ignored legacy extras for forward-compat
    ):
        self.cfg = Config(
            perturb_type=perturb_type,
            model_variant=model_variant,
            perturb_rank_shift=perturb_rank_shift,
            genes_to_perturb=genes_to_perturb,
            genes_to_keep=genes_to_keep,
            perturb_fraction=perturb_fraction,
            top_k=top_k,
            num_samples=int(num_samples),
            forward_batch_size=int(forward_batch_size),
            emb_mode = str(emb_mode),
            model_type=model_type,
            num_classes=int(num_classes),
            emb_layer=int(emb_layer),
            token_dictionary_file=token_dictionary_file,
            start_state=start_state,
            filter_data=filter_data,
            cell_inds_to_perturb=cell_inds_to_perturb,
            nproc=int(nproc),
        )

    def perturb_dataset(
        self,
        model_directory: str,
        input_data_file: str,
        output_directory: str,
        output_prefix: str,
    ) -> str:
        import pickle
        from typing import Dict, List, Tuple, Union
        import torch
        from datasets import Dataset
        from tqdm.auto import tqdm

        # ---- Model ----
        model = load_model(self.cfg.model_type, self.cfg.num_classes, model_directory, mode="eval")
        target_layer = ModelAdapter.quant_layers(model) + self.cfg.emb_layer
        model_input_size = ModelAdapter.get_model_input_size(model)
        half_size = model_input_size // 2   # first half: spot, second half: neighbor

        # ---- Load + filter dataset ----
        ds: Dataset = load_and_filter(self.cfg.filter_data, self.cfg.nproc, input_data_file)
        # filter cells by tokens_to_perturb for delete/inhibit  ---
        if (self.cfg.genes_to_perturb and
            (self.cfg.perturb_rank_shift in {"delete"} or self.cfg.perturb_type == "delete")):
            # Resolve tokens now (same logic as you already do later)
            import pickle
            with open(self.cfg.token_dictionary_file, "rb") as f:
                ens2tok_raw = pickle.load(f)
            ens2tok = validate_gene_token_mapping(ens2tok_raw)
            # resolve perturb targets (ENSEMBL -> token ids)
            resolved_perturb = []
            for x in self.cfg.genes_to_perturb:
                if x in ens2tok: resolved_perturb.append(ens2tok[x])
            resolved_perturb = sorted(set(resolved_perturb))
            if not resolved_perturb:
                raise ValueError("No valid genes_to_perturb in token dictionary.")
            # keep only rows whose input_ids contain ALL tokens_to_perturb (Geneformer semantics)

            tokset = set(resolved_perturb)

            if self.cfg.model_variant == 'extended':
                # keep examples where ALL perturb tokens appear in BOTH halves
                def _both_halves(example):
                    ids = example["input_ids"]
                    first  = set(int(t) for t in ids[:half_size])
                    second = set(int(t) for t in ids[half_size:])
                    return tokset.issubset(first) and tokset.issubset(second)

                ds = ds.filter(_both_halves, num_proc=self.cfg.nproc)
            elif self.cfg.model_variant == 'spot':
                def _has_all(example):
                    ids = set(int(t) for t in example["input_ids"])
                    return tokset.issubset(ids)
                ds = ds.filter(_has_all, num_proc=self.cfg.nproc)
            else:
                raise ValueError('Not valid model_variant option, please pick between: spot and extended')

        if self.cfg.start_state is not None:
            ds = filter_by_start_state(ds, self.cfg.start_state, self.cfg.nproc)
        if self.cfg.num_samples is not None:
            if len(ds) > self.cfg.num_samples:
                ds = ds.shuffle(seed=42)
                ds = ds.select([i for i in range(self.cfg.num_samples)])
        ds = ds.sort("length", reverse=True)

        if self.cfg.cell_inds_to_perturb is not None:
            ds = slice_by_inds_to_perturb(ds, self.cfg.cell_inds_to_perturb)



        # ---- Batching ----
        pad_id = ModelAdapter.get_pad_token_id(model)
        model_input_size = ModelAdapter.get_model_input_size(model)
        batcher = BatchMaker(
            pad_token_id=pad_id,
            model_input_size=model_input_size,
            batch_size=self.cfg.forward_batch_size,
        )

        # ---- Resolve Ensembl IDs <-> token IDs (robust) ----
        with open(self.cfg.token_dictionary_file, "rb") as f:
            ens2tok_raw = pickle.load(f)
        ens2tok = validate_gene_token_mapping(ens2tok_raw)
        tok2ens = {v: k for k, v in ens2tok.items()}

        def _as_int(x):
            try: return int(x)
            except Exception: return None

        def _resolve_ids(name, items):
            if items is None: return None
            resolved, unknown_ens, unknown_tok = [], [], []
            for x in items:
                xi = _as_int(x)
                if xi is not None:
                    (resolved if xi in tok2ens else unknown_tok).append(x); continue
                xs = str(x)
                (resolved if xs in ens2tok else unknown_ens).append(ens2tok.get(xs, xs))
            resolved = sorted(set([r for r in resolved if isinstance(r, int)]))
            if not resolved:
                raise ValueError(f"[{name}] No valid IDs resolved.")
            return resolved

        resolved_perturb = _resolve_ids("genes_to_perturb", self.cfg.genes_to_perturb) if self.cfg.genes_to_perturb else None
        resolved_keep    = _resolve_ids("genes_to_keep",    self.cfg.genes_to_keep) if self.cfg.genes_to_keep else None

        ops = PerturbOps(
            genes_to_keep=resolved_keep,
            genes_to_perturb=resolved_perturb,
            perturb_fraction=self.cfg.perturb_fraction,
            rank_shift=self.cfg.perturb_rank_shift,
            top_k=self.cfg.top_k,
            pad_token_id=pad_id,
        )

        def _make_batch(slice_ds: Dataset) -> Dict[str, torch.Tensor]:
            lengths = [int(x) for x in slice_ds["length"]]
            max_len = min(max(lengths), model_input_size)

            slice_ds.set_format(type="torch")
            ids_list = slice_ds["input_ids"]

            input_ids = pad_tensor_list(ids_list, max_len, pad_id, model_input_size)

            # per-example attention mask (Geneformer’s gen_attention_mask)
            attn_rows = []
            for L in lengths:
                row = torch.zeros(max_len, dtype=torch.long)
                row[:L] = 1
                attn_rows.append(row)
            attention_mask = torch.stack(attn_rows, dim=0).to(input_ids.device)

            lengths_t = torch.as_tensor(lengths, device=input_ids.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "lengths": lengths_t}
        
                
        cos_sims = defaultdict(list)

        # Decide pipeline: if targets provided, use fast GROUP path
        run_group = bool(resolved_perturb) or (self.cfg.perturb_type in {"group", "delete", "overexpress"})
        set_resolved = set(resolved_perturb or [])

        if run_group:
            def _make_group_perturbation_batch(example):
                ids = example["input_ids"]
                if self.cfg.model_variant == "extended": # extended model conditions
                    spot_ids    = ids[:half_size]
                    neigh_ids   = ids[half_size:]
                    # indices in EACH half
                    spot_idx  = [i for i, tok in enumerate(spot_ids)  if int(tok) in set_resolved]
                    neigh_idx = [i for i, tok in enumerate(neigh_ids) if int(tok) in set_resolved]
                    example["perturb_index"] = (spot_idx if spot_idx else []) + \
                                            ([half_size + i for i in neigh_idx] if neigh_idx else [])
                    if self.cfg.perturb_type == "delete" or self.cfg.perturb_rank_shift == "delete":
                        # delete both halves’ indices (one call handles absolute indices)
                        return delete_indices(example)

                    elif self.cfg.perturb_type == "overexpress" or self.cfg.perturb_rank_shift == "overexpress":
                        # overexpress in BOTH halves: insert tokens at front of spot & neighbor halves
                        example["tokens_to_perturb"] = list(set_resolved)
                        return overexpress_tokens_extended(example,
                                                        half_size=half_size,
                                                        model_input_size=model_input_size)
                    else:
                        return example

                else: #spot mode
                    indices = [i for i, tok in enumerate(ids) if int(tok) in set_resolved]
                    example["perturb_index"] = indices if indices else [-100]
                    if self.cfg.perturb_type == "delete" or self.cfg.perturb_rank_shift == "delete":
                        return delete_indices(example)
                    elif self.cfg.perturb_type == "overexpress" or self.cfg.perturb_rank_shift == "overexpress":
                        example["tokens_to_perturb"] = list(set_resolved)
                        example = overexpress_tokens(example, model_input_size)
                        example["n_overflow"] = calc_n_overflow(
                            model_input_size, example["length"], list(set_resolved), indices
                        )
                        return example
                    else:
                        return example

            perturbed_data = ds.map(_make_group_perturbation_batch, num_proc=self.cfg.nproc)

            if (self.cfg.perturb_type == "overexpress" or self.cfg.perturb_rank_shift == "overexpress"):
                if self.cfg.model_variant == "extended":
                    # add per-half overflow vector if not present
                    if "n_overflow_halves" not in ds.column_names:
                        ds = ds.add_column("n_overflow_halves", perturbed_data["n_overflow_halves"])
                    ds = ds.map(lambda ex: truncate_by_n_overflow_extended(ex, half_size=half_size),
                                num_proc=self.cfg.nproc)
                else:
                    if "n_overflow" not in ds.column_names:
                        ds = ds.add_column("n_overflow", perturbed_data["n_overflow"])
                    ds = ds.map(truncate_by_n_overflow, num_proc=self.cfg.nproc)

            # Key used for cell entries
            cell_key = (tuple(resolved_perturb) if resolved_perturb and len(resolved_perturb) > 1
                        else (resolved_perturb[0] if resolved_perturb else -1), "cell_emb")

            total = len(ds)
            for start in tqdm(range(0, total, self.cfg.forward_batch_size), desc="Perturbing(group)"):
                end = min(start + self.cfg.forward_batch_size, total)
                mb = ds.select(range(start, end))
                pb = perturbed_data.select(range(start, end))

                with torch.no_grad():
                    batch_o = _make_batch(mb)
                    out_o = ModelAdapter.forward(model, batch_o)
                    hid_o = ModelAdapter.pick_layer(out_o, target_layer)

                    batch_p = _make_batch(pb)
                    out_p = ModelAdapter.forward(model, batch_p)
                    hid_p = ModelAdapter.pick_layer(out_p, target_layer)

                                    # --- Align ORIGINAL vs PERTURBED for token-wise comparisons ---
                    need_cell = self.cfg.emb_mode in {"cell", "cell_and_gene"}
                    need_gene = self.cfg.emb_mode in {"gene", "cell_and_gene"}

                    if self.cfg.perturb_rank_shift == "delete" or self.cfg.perturb_type == "delete":
                        # remove deleted indices from ORIGINAL so it aligns with PERTURBED
                        ids_mb = batch_o["input_ids"]
                        lengths_mb = batch_o["lengths"]
                        indices_to_remove = []
                        for b in range(ids_mb.shape[0]):
                            Lb = int(lengths_mb[b])
                            idxs = [i for i in range(Lb) if int(ids_mb[b, i]) in set_resolved]
                            indices_to_remove.append(idxs)

                        hid_o_aligned = remove_indices_per_example(hid_o, lengths_mb, indices_to_remove)
                        hid_p_aligned = hid_p
                        ids_for_pairs = batch_p["input_ids"]  # reflects deletions

                    elif (self.cfg.model_variant == "extended" and
                        (self.cfg.perturb_type == "overexpress" or self.cfg.perturb_rank_shift == "overexpress")):
                        # per-example [k_spot, k_neigh] computed at map time
                        k_halves = pb["n_overflow_halves"]  # e.g. [[k_spot, k_neigh], ...]
                        k_spot_vec = torch.as_tensor([int(k[0]) for k in k_halves], device=hid_p.device, dtype=torch.long)
                        k_neig_vec = torch.as_tensor([int(k[1]) for k in k_halves], device=hid_p.device, dtype=torch.long)

                        # drop front k in EACH half on PERTURBED side
                        hid_p_aligned = remove_front_per_example_halves(hid_p, k_spot_vec, k_neig_vec, half_size)
                        ids_for_pairs = remove_front_per_example_halves_2d(batch_p["input_ids"], k_spot_vec, k_neig_vec, half_size)

                        # ORIGINAL already truncated per-half earlier
                        hid_o_aligned = hid_o

                    elif (self.cfg.perturb_type == "overexpress" or self.cfg.perturb_rank_shift == "overexpress"):
                        # SPOT overexpress (single sequence) with per-example k if available
                        if "n_overflow" in pb.column_names:
                            k_list = pb["n_overflow"]
                            k_vec = torch.as_tensor(k_list, device=hid_p.device, dtype=torch.long)
                        else:
                            k_vec = torch.as_tensor([len(set_resolved)] * (end - start),
                                                    device=hid_p.device, dtype=torch.long)

                        hid_p_aligned = remove_front_per_example(hid_p, k_vec)
                        ids_for_pairs = remove_front_per_example_2d(batch_p["input_ids"], k_vec)
                        hid_o_aligned = hid_o

                    else:
                        hid_o_aligned = hid_o
                        hid_p_aligned = hid_p
                        ids_for_pairs = batch_p["input_ids"]
                    
                    # Ensure ids_for_pairs lives on the same device as sims_tok/hiddens
                    if ids_for_pairs.device != hid_p_aligned.device:
                        ids_for_pairs = ids_for_pairs.to(hid_p_aligned.device, non_blocking=True)

                    # token-wise cosine similarities [B, L’]
                    sims_tok = quant_cos_sims_tokenwise(hid_o_aligned, hid_p_aligned)

                    # CELL score = masked mean over non-padding
                    if need_cell:
                        mask = (ids_for_pairs != pad_id).to(sims_tok.dtype)  # [B, L’]
                        den = mask.sum(dim=1).clamp_min(1)
                        sims_cell = ((sims_tok * mask).sum(dim=1) / den).tolist()
                        cos_sims[cell_key].extend(sims_cell if isinstance(sims_cell, list) else [float(sims_cell)])

                    # GENE pairs from the same token-wise sims
                    if need_gene:
                        ids_np = ids_for_pairs.detach().cpu().numpy()
                        sims_np = sims_tok.detach().cpu().numpy()
                        B, Lp = ids_np.shape
                        perturbed_list = list(set_resolved) if set_resolved else []
                        for b in range(B):
                            for t in range(Lp):
                                affected_tok = int(ids_np[b, t])
                                if affected_tok == pad_id:
                                    continue
                                for pert_tok in (perturbed_list if perturbed_list else [affected_tok]):
                                    cos_sims[(pert_tok, affected_tok)].append(float(sims_np[b, t]))

                                        
                    if self.cfg.emb_mode in {"gene", "cell_and_gene"}:
                        # Align sequences token-wise
                        if self.cfg.perturb_rank_shift in {"delete"} or self.cfg.perturb_type == "delete":
                            ids_mb = batch_o["input_ids"]
                            lengths_mb = batch_o["lengths"]
                            indices_to_remove = []
                            for b in range(ids_mb.shape[0]):
                                Lb = int(lengths_mb[b])
                                idxs = [i for i in range(Lb) if int(ids_mb[b, i]) in set_resolved]
                                indices_to_remove.append(idxs)
                            hid_o_aligned = remove_indices_per_example(hid_o, lengths_mb, indices_to_remove)
                            hid_p_aligned = hid_p
                            ids_for_pairs = batch_p["input_ids"]  # perturbed ids reflect deletion
                        elif self.cfg.perturb_rank_shift in {"overexpress"} or self.cfg.perturb_type == "overexpress":
                            n = len(set_resolved)
                            hid_p_aligned = hid_p[:, n:, :].contiguous()
                            hid_o_aligned = hid_o
                            ids_for_pairs = batch_p["input_ids"][:, n:].contiguous()
                        else:
                            hid_o_aligned = hid_o
                            hid_p_aligned = hid_p
                            ids_for_pairs = batch_p["input_ids"]

                        # token-wise cosine [B, L]
                        sims_tok = quant_cos_sims_tokenwise(hid_o_aligned, hid_p_aligned)

                        ids_np = ids_for_pairs.detach().cpu().numpy()
                        sims_np = sims_tok.detach().cpu().numpy()
                        B, Lp = ids_np.shape
                        # For groups, we record separate pairs for each perturbed token in the set
                        perturbed_list = list(set_resolved) if set_resolved else []
                        for b in range(B):
                            for t in range(Lp):
                                affected_tok = int(ids_np[b, t])
                                if affected_tok == pad_id:
                                    continue
                                for pert_tok in (perturbed_list if perturbed_list else [affected_tok]):
                                    cos_sims[(pert_tok, affected_tok)].append(float(sims_np[b, t]))

            out_prefix = str(Path(output_directory) / output_prefix)
            write_perturbation_dictionary(cos_sims, out_prefix)
            return f"{out_prefix}_raw.pickle"

        # ---- SINGLE fallback (only when no explicit target list) ----
        iterator = batcher.iter(ds, with_indices=False, progress_desc="Perturbing(single)")
        for batch in iterator:
            with torch.no_grad():
                out = ModelAdapter.forward(model, batch)
                hidden = ModelAdapter.pick_layer(out, target_layer)
            cell_emb = ModelAdapter.pool_mean(hidden, batch["lengths"], exclude_cls=True, exclude_eos=True)

            tokens_this_batch = list(ops.iter_single_tokens(batch["input_ids"]))
            for gene_tok in tokens_this_batch:
                perturbed_ids = ops.apply_single(batch["input_ids"], gene_tok)
                out_p = ModelAdapter.forward(model, {"input_ids": perturbed_ids, "attention_mask": batch["attention_mask"]})
                hidden_p = ModelAdapter.pick_layer(out_p, target_layer)
                cell_emb_p = ModelAdapter.pool_mean(hidden_p, batch["lengths"], exclude_cls=True, exclude_eos=True)
                sims = torch.nn.functional.cosine_similarity(cell_emb, cell_emb_p, dim=-1).tolist()
                cos_sims[(int(gene_tok), "cell_emb")].extend(sims if isinstance(sims, list) else [float(sims)])

        out_prefix = str(Path(output_directory) / output_prefix)
        write_perturbation_dictionary(cos_sims, out_prefix)
        return f"{out_prefix}_raw.pickle"
