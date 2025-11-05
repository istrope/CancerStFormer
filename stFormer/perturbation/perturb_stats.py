from __future__ import annotations

import os
import io
import json
import math
import pickle
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # optional
    GaussianMixture = None  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------------
# Utilities
# ----------------------------

def _is_pickle_path(p: Union[str, os.PathLike]) -> bool:
    s = str(p)
    return s.endswith(".pkl") or s.endswith(".pickle")

def _load_raw_pickle(path: Union[str, os.PathLike]) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def _iter_raw_pickles(path_or_dir: Union[str, os.PathLike]) -> Iterable[dict]:
    p = Path(path_or_dir)
    if p.is_file():
        yield _load_raw_pickle(p)
    else:
        for fp in sorted(p.glob("*_raw.pickle")):
            yield _load_raw_pickle(fp)

def _bh_fdr(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray([v if (v is not None and np.isfinite(v)) else 1.0 for v in pvals], dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        q[i] = min(prev, ranked[i] * n / rank)
        prev = q[i]
    out = np.empty_like(q)
    out[order] = q
    return out.tolist()

def _mannwhitney_u_vs_zero(x: np.ndarray) -> float:
    """Nonparametric test: compare distribution of x against zero values using a rank strategy.
    This approximates a two-sided test vs. a degenerate distribution at 0.
    Returns a p-value in [0,1]."""
    if x.size == 0:
        return 1.0
    # ranks of |x| with sign handling
    ranks = np.argsort(np.argsort(np.abs(x))) + 1
    # Effect sign: positive if median(x) > 0
    sign = 1 if np.median(x) >= 0 else -1
    u = float(np.sum(sign * ranks))
    # Normal approximation
    mu = 0.0
    sigma = math.sqrt(np.sum(ranks**2) / 12.0 + 1e-8)
    z = 0.0 if sigma == 0 else (u - mu) / sigma
    # two-sided
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5*(1.0 + erf(abs(z)/sqrt(2.0))))
    return float(np.clip(p, 0.0, 1.0))

def _safe_mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))

def _merge_raw_dicts(dicts: List[dict]) -> dict:
    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                merged[k].extend([float(x) for x in v])
            else:
                merged[k].append(float(v))
    return dict(merged)

def _split_cell_gene_subdicts(raw: dict) -> Tuple[dict, dict]:
    """Return (cell_dict, gene_dict)."""
    cell = {k: v for k, v in raw.items() if isinstance(k, tuple) and len(k) == 2 and k[1] == "cell_emb"}
    gene = {k: v for k, v in raw.items() if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], int) and isinstance(k[1], int)}
    return cell, gene

def _token_to_name(tok: int, tok2name: Optional[Dict[int, str]]) -> str:
    if tok2name is None:
        return str(tok)
    return tok2name.get(int(tok), str(tok))


# ----------------------------
# Stats implementations
# ----------------------------

def _mode_aggregate_data(raw_merged: dict,
                         tok2name: Optional[Dict[int, str]],
                         out_csv: Union[str, os.PathLike]) -> str:
    """Aggregate cell-level shifts for a single perturbation."""
    cell_dict, _ = _split_cell_gene_subdicts(raw_merged)
    rows = []
    for key, vals in cell_dict.items():
        pert = key[0]  # int or tuple
        m, s = _safe_mean_std(vals)
        rows.append({
            "Perturbed": _token_to_name(pert, tok2name) if not isinstance(pert, tuple) else ",".join(_token_to_name(t, tok2name) for t in pert),
            "Cosine_sim_mean": m,
            "Cosine_sim_stdev": s,
            "N_Detections": len(vals),
        })
    df = pd.DataFrame(rows).sort_values("Cosine_sim_mean", ascending=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return str(out_csv)


def _mode_aggregate_gene_shifts(raw_merged: dict,
                                tok2name: Optional[Dict[int, str]],
                                out_csv: Union[str, os.PathLike],
                                tok2ens: Optional[Dict[int, str]] = None) -> str:
    """
    Columns:
    Perturbed, Gene_name, Ensembl_ID, Affected, Affected_gene_name, Affected_Ensembl_ID,
    Cosine_sim_mean, Cosine_sim_stdev, N_Detections
    """
    _, gene_dict = _split_cell_gene_subdicts(raw_merged)
    rows = []
    for (pert_tok, aff_tok), vals in gene_dict.items():
        mean_v, std_v = _safe_mean_std(vals)
        pert_name = _token_to_name(pert_tok, tok2name)
        aff_name  = _token_to_name(aff_tok, tok2name)
        pert_ens = tok2ens.get(int(pert_tok), str(pert_tok)) if tok2ens else str(pert_tok)
        aff_ens  = tok2ens.get(int(aff_tok), str(aff_tok)) if tok2ens else str(aff_tok)
        rows.append({
            "Perturbed": pert_name,  # display name of perturbed gene
            "Gene_name": pert_name,
            "Ensembl_ID": pert_ens,
            "Affected": aff_name,    # display name of affected gene
            "Affected_gene_name": aff_name,
            "Affected_Ensembl_ID": aff_ens,
            "Cosine_sim_mean": mean_v,
            "Cosine_sim_stdev": std_v,
            "N_Detections": len(vals),
        })
    # Enforce exact column order
    cols = [
        "Perturbed","Gene_name","Ensembl_ID",
        "Affected","Affected_gene_name","Affected_Ensembl_ID",
        "Cosine_sim_mean","Cosine_sim_stdev","N_Detections",
    ]
    df = pd.DataFrame(rows, columns=cols).sort_values(["Cosine_sim_mean"],ascending=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return str(out_csv)


def _mode_vs_null(raw_merged: dict,
                  null_merged: dict,
                  tok2name: Optional[Dict[int, str]],
                  out_csv: Union[str, os.PathLike]) -> str:
    """Compare cell shifts vs a NULL distribution (separate raw dict set)."""
    cell_t, _ = _split_cell_gene_subdicts(raw_merged)
    cell_n, _ = _split_cell_gene_subdicts(null_merged)
    # assume single cell key per run (typical)
    rows = []
    for k, vals_t in cell_t.items():
        vals_n = []
        # If null has matching key, use it; else flatten all
        if k in cell_n:
            vals_n = cell_n[k]
        else:
            for _, v in cell_n.items():
                vals_n.extend(v)
        # effect size & test
        mt, st = _safe_mean_std(vals_t)
        mn, sn = _safe_mean_std(vals_n)
        # Two-sample MWU approximate p-value by shift vs zero after centering on null mean
        x = np.asarray(vals_t, dtype=float) - (mn if np.isfinite(mn) else 0.0)
        pval = _mannwhitney_u_vs_zero(x)
        rows.append({
            "Perturbed": _token_to_name(k[0], tok2name) if not isinstance(k[0], tuple) else ",".join(_token_to_name(t, tok2name) for t in k[0]),
            "Test_mean": mt, "Test_stdev": st, "Test_n": len(vals_t),
            "Null_mean": mn, "Null_stdev": sn, "Null_n": len(vals_n),
            "p_value": pval,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_value"] = _bh_fdr(df["p_value"].tolist())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return str(out_csv)


def _mode_goal_state_shift(raw_by_state: Dict[str, dict],
                           state_cfg: dict,
                           tok2name: Optional[Dict[int, str]],
                           out_csv: Union[str, os.PathLike]) -> str:
    """State-aware goal shift: compare start vs goal (and optional alt states)."""
    # Expect keys: 'start_state','goal_state', optional 'alt_states' (list)
    start_key = state_cfg.get("start_state")
    goal_key  = state_cfg.get("goal_state")
    alt_list  = state_cfg.get("alt_states", []) or []
    if start_key is None or goal_key is None:
        raise ValueError("goal_state_shift requires 'start_state' and 'goal_state' in cell_states_to_model")

    # Merge cell dicts per state
    def merge_state_cell(d):
        cell, _ = _split_cell_gene_subdicts(d)
        all_vals = []
        for _, v in cell.items():
            all_vals.extend(v)
        return np.asarray(all_vals, dtype=float)

    x_start = merge_state_cell(raw_by_state[start_key]) if start_key in raw_by_state else np.asarray([], dtype=float)
    x_goal  = merge_state_cell(raw_by_state[goal_key]) if goal_key in raw_by_state else np.asarray([], dtype=float)
    p_start_vs_goal = _mannwhitney_u_vs_zero(x_start - (np.nanmean(x_goal) if x_goal.size else 0.0))

    rows = [{
        "Start_state": start_key,
        "Goal_state": goal_key,
        "Start_mean": float(np.nanmean(x_start)) if x_start.size else float("nan"),
        "Goal_mean": float(np.nanmean(x_goal)) if x_goal.size else float("nan"),
        "Start_n": int(x_start.size), "Goal_n": int(x_goal.size),
        "p_value": p_start_vs_goal,
    }]

    # Optional alt comparisons
    for alt in alt_list:
        if alt not in raw_by_state:
            continue
        x_alt = merge_state_cell(raw_by_state[alt])
        p_goal_vs_alt = _mannwhitney_u_vs_zero(x_goal - (np.nanmean(x_alt) if x_alt.size else 0.0))
        rows.append({
            "Alt_state": alt,
            "Alt_mean": float(np.nanmean(x_alt)) if x_alt.size else float("nan"),
            "Alt_n": int(x_alt.size),
            "Alt_p_value": p_goal_vs_alt,
        })

    df = pd.DataFrame(rows)
    # FDR over available p-values columns
    pcols = [c for c in df.columns if c.endswith("_value")]
    if pcols:
        allp = []
        for c in pcols:
            allp.extend(df[c].fillna(1.0).tolist())
        q = _bh_fdr(allp)
        # split back
        i = 0
        for c in pcols:
            n = df[c].shape[0]
            df[c.replace("_value", "_qvalue")] = q[i:i+n]
            i += n

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return str(out_csv)


def _mode_mixture_model(raw_merged: dict,
                        tok2name: Optional[Dict[int, str]],
                        out_csv: Union[str, os.PathLike]) -> str:
    """Two-component GMM on per-gene means, label 'impact' component and % impact."""
    if GaussianMixture is None:
        raise ImportError("scikit-learn is required for mixture_model mode but is not available.")

    _, gene_dict = _split_cell_gene_subdicts(raw_merged)
    # compute per-gene means (collapse across cells)
    per_gene = [(k[1], float(np.nanmean(v))) for k, v in gene_dict.items()]  # (affected_token, mean)
    if not per_gene:
        df = pd.DataFrame(columns=["Affected", "Mean", "Impact_component", "Impact_component_percent"])
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return str(out_csv)

    tokens, means = zip(*per_gene)
    X = np.array(means, dtype=float).reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gmm.fit(X)
    labels = gmm.predict(X)
    # define "impact" as component with lower mean (stronger negative shift) or higher |mean|
    comp_means = gmm.means_.flatten()
    impact_label = int(np.argmin(comp_means))  # lower mean = more negative shift
    impact_mask = (labels == impact_label)
    impact_pct = float(np.mean(impact_mask) * 100.0)

    rows = []
    for tok, m, lab in zip(tokens, means, labels):
        rows.append({
            "Affected": _token_to_name(tok, tok2name),
            "Mean": float(m),
            "Impact_component": int(lab),
        })
    df = pd.DataFrame(rows)
    df["Impact_component_percent"] = impact_pct
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return str(out_csv)

@dataclass
class PerturberStats:
    mode: str
    top_k: Optional[int] = None
    nperms: Optional[int] = None
    fdr_alpha: float = 0.05
    min_cells: int = 1
    # dictionaries for mapping
    gene_token_id_dict: Optional[Dict[str, int]] = None  # ENSEMBL->token
    gene_id_name_dict: Optional[Dict[str, str]] = None   # ENSEMBL->symbol
    cell_states_to_model: Optional[dict] = None          # {"state_key":..., "start_state":..., "goal_state":..., "alt_states":[...]}

    def _tok_maps(self):
        """Return (tok2name, tok2ens) if dicts provided; else (None, None)."""
        if not self.gene_token_id_dict or not self.gene_id_name_dict:
            return None, None
        tok2name = {}
        tok2ens = {}
        for ens, tok in self.gene_token_id_dict.items():
            try:
                itok = int(tok)
            except Exception:
                continue
            tok2ens[itok] = str(ens)
            tok2name[itok] = str(self.gene_id_name_dict.get(ens, str(ens)))
        return tok2name, tok2ens

    def _load_and_merge(self, path_or_dir: Union[str, os.PathLike]) -> dict:
        dicts = list(_iter_raw_pickles(path_or_dir))
        if not dicts:
            return {}
        return _merge_raw_dicts(dicts)

    def _load_by_state(self, base_dir: Union[str, os.PathLike]) -> Dict[str, dict]:
        """Load per-state raw dicts from subdirectories located under base_dir."""
        assert self.cell_states_to_model, "cell_states_to_model must be provided for goal_state_shift"
        state_key = self.cell_states_to_model.get("state_key", "state")
        states = []
        for k, v in self.cell_states_to_model.items():
            if k == "state_key" or v is None:
                continue
            if isinstance(v, list):
                states.extend(v)
            else:
                states.append(v)
        out = {}
        base = Path(base_dir)
        for s in states:
            # expect pickles under base_dir/<state>/..._raw.pickle
            sdir = base / str(s)
            if sdir.is_dir():
                out[str(s)] = self._load_and_merge(sdir)
        return out

    # main entry
    def compute_stats(self,
                      input_path_or_dir: str,
                      null_path_or_dir: Optional[str],
                      output_directory: str,
                      output_prefix: str) -> str:
        tok2name, tok2ens = self._tok_maps()

        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "aggregate_data":
            merged = self._load_and_merge(input_path_or_dir)
            out_csv = out_dir / f"{output_prefix}_aggregate_data.csv"
            return _mode_aggregate_data(merged, tok2name, out_csv)

        if self.mode == "aggregate_gene_shifts":
            merged = self._load_and_merge(input_path_or_dir)
            out_csv = out_dir / f"{output_prefix}_aggregate_gene_shifts.csv"
            return _mode_aggregate_gene_shifts(merged, tok2name, out_csv, tok2ens)

        if self.mode == "vs_null":
            if null_path_or_dir is None:
                raise ValueError("vs_null requires null_path_or_dir")
            merged_t = self._load_and_merge(input_path_or_dir)
            merged_n = self._load_and_merge(null_path_or_dir)
            out_csv = out_dir / f"{output_prefix}_vs_null.csv"
            return _mode_vs_null(merged_t, merged_n, tok2name, out_csv)

        if self.mode == "goal_state_shift":
            if not self.cell_states_to_model:
                raise ValueError("goal_state_shift requires cell_states_to_model")
            # Expect base_dir containing subdirs per state (names matching the config values)
            by_state = self._load_by_state(input_path_or_dir)
            out_csv = out_dir / f"{output_prefix}_goal_state_shift.csv"
            return _mode_goal_state_shift(by_state, self.cell_states_to_model, tok2name, out_csv)

        if self.mode == "mixture_model":
            merged = self._load_and_merge(input_path_or_dir)
            out_csv = out_dir / f"{output_prefix}_mixture_model.csv"
            return _mode_mixture_model(merged, tok2name, out_csv)

        raise ValueError(f"Unknown mode: {self.mode}")
