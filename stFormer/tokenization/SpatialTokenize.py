#!/usr/bin/env python3
"""
spatial_tokenization.py

This module provides comprehensive tools for spatial transcriptomics tokenization:
  1. Computing gene expression medians via T‑Digest (streaming .loom or chunked .h5ad).
  2. Merging multiple T‑Digest pickles into a unified median dictionary.
  3. Creating a token dictionary from median values.
  4. Tokenizing Visium .h5ad or .loom files with two modes:
     - "spot": per-spot top-ranked gene tokens (up to `gene_length`).
     - "neighborhood": concatenated top `gene_length` spot tokens + top `gene_length` neighbor tokens.
  5. Outputs a Hugging Face Dataset ready for downstream tasks.

All functionality is in importable classes/functions; no main-level execution.

Example usage:
    from pathlib import Path
    from spatial_tokenization import (
        MedianEstimator,
        merge_tdigest_dicts,
        create_token_dictionary,
        SpatialTokenizer
    )

    # 1. Compute medians
    genes = ['GeneA', 'GeneB']
    est = MedianEstimator(genes)
    est.compute_tdigests(Path('sample.loom'))
    med_dict = est.get_median_dict()

    # 2. Merge digests (optional)
    merged = merge_tdigest_dicts(Path('digests'))
    est.merge_with(merged)
    med_dict = est.get_median_dict()

    # 3. Build token dict
    token_dict = create_token_dictionary(med_dict)

    # 4a. Spot-only tokenization (top 2048 genes)
    tok_spot = SpatialTokenizer(
        mode='spot',
        gene_length=2048,
        custom_meta={'sample_id':'sample'},
        nproc=4,
        gene_median_file=Path('gene_median_dict.pickle'),
        token_dict_file=Path('token_dict.pickle')
    )
    tok_spot.tokenize(
        data_dir=Path('/input'),
        out_dir=Path('/output'),
        prefix='visium_spot',
        file_format='h5ad'
    )

    # 4b. Neighborhood tokenization (2048 spot + 2048 neighbor = 4096 tokens)
    tok_nei = SpatialTokenizer(
        mode='neighborhood',
        gene_length=2048,
        custom_meta={'sample_id':'sample'},
        nproc=4,
        gene_median_file=Path('gene_median_dict.pickle'),
        token_dict_file=Path('token_dict.pickle')
    )
    tok_nei.tokenize(
        data_dir=Path('/input'),
        out_dir=Path('/output'),
        prefix='visium_neighborhood',
        file_format='loom'
    )
"""

import math
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal

import numpy as np
import anndata as ad
import scanpy as sc
import loompy
import scipy.sparse as sp
from scipy.spatial import Delaunay
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import crick.tdigest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Median Estimation -------------------- #
class MedianEstimator:
    """
    Stream through .loom or chunk through .h5ad to update per-gene T‑Digests.
    """
    def __init__(self, gene_list: List[str], norm_target: float = 1e4):
        self.gene_list = gene_list
        self.norm_target = norm_target
        self.tdigests: Dict[str, crick.tdigest.TDigest] = {g: crick.tdigest.TDigest() for g in gene_list}

    def compute_tdigests(self, file_path: Path, chunk: int = 1000) -> np.ndarray:
        sfx = file_path.suffix.lower()
        if sfx == '.loom':
            with loompy.connect(str(file_path)) as ds:
                var = ds.ra.get('ensembl_id')
                coding = [i for i, g in enumerate(var) if g in self.gene_list]
                totals = np.zeros(ds.shape[1], float)
                for _, _, view in ds.scan(items=coding, axis=0):
                    totals += view.view.sum(axis=0)
                for idx, _, view in tqdm(ds.scan(items=coding, axis=0), total=len(coding), desc='TDigest Loom'):
                    g = var[idx]
                    vals = view.view.flatten() / totals * self.norm_target
                    vals = vals[vals > 0]
                    if vals.size:
                        self.tdigests[g].update(vals)
                return totals
        elif sfx == '.h5ad':
            adata = ad.read_h5ad(str(file_path), backed='r')
            var = adata.var['ensembl_id'] if 'ensembl_id' in adata.var else adata.var_names
            coding = [i for i, g in enumerate(var) if g in self.gene_list]
            N = adata.n_obs
            totals = np.zeros(N, float)
            idxs = np.arange(N)
            for b in np.array_split(idxs, int(np.ceil(N / chunk))):
                X = adata[b, coding].X
                X = X.toarray() if sp.issparse(X) else X
                totals[b] = X.sum(axis=1)
            for b in np.array_split(idxs, int(np.ceil(N / chunk))):
                X = adata[b, coding].X
                X = X.toarray() if sp.issparse(X) else X
                Xn = X / totals[b][:, None] * self.norm_target
                for j, gi in enumerate(coding):
                    vals = Xn[:, j]
                    vals = vals[vals > 0]
                    if vals.size:
                        self.tdigests[var[gi]].update(vals)
            adata.file.close()
            return totals
        else:
            raise ValueError('Expect .loom or .h5ad')

    def get_median_dict(self, detected_only: bool = True) -> Dict[str, float]:
        med = {g: td.quantile(0.5) for g, td in self.tdigests.items()}
        if detected_only:
            med = {g: m for g, m in med.items() if not math.isnan(m)}
        return med

    def merge_with(self, other: Dict[str, crick.tdigest.TDigest]) -> None:
        for g, td in other.items():
            if g in self.tdigests:
                self.tdigests[g].merge(td)
            else:
                self.tdigests[g] = td

# -------------------- Merge Utility -------------------- #
def merge_tdigest_dicts(directory: Path, pattern: str = '*.pickle') -> Dict[str, crick.tdigest.TDigest]:
    merged: Dict[str, crick.tdigest.TDigest] = {}
    for f in directory.glob(pattern):
        with open(f, 'rb') as fp:
            d = pickle.load(fp)
        for g, td in d.items():
            if g in merged:
                merged[g].merge(td)
            else:
                merged[g] = td
    return merged

# -------------------- Token Dictionary -------------------- #
def create_token_dictionary(median_dict: Dict[str, float], reserved: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    if reserved is None:
        reserved = {'<pad>': 0, '<mask>': 1}
    genes = [g for g, m in median_dict.items() if not math.isnan(m)]
    token_dict = reserved.copy()
    for i, g in enumerate(genes, start=len(reserved)):
        token_dict[g] = i
    return token_dict


def build_custom_tokenizer(
    token_dict_path: str,
    pad_token: str = "<pad>",
    mask_token: str = "<mask>",
    mode: Literal['spot', 'neighborhood'] = 'spot'
    ) -> PreTrainedTokenizerFast:
    """
    Build a HuggingFace Fast Tokenizer from a gene->ID pickle vocab,
    using a simple WordLevel model (no merges file needed).
    """
    if mode == 'spot':
        max_length = 2048
    else: #mode neighborhood
        max_length = 4096
    # 1) load your vocab dict: str->int
    with open(token_dict_path, "rb") as f:
        vocab = pickle.load(f)

    # 2) build a tokenizers.WordLevel model around it
    wordlevel = models.WordLevel(vocab=vocab, unk_token="<unk>")
    tokenizer_obj = Tokenizer(wordlevel)
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_obj.decoder      = decoders.WordPiece()  # join on nothing

    # 3) wrap it in the HF Fast API
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        model_max_length = max_length,
        unk_token="<unk>",
        pad_token=pad_token,
        mask_token=mask_token,
        cls_token='<cls>',
        sep_token="<sep>"
    )
# -------------------- Tokenization Helpers -------------------- #
def ensure_graph(adata: ad.AnnData, key: str = 'spatial') -> None:
    # if there's already a nontrivial connectivity, leave it alone
    if 'spatial_connectivities' in adata.obsp and adata.obsp['spatial_connectivities'].nnz > 0:
        return

    coords = adata.obsm[key]
    if coords is None:
        raise KeyError(f'Missing spatial coords at adata.obsm["{key}"]')

    # build Delaunay graph
    tri = Delaunay(coords)
    n = coords.shape[0]
    rows, cols = [], []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                rows.extend([simplex[i], simplex[j]])
                cols.extend([simplex[j], simplex[i]])
    adata.obsp['spatial_connectivities'] = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    )
    logger.info(f'Delaunay graph built; nnz = {adata.obsp["spatial_connectivities"].nnz}')


def rank_genes(vec: np.ndarray, toks: np.ndarray) -> np.ndarray:
    idx = np.argsort(-vec)
    return toks[idx]

def check_format(adata: ad.AnnData) -> Dict[str, object]:
    res = {'valid': True, 'messages': []}
    if 'ensembl_id' not in adata.var:
        raise AttributeError('Missing ensembl_id column in var')
    if 'n_counts' not in adata.obs:
        raise AttributeError('Missing n_counts in metadata')
    return res

# -------------------- Spatial Tokenizer -------------------- #
class SpatialTokenizer:
    """
    Tokenize in 'spot' or 'neighborhood' modes:
      - 'spot': top gene_length tokens from spot only.
      - 'neighborhood': top gene_length spot + top gene_length neighbor tokens.
    """
    def __init__(
        self,
        mode: Literal['spot', 'neighborhood'] = 'spot',
        gene_length: int = 2048,
        custom_meta: Optional[Dict[str, str]] = None,
        nproc: int = 1,
        down_pct: Optional[float] = None,
        down_seed: Optional[int] = None,
        gene_median_file: Path = Path('gene_median_dict.pickle'),
        token_dict_file: Path = Path('token_dict.pickle'),
        chunk: int = 512,
        target: float = 1e4,
    ):
        self.mode = mode
        self.gene_length = gene_length
        self.meta_map = custom_meta or {}
        self.nproc = nproc
        self.down_pct = down_pct
        self.down_seed = down_seed
        self.chunk = chunk
        self.target = target

        with open(gene_median_file, 'rb') as f:
            self.med = pickle.load(f)
        with open(token_dict_file, 'rb') as f:
            self.tok = pickle.load(f)
        self.genes = list(self.med.keys())
        #self.tokenizer = build_custom_tokenizer(token_dict_file)

    def tokenize(
        self,
        data_dir: Path,
        out_dir: Path,
        prefix: str,
        file_format: Literal['h5ad', 'loom'] = 'h5ad',
    ) -> None:
        paths = list(data_dir.glob(f'*.{file_format}'))
        cells_all, nei_all, meta_all = [], [], {v: [] for v in self.meta_map.values()}

        for p in paths:
            ad = sc.read_h5ad(str(p)) if file_format == 'h5ad' else sc.read_loom(str(p), sparse=True)
            ensure_graph(ad)
            c, n, m = self._tokenize_ad(ad)
            cells_all += c
            nei_all += n
            for ik, ok in self.meta_map.items():
                meta_all[ok] += m.get(ik, [])

        ds = self._make_ds(cells_all, nei_all, meta_all)
        out_path = out_dir / f"{prefix}.dataset"
        ds.save_to_disk(str(out_path))
        logger.info(f'Saved → {out_path}')

    def _tokenize_ad(
        self,
        ad: ad.AnnData,
    ):
        ad = ad[ad.obs['n_counts'] > 0]
        if self.down_pct:
            idxs = np.arange(ad.n_obs)
            sel, _ = train_test_split(idxs, test_size=self.down_pct, random_state=self.down_seed)
            ad = ad[sel, :]

        check_format(ad)
        logger.info(f'Passed Anndata Check: n_counts and ensembl_id')

        var = ad.var['ensembl_id'] if 'ensembl_id' in ad.var else ad.var_names
        idxs = [i for i, g in enumerate(var) if g in self.genes]
        tokens = np.array([self.tok[var[i]] for i in idxs])
        norms = np.array([self.med[var[i]] for i in idxs])
        A = ad.obsp['spatial_connectivities']
        meta = {ik: ad.obs[ik].tolist() for ik in self.meta_map}

        # pre-load full gene matrix
        full_X = ad[:, idxs].X
        full_X = full_X.toarray() if sp.issparse(full_X) else full_X

        c_out, n_out = [], []
        for start in range(0, ad.n_obs, self.chunk):
            batch = np.arange(start, min(start + self.chunk, ad.n_obs))
            X_spot = full_X[batch, :]
            ncnt = ad.obs['n_counts'].values[batch][:, None]

            spot = (X_spot / ncnt * self.target) / norms
            nei_mat = (A[batch, :].dot(full_X) / ncnt * self.target) / norms

            for i, r in enumerate(spot):
                spot_tokens = rank_genes(r[r > 0], tokens[r > 0])[:self.gene_length]
                if self.mode == 'spot':
                    c_out.append(spot_tokens)
                else:  # neighborhood
                    mask = nei_mat[i] > 0
                    nei_tokens = rank_genes(nei_mat[i][mask], tokens[mask])[:self.gene_length]


                    combined = np.concatenate([spot_tokens, nei_tokens])
                    c_out.append(combined)

        return c_out, n_out, meta

    def _make_ds(
        self,
        cells: List[np.ndarray],
        nei: List[np.ndarray],
        meta: Dict[str, List],
    ) -> Dataset:
        data: Dict[str, List] = {'input_ids': cells}
        # add metadata columns, converting all values to strings for Arrow
        for k, v in meta.items():
            data[k] = [str(x) for x in v]

        # batch into HF Datasets
        batches, keys, vals = [], list(data.keys()), list(data.values())
        for i in range(0, len(vals[0]), 10000):
            sub = {k: v[i:i+10000] for k, v in zip(keys, vals)}
            batches.append(Dataset.from_dict(sub))
        ds = concatenate_datasets(batches)

        ds = ds.map(lambda ex: {'length': len(ex['input_ids'])}, num_proc=self.nproc)
        return ds

