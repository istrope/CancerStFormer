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
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import crick.tdigest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Median Estimation 
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

#  Merge Utility  #
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

#  Token Dictionary 
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
#  Tokenization Helpers 
def ensure_graph(adata: ad.AnnData, key: str = 'spatial') -> None:
    # if there's already a nontrivial connectivity, leave it alone
    if 'spatial_connectivities' in adata.obsp and adata.obsp['spatial_connectivities'].nnz > 0:
        return
    
    if key not in adata.obsm:
        raise KeyError(f'Missing spatial coords at adata.obsm["{key}"]')
    
    coords = adata.obsm[key]
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


def rank_genes_vectorized(expr_matrix: np.ndarray,
                          token_array: np.ndarray,
                          top_k: int):
    """
    expr_matrix: shape (batch_size, n_genes), already normalized
    token_array: shape (n_genes,)
    returns: array of shape (batch_size, top_k) of token IDs
    """
    n_cells,n_genes = expr_matrix.shape
    if n_genes <= top_k:
        idx_sorted = np.argsort(-expr_matrix,axis=1)
        return(token_array[idx_sorted])
    # find the top_k gene‐indices *unsorted* per row
    idx_part = np.argpartition(-expr_matrix, top_k, axis=1)[:, :top_k]
    
    # gather their expression values
    row_idx = np.arange(expr_matrix.shape[0])[:, None]
    top_vals = expr_matrix[row_idx, idx_part]
    
    # now sort within each row by descending value
    order = np.argsort(-top_vals, axis=1)
    sorted_idx = idx_part[row_idx, order]
    
    # map back to tokens
    return token_array[sorted_idx]


def check_format(adata: ad.AnnData):
    if 'ensembl_id' not in adata.var:
        raise ValueError('ensembl_id not in adata.var for file {file}, ' \
        'please rename ensembl column or convert names ensembl')
    if 'n_counts' not in adata.obs:
        raise ValueError('missing n_counts value from adata.var for file {file}')

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

    def _process_file(self, path: Path):
        ad = sc.read_h5ad(str(path)) if str(path).endswith('.h5ad') else sc.read_loom(str(path))
        if self.mode == "neighborhood":
            ensure_graph(ad)
        return self._tokenize_ad(ad)

    def tokenize(self, data_dir: Path, out_dir: Path, prefix: str) -> None:
        paths = list(data_dir.glob("*.h5ad")) + list(data_dir.glob("*.loom"))
        if not paths:
            raise ValueError(f"No .h5ad/.loom files found in {data_dir!r}")
        cells_all, meta_all = [], {v: [] for v in self.meta_map.values()}

        def _handle_result(c, m):
            cells_all.extend(c)
            for ik, ok in self.meta_map.items():
                meta_all[ok].extend(m.get(ik, []))

        for p in tqdm(paths, desc='Tokenizing Anndata', total=len(paths)):
            self.file = p
            c, m = self._process_file(p)
            _handle_result(c, m)

        logger.info('Making Hugging Face Dataset')
        ds = self._make_ds(cells_all, meta_all)
        out_path = out_dir / f"{prefix}.dataset"
        ds.save_to_disk(str(out_path))
        logger.info(f'Saved → {out_path}')

    def _tokenize_ad(self, ad: sc.AnnData):
        # Ensure n_counts field exists
        if 'n_counts' not in ad.obs:
            ad.obs['n_counts'] = ad.X.sum(axis=1).A1 if sp.issparse(ad.X) else ad.X.sum(axis=1)
        # Filter zero-count cells
        ad.obs['n_counts'] = ad.obs['n_counts'].astype(int)
        ad = ad[ad.obs['n_counts'] > 0]
        ad.var_names_make_unique()
        # Optional downsampling
        if self.down_pct:
            idxs = np.arange(ad.n_obs)
            sel, _ = train_test_split(idxs, test_size=self.down_pct, random_state=self.down_seed)
            ad = ad[sel, :]

        check_format(ad)

        var = ad.var['ensembl_id']
        idxs = [i for i, g in enumerate(var) if g in self.genes]
        tokens = np.array([self.tok[var.iloc[i]] for i in idxs])
        norms = np.array([self.med[var.iloc[i]] for i in idxs])

        if self.mode == 'neighborhood':
            A = ad.obsp['spatial_connectivities']

        # Load expression matrix
        X = ad[:, idxs].X
        X = X.toarray() if sp.issparse(X) else X
        
        # Detect raw vs normalized: integer row sums == raw counts
        row_sums = X.sum(axis=1)
        is_raw = np.allclose(row_sums, np.round(row_sums))
        #logger.info(f"[Auto-detect] Input appears to be {'raw counts' if is_raw else 'normalized values'}.")

        n_cells = X.shape[0]
        c_out = []
        ncnt = ad.obs['n_counts'].values[:,None]
        
        for start in range(0, n_cells, self.chunk):
            end = min(start + self.chunk, n_cells)
            X_batch = X[start:end, :]
            ncnt_batch = ncnt[start:end, :]

            # Spot normalization
            if is_raw:
                spot_norm = (X_batch / ncnt_batch * self.target) / norms
            else:
                spot_norm = X_batch / norms

            spot_tok = rank_genes_vectorized(spot_norm, tokens, self.gene_length)

            if self.mode == 'spot':
                c_out.extend(spot_tok.tolist())
            else:
                # Neighborhood normalization
                A_batch = A[start:end, :]
                nei_batch = A_batch.dot(X)
                if is_raw:
                    nei_norm = (nei_batch / ncnt_batch * self.target) / norms
                else:
                    nei_norm = nei_batch / norms
                nei_tok = rank_genes_vectorized(nei_norm, tokens, self.gene_length)
                combined = np.concatenate([spot_tok, nei_tok], axis=1)
                c_out.extend(combined.tolist())
        
        meta = {}
        for ik in self.meta_map.keys():
            if ik not in ad.obs.columns: #handles metadata that doesn't exist
                meta[ik] = [np.nan] *ad.shape[0]
                logging.info(f'metadata value: {ik} not present for file {self.file}')
            else:
                meta[ik] = ad.obs[ik].astype(str).tolist()
        #meta = {ik: ad.obs[ik].astype(str).tolist() for ik in self.meta_map}
        return c_out, meta

    def _make_ds(self, cells: List[np.ndarray], meta: Dict[str, List]) -> Dataset:
        data = {"input_ids": cells, **{k: [str(x) for x in v] for k, v in meta.items()}}
        ds = Dataset.from_dict(data)

        def add_length(batch):
            return {"length": [len(ids) for ids in batch["input_ids"]]}

        ds = ds.map(add_length, batched=True, batch_size=self.chunk, num_proc=self.nproc)
        return ds


