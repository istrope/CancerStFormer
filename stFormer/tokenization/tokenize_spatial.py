#!/usr/bin/env python3
"""
spot_neighbor_tokenization.py

This module provides tokenization utilities for Visium spatial transcriptomics:
  1. Ensure or compute a neighbor graph via Delaunay triangulation (adata.obsp['spatial_connectivities']).
  2. Stream or load .h5ad/.loom datasets and chunk through cells for memory-efficient tokenization.
  3. Generate two token sequences per cell:
     - 'input_ids': spot-only ranked gene tokens.
     - 'neighbor_ids': spot+neighbor ranked gene tokens.
  4. Concatenate spot and neighbor tokens into a single 'input_ids' vector in the final HF Dataset.

Example usage:
    from spot_neighbor_tokenization import SpotNeighborTranscriptomeTokenizer
    from pathlib import Path

    tokenizer = SpotNeighborTranscriptomeTokenizer(
        custom_attr_name_dict={'sample_id':'sample'},
        nproc=8,
        downsample_percent=None,
        downsample_seed=42,
        gene_median_file=Path('gene_median_dict.pickle'),
        token_dictionary_file=Path('token_dict.pickle'),
    )

    tokenizer.tokenize_data(
        data_directory=Path('/path/to/h5ad_or_loom'),
        output_directory=Path('/output'),
        output_prefix='visium_neighbor',
        file_format='h5ad'  # or 'loom'
    )

All classes and functions are importable without executing at the module level.
"""

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_spatial_connectivities(
    adata: ad.AnnData,
    spatial_key: str = "spatial"
) -> None:
    """
    Compute adata.obsp['spatial_connectivities'] via Delaunay if missing.
    Expects coords in adata.obsm[spatial_key] or ['X_spatial'].
    """
    if "spatial_connectivities" in adata.obsp:
        return

    coords = adata.obsm.get(spatial_key) or adata.obsm.get("X_spatial")
    if coords is None:
        raise KeyError(
            f"Missing spatial coords: obsm['{spatial_key}'] or ['X_spatial'] required"
        )

    tri = Delaunay(coords)
    n = coords.shape[0]
    rows, cols = [], []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                rows.extend([simplex[i], simplex[j]])
                cols.extend([simplex[j], simplex[i]])
    graph = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adata.obsp['spatial_connectivities'] = graph
    logger.info("Built spatial_connectivities via Delaunay")

def rank_genes(gene_vector: np.ndarray, gene_tokens: np.ndarray) -> np.ndarray:
    """Return gene_tokens sorted by descending gene_vector."""
    idx = np.argsort(-gene_vector)
    return gene_tokens[idx]

def check_anndata_format(adata: ad.AnnData) -> Dict[str, object]:
    """Ensure 'ensembl_id' in var and 'n_counts' in obs."""
    result = {"valid": True, "messages": []}
    if 'ensembl_id' not in adata.var:
        result['valid'] = False
        result['messages'].append("Missing 'ensembl_id' in var.")
    if 'n_counts' not in adata.obs:
        result['valid'] = False
        result['messages'].append("Missing 'n_counts' in obs.")
    return result

class SpotNeighborTranscriptomeTokenizer:
    """
    Tokenizes .h5ad or .loom files into HF Datasets by computing:
      - spot-only token ranks ('input_ids')
      - spot+neighbor ranks ('neighbor_ids')
      and concatenating them in the final dataset.
    """
    def __init__(
        self,
        custom_attr_name_dict: Optional[Dict[str, str]] = None,
        nproc: int = 1,
        downsample_percent: Optional[float] = None,
        downsample_seed: Optional[int] = None,
        gene_median_file: Path = Path(__file__).parent / "gene_median_dict.pickle",
        token_dictionary_file: Path = Path(__file__).parent / "token_dict.pickle",
        chunk_size: int = 512,
        target_sum: float = 1e4,
        gene_length: Optional[int] = 2048
    ):
        self.custom_attr_name_dict = custom_attr_name_dict or {}
        self.nproc = nproc
        self.downsample_percent = downsample_percent
        self.downsample_seed = downsample_seed
        self.chunk_size = chunk_size
        self.target_sum = target_sum
        self.gene_length = gene_length

        with open(gene_median_file, 'rb') as f:
            self.gene_median_dict = pickle.load(f)
        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)

        self.genelist = set(self.gene_median_dict.keys())

    def tokenize_data(
        self,
        data_directory: Path,
        output_directory: Path,
        output_prefix: str,
        file_format: Literal['h5ad','loom'] = 'h5ad',
    ) -> None:
        """
        Tokenize all files in data_directory and save HF Dataset to output_directory.
        """
        cells, nei, meta = self._tokenize_files(data_directory, file_format)
        ds = self._create_dataset(cells, nei, meta)
        out = output_directory / f"{output_prefix}.dataset"
        ds.save_to_disk(str(out))
        logger.info(f"Saved dataset → {out}")

    def _tokenize_files(self, data_dir: Path, fmt: str):
        paths = list(data_dir.glob(f"*.{fmt}"))
        if not paths:
            raise FileNotFoundError(f"No *.{fmt} in {data_dir}")

        all_cells, all_nei = [], []
        meta = {out: [] for out in self.custom_attr_name_dict.values()}

        for p in paths:
            logger.info(f"Tokenizing {p.name}")
            if fmt == 'h5ad':
                adata = sc.read_h5ad(str(p))
            else:
                adata = sc.read_loom(str(p), sparse=True)
            ensure_spatial_connectivities(adata)
            c, n, m = self._tokenize_adata(adata)
            all_cells.extend(c)
            all_nei.extend(n)
            for ik, ok in self.custom_attr_name_dict.items():
                meta[ok].extend(m[ik])
        return all_cells, all_nei, meta

    def _tokenize_adata(self, adata: ad.AnnData):
        adata = adata[adata.obs['n_counts']>0]
        if self.downsample_percent:
            idxs = np.arange(adata.n_obs)
            sel, _ = train_test_split(idxs, test_size=self.downsample_percent,
                                      random_state=self.downsample_seed)
            adata = adata[sel,:]

        fmt = check_anndata_format(adata)
        logger.info(f"Format check: {fmt}")

        var_ids = adata.var['ensembl_id'] if 'ensembl_id' in adata.var.columns else adata.var_names
        coding = [i for i,g in enumerate(var_ids) if g in self.genelist]
        tokens = np.array([self.gene_token_dict[var_ids[i]] for i in coding])
        norms  = np.array([self.gene_median_dict[var_ids[i]] for i in coding])

        A = adata.obsp['spatial_connectivities']
        cell_meta = {ik: adata.obs[ik].tolist() for ik in self.custom_attr_name_dict}

        cells_out, nei_out = [], []
        N = adata.n_obs
        for start in range(0, N, self.chunk_size):
            batch = np.arange(start, min(start+self.chunk_size, N))
            X = adata[batch, coding].X
            X = X.toarray() if sp.issparse(X) else X
            ncnt = adata.obs['n_counts'].values[batch][:,None]

            spot = (X/ncnt*self.target_sum)/norms
            nei  = (A[batch,:].dot(X)/ncnt*self.target_sum)/norms

            for r in spot:
                nz = r!=0
                cells_out.append(rank_genes(r[nz], tokens[nz]))
            for r in nei:
                nz = r!=0
                nei_out.append(rank_genes(r[nz], tokens[nz]))

        return cells_out, nei_out, cell_meta

    def _create_dataset(self, cells: List[np.ndarray], nei: List[np.ndarray], meta: Dict[str,List]):
        data = {'input_ids': cells, 'neighbor_ids': nei}
        if meta:
            data.update(meta)

        batches = []
        keys = list(data.keys()); vals = list(data.values())
        for i in range(0, len(vals[0]), 10000):
            sub = {k: v[i:i+10000] for k,v in zip(keys, vals)}
            batches.append(Dataset.from_dict(sub))

        ds = concatenate_datasets(batches).map(
            lambda ex: {'input_ids': ex['input_ids'][:self.gene_length]+ex['neighbor_ids'][:self.gene_length], 'length': len(ex['input_ids'])},
            num_proc=self.nproc
        )
        return ds.remove_columns('neighbor_ids')
