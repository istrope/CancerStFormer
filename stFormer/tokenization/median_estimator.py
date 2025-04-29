"""

This module provides functions and classes for processing spatial transcriptomics datasets:
  1. Computing gene expression medians using T‑Digest approximations from loom or h5ad files.
  4. Creating and merging token dictionaries from gene median dictionaries.


Example usage:
    from st_tokenization import MedianEstimator, create_token_dictionary, TranscriptomeTokenizer, merge_tdigest_dicts
    from pathlib import Path

    # 1. Estimate gene medians in one step (streaming .loom or memory-mapped .h5ad)
    genes = ['GeneA', 'GeneB', 'GeneC']
    estimator = MedianEstimator(gene_list=genes)
    total_counts = estimator.compute_tdigests(Path('sample.loom'))  # or .h5ad
    median_dict = estimator.get_median_dict()

    # 2. (Optional) merge multiple partial digests
    merged_td = merge_tdigest_dicts(Path('pickles'))
    estimator.merge_with(merged_td)
    median_dict = estimator.get_median_dict()

    # 3. Create token dictionary from medians
    token_dict = create_token_dictionary(median_dict)
"""


import math
import pickle
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import anndata as ad
import loompy
import scipy.sparse as sp
from tqdm import tqdm
import crick.tdigest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# -------------------- Median Estimation -------------------- #

class MedianEstimator:
    """
    Computes gene medians using T‑Digest approximations from .loom or memory-mapped .h5ad files.

    Example:
        estimator = MedianEstimator(['GeneA', 'GeneB'])
        totals = estimator.compute_tdigests(Path('data.h5ad'))
        medians = estimator.get_median_dict()
    """
    def __init__(self, gene_list: List[str], normalization_target: float = 10000):
        self.gene_list = gene_list
        self.normalization_target = normalization_target
        self.tdigests: Dict[str, crick.tdigest.TDigest] = {
            gene: crick.tdigest.TDigest() for gene in gene_list
        }

    def compute_tdigests(self, file_path: Path, chunk_size: int = 1000) -> np.ndarray:
        """
        Read a .loom or .h5ad file and update all T‑Digests in a streaming or chunked fashion.

        For .loom: uses loompy.scan. For .h5ad: uses HDF5 backed AnnData with chunked reads.

        Returns:
            np.ndarray: Total counts per cell for the selected genes.
        """
        if isinstance(file_path,str):
            file_path = Path(file_path)
            
        suffix = file_path.suffix.lower()

        if suffix == '.loom':
            # Stream rows from loom
            with loompy.connect(str(file_path)) as ds:
                var_ids = ds.ra.get('ensembl_id')
                if var_ids is None:
                    raise ValueError("Missing 'ensembl_id' in loom file")
                coding = [i for i, g in enumerate(var_ids) if g in self.gene_list]

                totals = np.zeros(ds.shape[1], dtype=float)
                # first pass: per-cell sums
                for _, _, view in ds.scan(items=coding, axis=0):
                    totals += np.nansum(view.view, axis=0)

                # second pass: update T‑Digests
                for idx, _, view in tqdm(ds.scan(items=coding, axis=0), total=len(coding), desc='TDigest (loom)'):
                    gene = var_ids[idx]
                    vals = view.view.flatten() / totals * self.normalization_target
                    vals = np.where(vals == 0, np.nan, vals)
                    valid = vals[~np.isnan(vals)]
                    if valid.size > 0:
                        self.tdigests[gene].update(valid)
                return totals

        elif suffix == '.h5ad':
            # Memory-map h5ad and iterate in chunks
            adata = ad.read_h5ad(str(file_path), backed='r')
            var_ids = adata.var['ensembl_id'] if 'ensembl_id' in adata.var.columns else adata.var_names
            coding = [i for i, g in enumerate(var_ids) if g in self.gene_list]
            n_cells = adata.n_obs

            totals = np.zeros(n_cells, dtype=float)
            idxs = np.arange(n_cells)
            # first pass: compute totals per chunk
            for batch in tqdm(np.array_split(idxs, int(np.ceil(n_cells/chunk_size))), desc='Compute totals (h5ad)'):
                Xb = adata[batch, coding].X
                if sp.issparse(Xb):
                    Xb = Xb.toarray()
                totals[batch] = np.sum(Xb, axis=1)

            # second pass: update T‑Digests
            for batch in tqdm(np.array_split(idxs, int(np.ceil(n_cells/chunk_size))), desc='TDigest (h5ad)'):
                Xb = adata[batch, coding].X
                if sp.issparse(Xb):
                    Xb = Xb.toarray()
                # normalize
                X_norm = (Xb / totals[batch][:, None]) * self.normalization_target
                for j, gene_idx in enumerate(coding):
                    gene = var_ids[gene_idx]
                    vals = X_norm[:, j]
                    # filter zeros
                    vals = vals[vals > 0]
                    if vals.size > 0:
                        self.tdigests[gene].update(vals)
            adata.file.close()
            return totals

        else:
            raise ValueError('File must have .loom or .h5ad extension')

    def get_median_dict(self, detected_only: bool = True) -> Dict[str, float]:
        """
        Return a mapping of gene IDs to median expression values.

        Example:
            medians = estimator.get_median_dict()
        """
        med = {g: td.quantile(0.5) for g, td in self.tdigests.items()}
        if detected_only:
            med = {g: m for g, m in med.items() if not math.isnan(m)}
        return med

    def merge_with(self, new_tdigest_dict: Dict[str, crick.tdigest.TDigest]) -> None:
        """
        Merge an external T‑Digest dictionary into this estimator.

        Example:
            merged = merge_tdigest_dicts(Path('pickles'))
            estimator.merge_with(merged)
        """
        for g, new_td in new_tdigest_dict.items():
            if g in self.tdigests:
                self.tdigests[g].merge(new_td)
            else:
                self.tdigests[g] = new_td

# -------------------- Merge Utility -------------------- #

def merge_tdigest_dicts(directory: Path, pattern: str = "*.pickle") -> Dict[str, crick.tdigest.TDigest]:
    """
    Merge multiple T‑Digest pickle files from a directory.

    Example:
        merged = merge_tdigest_dicts(Path('digests'))
    """
    merged = {}
    for f in directory.glob(pattern):
        with open(f, 'rb') as fp:
            td = pickle.load(fp)
        for g, td_obj in td.items():
            if g in merged:
                merged[g].merge(td_obj)
            else:
                merged[g] = td_obj
    return merged