import math
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import anndata as ad
import loompy
import scipy.sparse as sp
from tqdm import tqdm
import crick.tdigest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedianEstimator:
    """
    Computes gene medians using T-Digest approximations from .loom or memory-mapped .h5ad files.
    Can operate on a directory of files, caching datasets to minimize I/O.

    Example (directory, cached):
        estimator = MedianEstimator(
            data_dir='data',           # directory containing .h5ad/.loom files
            extension='.h5ad',         # file extension to process
            out_path='output',         # where to save results
            merge_tdigests=True        # merge T-Digests across samples
        )
        estimator.compute_tdigests()  # loads files once, processes all
        medians = estimator.get_median_dict()
    """
    def __init__(
        self,
        data_dir: Path | str,
        extension: str = '.h5ad',
        out_path: Path | str = 'output',
        merge_tdigests: bool = False,
        normalization_target: float = 10000.0,
    ):
        self.data_dir = Path(data_dir)
        self.extension = extension.lower()
        self.out_path = Path(out_path)
        self.merge_tdigests = merge_tdigests
        self.normalization_target = normalization_target

        # Cache for opened datasets
        self.datasets: Dict[Path, Any] = {}
        # Gene list and T-Digests will be set during processing
        self.gene_list: List[str] = []
        self.tdigests: Dict[str, crick.tdigest.TDigest] = {}

    def _load_datasets(self) -> None:
        """
        Cache all files matching extension in self.data_dir.
        """
        if self.datasets:
            return
        files = sorted(self.data_dir.glob(f'*{self.extension}'))
        if not files:
            raise ValueError(f'No files with extension {self.extension} in {self.data_dir}')
        for f in files:
            if f.suffix.lower() == '.h5ad':
                adata = ad.read_h5ad(str(f), backed='r')
                self.datasets[f] = adata
            elif f.suffix.lower() == '.loom':
                ds = loompy.connect(str(f))
                self.datasets[f] = ds

    def compute_tdigests(self, file_path: Optional[Path | str] = None, chunk_size: int = 1000) -> Optional[np.ndarray]:
        """
        Compute T-Digests:
        - If file_path given, processes single file (caches file on first call).
        - If no file_path, loads all in data_dir, caches once, then processes directory.
        """
        if file_path:
            p = Path(file_path)
            if p not in self.datasets:
                # cache single file
                self.data_dir = p.parent
                self.extension = p.suffix.lower()
                self._load_datasets()
            return self._compute_tdigests_file(p, chunk_size)
        else:
            self._load_datasets()
            return self._compute_tdigests_dir(chunk_size)

    def _compute_tdigests_dir(self, chunk_size: int) -> None:
        """
        Process all cached datasets; merge or separate per merge_tdigests flag.
        """
        files = list(self.datasets.keys())
        if self.merge_tdigests:
            # Union gene IDs
            gene_ids = set()
            for f, ds in self.datasets.items():
                gene_ids.update(self._get_gene_ids_handle(ds))
            self.gene_list = sorted(gene_ids)
            self.tdigests = {g: crick.tdigest.TDigest() for g in self.gene_list}
            # Process each dataset
            for f, ds in self.datasets.items():
                logger.info(f"Merging T-Digests from {f.name}")
                self._compute_tdigests_handle(ds, f, chunk_size)
        else:
            # Per-file t-digest
            last_td = {}
            for f, ds in self.datasets.items():
                genes = self._get_gene_ids_handle(ds)
                td_dict = {g: crick.tdigest.TDigest() for g in genes}
                self._compute_tdigests_handle(ds, f, chunk_size, td_dict)
                last_td = td_dict
            self.gene_list = genes
            self.tdigests = last_td

    def _compute_tdigests_file(self, file_path: Path, chunk_size: int, tdigests: Optional[Dict[str, crick.tdigest.TDigest]] = None) -> np.ndarray:
        """
        Wrapper for directory-agnostic single-file computation.
        """
        ds = self.datasets[file_path]
        return self._compute_tdigests_handle(ds, file_path, chunk_size, tdigests)

    def _compute_tdigests_handle(
        self,
        ds: Any,
        file_path: Path,
        chunk_size: int,
        tdigests: Optional[Dict[str, crick.tdigest.TDigest]] = None
    ) -> np.ndarray:
        """
        Core routine: update tdigests for one dataset handle.
        Returns per-cell totals array.
        """
        if tdigests is None:
            tdigests = self.tdigests
        suffix = file_path.suffix.lower()
        # Extract gene IDs and coding indices
        if suffix == '.loom':
            var_ids = ds.ra.get('ensembl_id')
            if var_ids is None:
                raise ValueError("Missing 'ensembl_id' in loom file")
            coding = [i for i, g in enumerate(var_ids) if g in tdigests]
            # first pass: totals
            totals = np.zeros(ds.shape[1], dtype=float)
            for _, _, view in ds.scan(items=coding, axis=0):
                totals += np.nansum(view.view, axis=0)
            # second pass: update
            for idx, _, view in tqdm(ds.scan(items=coding, axis=0), total=len(coding), desc='TDigest (loom)'):
                gene = var_ids[idx]
                vals = view.view.flatten() / totals * self.normalization_target
                vals = np.where(vals == 0, np.nan, vals)
                valid = vals[~np.isnan(vals)]
                if valid.size > 0:
                    tdigests[gene].update(valid)
            return totals
        elif suffix == '.h5ad':
            adata = ds  # backed AnnData
            var_ids = adata.var['ensembl_id'] if 'ensembl_id' in adata.var.columns else adata.var_names
            coding = [i for i, g in enumerate(var_ids) if g in tdigests]
            n_cells = adata.n_obs
            totals = np.zeros(n_cells, dtype=float)
            idxs = np.arange(n_cells)
            # totals pass
            for batch in tqdm(np.array_split(idxs, int(np.ceil(n_cells / chunk_size))), desc='Compute totals (h5ad)'):
                Xb = adata[batch, coding].X
                if sp.issparse(Xb):
                    Xb = Xb.toarray()
                totals[batch] = np.sum(Xb, axis=1)
            # update pass
            for batch in tqdm(np.array_split(idxs, int(np.ceil(n_cells / chunk_size))), desc='TDigest (h5ad)'):
                Xb = adata[batch, coding].X
                if sp.issparse(Xb):
                    Xb = Xb.toarray()
                X_norm = (Xb / totals[batch][:, None]) * self.normalization_target
                for j, gene_idx in enumerate(coding):
                    gene = var_ids[gene_idx]
                    vals = X_norm[:, j]
                    vals = vals[vals > 0]
                    if vals.size > 0:
                        tdigests[gene].update(vals)
            adata.file.close()
            return totals
        else:
            raise ValueError('Unsupported extension ' + suffix)

    def _get_gene_ids_handle(self, ds: Any) -> List[str]:
        """
        Extract gene IDs from an open dataset handle.
        """
        # h5ad
        if hasattr(ds, 'var'):
            if 'ensembl_id' in ds.var.columns:
                return ds.var['ensembl_id'].tolist()
            else:
                raise ValueError("Missing 'ensembl_id' in h5ad file")
        # loom
        elif hasattr(ds, 'ra'):
            var_ids = ds.ra.get('ensembl_id')
            if var_ids is None:
                raise ValueError("Missing 'ensembl_id' in loom file")
            return list(var_ids)
        else:
            return []

    def get_median_dict(self, detected_only: bool = True) -> Dict[str, float]:
        """
        Return a mapping of gene IDs to median expression values.
        """
        med = {g: td.quantile(0.5) for g, td in self.tdigests.items()}
        if detected_only:
            med = {g: m for g, m in med.items() if not math.isnan(m)}
        self.medians = med

    def merge_with(self, new_tdigest_dict: Dict[str, crick.tdigest.TDigest]) -> None:
        """
        Merge an external T-Digest dictionary into this estimator.
        """
        for g, new_td in new_tdigest_dict.items():
            if g in self.tdigests:
                self.tdigests[g].merge(new_td)
            else:
                self.tdigests[g] = new_td

    def write_tdigests(self):
        self.out_path.mkdir(parents=True, exist_ok=True)
        if self.merge_tdigests: #merged tdigests 
            out_file = self.out_path / 'merged_tdigests.pickle'
            with open(out_file, 'wb') as fp:
                pickle.dump(self.tdigests, fp)
            logger.info(f"Saved merged T-Digests to {out_file}")
        else:
            # Per-file t-digest
            last_td = {}
            for f, ds in self.datasets.items():
                out_file = self.out_path / f'{f.stem}_tdigests.pickle'
                with open(out_file, 'wb') as fp:
                    pickle.dump(ds, fp)
                logger.info(f"Saved T-Digests to {out_file}")

    def write_medians(self):
        self.out_path.mkdir(parents=True,exist_ok=True)
        out_file = self.out_path / 'gene_medians.pickle'
        with open(out_file,'wb') as fp:
            pickle.dump(self.medians,fp)
        logger.info(f'Saved gene medians to {out_file}')



def merge_tdigest_dicts(directory: Path, pattern: str = "*.pickle") -> Dict[str, crick.tdigest.TDigest]:
    """
    Merge multiple T-Digest pickle files from a directory.
    """
    merged: Dict[str, crick.tdigest.TDigest] = {}
    for f in Path(directory).glob(pattern):
        with open(f, 'rb') as fp:
            td = pickle.load(fp)
        for g, td_obj in td.items():
            if g in merged:
                merged[g].merge(td_obj)
            else:
                merged[g] = td_obj
    return merged
