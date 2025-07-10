#!/usr/bin/env python3
"""
st_tokenization.py
    1. Tokenizing AnnData (.h5ad) and loom files based on precomputed gene medians and token dictionaries.

All functionality is encapsulated in importable classes and functions without any main-level execution.


    #. Tokenize a directory of files
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={'sample_id': 'sample'},
        nproc=4,
        gene_median_file=Path('gene_median_dict.pickle'),
        token_dictionary_file=Path('token_dict.pickle')
    )
    tokenizer.tokenize_data(
        data_directory=Path('/input/data_dir'),
        output_directory=Path('/output'),
        output_prefix='myproject',
        file_format='h5ad'  # or 'loom'
    )
"""

import math
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal

import numpy as np
import anndata as ad
import loompy
import scipy.sparse as sp
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Token Dictionary Creation -------------------- #

def create_token_dictionary(gene_median_dict: Dict[str, float], reserved_tokens: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Create a token dictionary mapping genes to integer tokens.

    Example:
        token_dict = create_token_dictionary(medians)
    """
    if reserved_tokens is None:
        reserved_tokens = {'<pad>': 0, '<mask>': 1}
    genes = [g for g, m in gene_median_dict.items() if not math.isnan(m)]
    token_dict = reserved_tokens.copy()
    for i, g in enumerate(genes, start=len(reserved_tokens)):
        token_dict[g] = i
    return token_dict

# -------------------- Tokenization -------------------- #

def rank_genes(gene_vector: np.ndarray, gene_tokens: np.ndarray) -> np.ndarray:
    """
    Rank tokens in descending order of expression.

    Example:
        ranked = rank_genes([5, 2, 0], [10, 11, 12])
    """
    idx = np.argsort(-gene_vector)
    return gene_tokens[idx]

def check_anndata_format(adata: ad.AnnData) -> Dict[str, object]:
    """
    Validate that an AnnData has required fields.

    Returns dict with 'valid' flag and 'messages'.
    """
    result = {"valid": True, "messages": []}
    if "ensembl_id" not in adata.var:
        result["valid"] = False
        result["messages"].append("Missing 'ensembl_id' in adata.var.")
    if "n_counts" not in adata.obs:
        result["valid"] = False
        result["messages"].append("Missing 'n_counts' in adata.obs.")
    return result

# -------------------- Transcriptome Tokenizer -------------------- #

class TranscriptomeTokenizer:
    """
    Tokenizes h5ad or loom datasets into tokenized HuggingFace datasets using precomputed medians & token dicts.

    Example:
        tokenizer = TranscriptomeTokenizer...
        tokenizer.tokenize_data...
    """
    def __init__(
        self,
        custom_attr_name_dict: Optional[Dict[str, str]] = None,
        nproc: int = 1,
        downsample_percent: Optional[float] = None,
        downsample_seed: Optional[int] = None,
        gene_median_file: Path = Path(__file__).parent / "gene_median_dict.pickle",
        token_dictionary_file: Path = Path(__file__).parent / "token_dict.pickle",
        gene_length : Optional[int] = 2048
    ):
        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc
        self.downsample_percent = downsample_percent
        self.downsample_seed = downsample_seed
        self.gene_length = gene_length

        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        self.genelist_dict = {gene: True for gene in self.gene_median_dict.keys()}

    def tokenize_data(
        self,
        data_directory: Path,
        output_directory: Path,
        output_prefix: str,
        file_format: Literal["h5ad", "loom"] = "h5ad",
        use_generator: bool = False,
    ) -> None:
        tokenized_cells, cell_metadata = self.tokenize_files(data_directory, file_format)
        dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator)
        output_path = output_directory / f"{output_prefix}.dataset"
        dataset.save_to_disk(str(output_path))
        logger.info(f"Tokenized dataset saved to: {output_path}")

    def tokenize_files(self, data_directory: Path, file_format: str):
        tokenized_cells = []
        cell_metadata: Dict[str, List] = {}
        if self.custom_attr_name_dict:
            cell_metadata = {out_key: [] for out_key in self.custom_attr_name_dict.values()}

        if file_format == "h5ad":
            files = list(data_directory.glob("*.h5ad"))
            if not files:
                raise FileNotFoundError(f"No .h5ad files found in {data_directory}")
            for file_path in files:
                logger.info(f"Tokenizing h5ad file: {file_path}")
                cells, metadata = self.tokenize_anndata(file_path)
                tokenized_cells.extend(cells)
                if metadata and self.custom_attr_name_dict:
                    for in_key, out_key in self.custom_attr_name_dict.items():
                        cell_metadata[out_key].extend(metadata.get(in_key, []))

        elif file_format == "loom":
            files = list(data_directory.glob("*.loom"))
            if not files:
                raise FileNotFoundError(f"No .loom files found in {data_directory}")
            for file_path in files:
                logger.info(f"Tokenizing loom file: {file_path}")
                cells, _ = self.tokenize_loom(file_path)
                tokenized_cells.extend(cells)

        else:
            raise ValueError("file_format must be 'h5ad' or 'loom'")

        return tokenized_cells, cell_metadata

    def tokenize_anndata(
        self,
        adata_file_path: Path,
        target_sum: int = 10_000,
        chunk_size: int = 512,
    ):
        # load in backed mode and scan in chunks to reduce peak memory
        adata = ad.read_h5ad(str(adata_file_path), backed='r')
        adata = adata[adata.obs["n_counts"] != 0]
        if self.downsample_percent:
            idxs = list(range(adata.n_obs))
            selected, _ = train_test_split(idxs, test_size=self.downsample_percent, random_state=self.downsample_seed)
            adata = adata[selected, :]

        fmt_check = check_anndata_format(adata)
        logger.info(f"AnnData format check: {fmt_check}")

        var_ids = adata.var['ensembl_id']
        coding_indices = np.where([self.genelist_dict.get(g, False) for g in var_ids])[0]
        norm_factors = np.array([self.gene_median_dict[g] for g in var_ids[coding_indices]])
        gene_tokens = np.array([self.gene_token_dict[g] for g in var_ids[coding_indices]])

        n_cells = adata.n_obs
        obs_idxs = np.arange(n_cells)
        totals = np.zeros(n_cells, dtype=float)

        # first pass: compute totals per chunk
        for batch in tqdm(np.array_split(obs_idxs, int(np.ceil(n_cells/chunk_size))), desc='Compute totals (tokenize)'):
            Xb = adata[batch, coding_indices].X
            if sp.issparse(Xb): Xb = Xb.toarray()
            totals[batch] = np.sum(Xb, axis=1)

        tokenized = []
        metadata = {key: [] for key in self.custom_attr_name_dict} if self.custom_attr_name_dict else None

        # second pass: normalize and rank per chunk
        for batch in tqdm(np.array_split(obs_idxs, int(np.ceil(n_cells/chunk_size))), desc='Tokenize (chunks)'):
            Xb = adata[batch, coding_indices].X
            if sp.issparse(Xb): Xb = Xb.toarray()
            X_norm = (Xb / totals[batch][:, None] * target_sum) / norm_factors
            for row in X_norm:
                nz = row != 0
                tokenized.append(rank_genes(row[nz], gene_tokens[nz]))
            if metadata:
                for key in self.custom_attr_name_dict:
                    metadata[key].extend(adata[batch].obs.get(key, []).tolist())

        adata.file.close()
        return tokenized, metadata

    def tokenize_loom(self, loom_file_path: Path):
        """
        Tokenize a loom file based on gene medians and token dictionary.
        """
        with loompy.connect(str(loom_file_path)) as ds:
            var_ids = ds.ra.get("ensembl_id")
            if var_ids is None:
                raise ValueError("Missing 'ensembl_id' in loom file")
            coding_indices = [i for i, g in enumerate(var_ids) if self.genelist_dict.get(g, False)]
            if not coding_indices:
                logger.warning("No matching genes found in loom file.")
                return [], {}

            data_list = []
            for _, _, view in ds.scan(items=coding_indices, axis=0):
                arr = view.view
                row = arr[0, :] if arr.ndim == 2 else arr
                data_list.append(row)
            mat = np.vstack(data_list).T  # cells x genes

            gene_tokens = np.array([self.gene_token_dict[var_ids[idx]] for idx in coding_indices])
            tokenized = []
            for vec in mat:
                nz = vec != 0
                tokenized.append(rank_genes(vec[nz], gene_tokens[nz]))

        return tokenized, {}

    def create_dataset(
        self,
        tokenized_cells: List[np.ndarray],
        cell_metadata: Dict[str, List],
        use_generator: bool = False,
        batch_size: int = 10000,
    ) -> Dataset:
        data = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict and cell_metadata:
            data.update(cell_metadata)
        batches = []
        keys = list(data.keys())
        vals = list(data.values())
        for i in range(0, len(vals[0]), batch_size):
            batch_dict = {k: v[i:i+batch_size] for k, v in zip(keys, vals)}
            batches.append(Dataset.from_dict(batch_dict))
        ds = concatenate_datasets(batches)
        ds = ds.map(lambda ex: {"input_ids": ex["input_ids"][:self.gene_length]}, num_proc=self.nproc)
        ds = ds.map(lambda ex: {"length": len(ex["input_ids"])}, num_proc=self.nproc)
        return ds
