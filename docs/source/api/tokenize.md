# Spatial Transcriptomics Tokenization & Embedding API

## Overview

This API provides tools to preprocess, tokenize, and embed spatial transcriptomics data using T-Digest-based normalization and transformer-based models (Geneformer). This documentation follows a Scanpy-style format to help with integration into ReadTheDocs.

---

## `MedianEstimator`

**Module**: `spatial_tokenization`

Estimates gene expression medians using T-Digest from `.h5ad` or `.loom` files.

### `__init__`

```python
MedianEstimator(
    data_dir: Union[str, Path],
    extension: str = ".h5ad",
    out_path: Union[str, Path] = "output",
    merge_tdigests: bool = False,
    normalization_target: float = 10000.0
)
```

**Parameters:**

- **data\_dir** : `Path | str`\
  Directory containing `.h5ad` or `.loom` files.
- **extension** : `str`, default: `.h5ad`\
  File extension to identify files to process.
- **out\_path** : `Path | str`, default: `output`\
  Output directory for saving results.
- **merge\_tdigests** : `bool`, default: `False`\
  If True, merge T-Digests across all files.
- **normalization\_target** : `float`, default: `10000.0`\
  Target value to normalize UMI counts.

---

### `compute_tdigests`

```python
compute_tdigests(file_path: Optional[Path | str] = None, chunk_size: int = 1000)
```

Compute gene-level T-Digests per dataset.

**Parameters:**

- **file\_path** : `Path | str | None`, default: `None`\
  Optional single file to process. If None, all files in `data_dir` are processed.
- **chunk\_size** : `int`, default: `1000`\
  Chunk size for processing cells.

**Returns:**

- `np.ndarray` or `None` — Per-cell totals if `file_path` provided.

---

### `get_median_dict`

```python
get_median_dict(detected_only: bool = True)
```

Return dictionary of gene → median value.

**Parameters:**

- **detected\_only** : `bool`, default: `True`\
  Exclude genes with NaN values.

**Returns:**

- `Dict[str, float]` — Gene → median mapping.

---

### `write_tdigests`

```python
write_tdigests()
```

Write per-gene or merged T-Digests to disk.

**Returns:**

- `None`

---

### `write_medians`

```python
write_medians()
```

Write computed medians to pickle file.

**Returns:**

- `None`

---

## Utility Functions

### `merge_tdigest_dicts`

```python
merge_tdigest_dicts(directory: Path, pattern: str = "*.pickle")
```

Merge multiple T-Digest dictionaries.

**Parameters:**

- **directory** : `Path`\
  Directory containing `.pickle` files.
- **pattern** : `str`, default: `*.pickle`\
  Pattern to match files.

**Returns:**

- `Dict[str, Tdigest]` — Merged gene-level T-Digest dictionary.

---

### `create_token_dictionary`

```python
create_token_dictionary(median_dict: Dict[str, float], reserved: Optional[Dict[str, int]] = None)
```

Create token dictionary from gene medians.

**Parameters:**

- **median\_dict** : `Dict[str, float]`\
  Gene → median mapping.
- **reserved** : `Dict[str, int]`, default: `{ '<pad>': 0, '<mask>': 1 }`\
  Reserved tokens to include in dictionary.

**Returns:**

- `Dict[str, int]` — Token dictionary.

---

## `SpatialTokenizer`

**Module**: `spatial_tokenization`

Tokenizes spatial transcriptomic datasets using either spot-level or neighborhood-level context.

### `__init__`

```python
SpatialTokenizer(
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
)
```

**Parameters:**

- **mode** : `'spot' | 'neighborhood'`, default: `'spot'`\
  Tokenization mode. 'neighborhood' includes spatial neighbors.
- **gene\_length** : `int`, default: `2048`\
  Number of top genes per cell/spot.
- **custom\_meta** : `dict`, default: `None`\
  Mapping of obs columns to metadata fields.
- **nproc** : `int`, default: `1`\
  Number of processes for parallel processing.
- **down\_pct** : `float`, default: `None`\
  Fraction of cells to downsample.
- **down\_seed** : `int`, default: `None`\
  Random seed for downsampling.
- **gene\_median\_file** : `Path`, default: `'gene_median_dict.pickle'`\
  Path to gene median values.
- **token\_dict\_file** : `Path`, default: `'token_dict.pickle'`\
  Path to gene token dictionary.
- **chunk** : `int`, default: `512`\
  Chunk size during processing.
- **target** : `float`, default: `1e4`\
  Normalization target for counts.

---

### `tokenize`

```python
tokenize(data_dir: Path, out_dir: Path, prefix: str)
```

Tokenize all `.h5ad` or `.loom` files in the given directory.

**Parameters:**

- **data\_dir** : `Path`\
  Directory with input files.
- **out\_dir** : `Path`\
  Directory to save HuggingFace-style output.
- **prefix** : `str`\
  Prefix for output dataset.

**Returns:**

- `None`

---

## `EmbExtractor`

**Module**: `embedding_extractor`

Extracts embeddings from tokenized inputs using Geneformer models.

### `__init__`

```python
EmbExtractor(
    model_type='Pretrained',
    num_classes=0,
    emb_mode='cell',
    cell_emb_style='mean_pool',
    gene_emb_style='mean_pool',
    filter_data=None,
    max_ncells=1000,
    emb_layer=-1,
    emb_label=None,
    labels_to_plot=None,
    forward_batch_size=100,
    nproc=4,
    summary_stat=None,
    token_dictionary_file=None
)
```

**Parameters:**

- **model\_type** : `str`, default: `'Pretrained'`\
  One of `'Pretrained'`, `'GeneClassifier'`, `'CellClassifier'`.
- **num\_classes** : `int`, default: `0`\
  Number of output classes if model is a classifier.
- **emb\_mode** : `str`, default: `'cell'`\
  One of `'cls'`, `'cell'`, or `'gene'`.
- **cell\_emb\_style** : `str`, default: `'mean_pool'`\
  Cell embedding strategy (currently only `'mean_pool'`).
- **gene\_emb\_style** : `str`, default: `'mean_pool'`\
  Gene embedding strategy (currently only `'mean_pool'`).
- **filter\_data** : `dict | None`, default: `None`\
  Dictionary to filter cells (e.g., `{"cell_type": ["neuron"]}`).
- **max\_ncells** : `int`, default: `1000`\
  Max cells to extract embeddings from.
- **emb\_layer** : `int`, default: `-1`\
  Layer index to extract from: `-1` (second-to-last), `0` (last).
- **emb\_label** : `list | None`, default: `None`\
  Columns from the dataset to append as labels.
- **labels\_to\_plot** : `list | None`, default: `None`\
  Labels to use in plots.
- **forward\_batch\_size** : `int`, default: `100`\
  Forward batch size for inference.
- **nproc** : `int`, default: `4`\
  Number of parallel processes.
- **summary\_stat** : `str | None`, default: `None`\
  Options: `'mean'`, `'median'`, `'exact_mean'`, `'exact_median'`.
- **token\_dictionary\_file** : `Path | None`\
  Path to token dictionary pickle file.

---

### `extract_embs`

```python
extract_embs(model_directory, input_data_file, output_directory, output_prefix, output_torch_embs=False, cell_state=None)
```

Extract embeddings from a tokenized dataset.

**Returns:**

- `pd.DataFrame` or tuple with `torch.Tensor`

---

### `get_state_embs`

```python
get_state_embs(cell_states_to_model, model_directory, input_data_file, output_directory, output_prefix, output_torch_embs=True)
```

Compute state-based embedding dictionaries for perturbation modeling.

**Returns:**

- `Dict[str, torch.Tensor]`

---

### `plot_embs`

```python
plot_embs(embs, plot_style, output_directory, output_prefix, max_ncells_to_plot=1000, kwargs_dict=None)
```

Generate UMAP or heatmap plots for embeddings.

**Returns:**

- `None`
