# In-Silico Perturbation

This module implements in-silico perturbation for STFormer-style models, including
dataset filtering, model adapters, perturbation operators, and downstream statistics.

---

## class: Perturber

```python
class Perturber:
    def __init__(
        self,
        perturb_type,
        mode,
        perturb_rank_shift,
        genes_to_perturb,
        genes_to_keep,
        perturb_fraction,
        num_samples,
        top_k,
        model_type,
        anchor_gene,
        cell_states_to_model,
        cell_emb_style,
        num_classes,
        start_state,
        filter_data,
        emb_layer,
        emb_mode,
        forward_batch_size,
        nproc,
        token_dictionary_file,
        cell_inds_to_perturb=None,
        **kwargs,
    )
````

Instantiate an in-silico perturber to simulate gene-level or cell-level perturbations on
tokenized spot or extended (spot + neighbor) sequences.

**Parameters**

- **perturb_type** (`str`): High-level perturbation mode. Supports
  group-style operations such as `"group"`, `"single"`, `"delete"`, or `"overexpress"`.
- **mode** (`str`): Dataset tokenization mode; `"spot"` for single sequences or
  `"extended"` for concatenated spot/neighbor sequences.
- **perturb_rank_shift** (`str` or `None`): Rank-shift operation label (e.g. `"delete"`, `"overexpress"`);
  primarily used to distinguish behaviors for group and single perturbations.
- **genes_to_perturb** (`list[str | int]` or `None`): Gene identifiers (ENSEMBL IDs or token IDs)
  to perturb. If `None`, falls back to single-token scan mode.
- **genes_to_keep** (`list[str | int]` or `None`): Genes that should never be perturbed; all tokens
  in this set are excluded from perturbation.
- **perturb_fraction** (`float` or `None`): Fraction of eligible tokens (after `genes_to_keep` filtering)
  to perturb in group mode. Must lie in `(0, 1]` if provided.
- **num_samples** (`int`): Maximum number of examples to perturb after filtering; dataset is shuffled
  and truncated if larger than this value.
- **top_k** (`int` or `None`): Upper bound on the number of positions to perturb across a batch in group mode.
- **model_type** (`str`): Type of model to load. One of `"Pretrained"`, `"GeneClassifier"`, or `"CellClassifier"`.
- **anchor_gene** (`str` or `int` or `None`): Optional anchor identifier (ENSEMBL or token ID); when
  `genes_to_perturb` is omitted, a valid `anchor_gene` will be used as the sole perturbation target.
- **cell_states_to_model** (`dict` or `None`): Optional configuration describing start/goal/alternate
  cell states for downstream statistics; passed to perturbation stats rather than used directly here.
- **cell_emb_style** (`str`): Strategy to generate cell embeddings from token embeddings; currently
  `"mean_pool"` is used to average token embeddings over the sequence.
- **num_classes** (`int`): Number of label classes when `model_type` is a classifier; ignored for
  `"Pretrained"` models.
- **start_state** (`dict` or `None`): Optional configuration for restricting the dataset to a specific
  start cell state (e.g. `{"state_key": "Group", "start_state": "Vehicle"}`).
- **filter_data** (`dict` or `None`): Column-to-values mapping used to pre-filter the input dataset
  before any perturbations (e.g. `{"Tissue": ["Brain"]}`).
- **emb_layer** (`int`): Layer offset (added to the total number of transformer layers) for selecting
  the representation used in similarity calculations (e.g. `-1` for final hidden layer).
- **emb_mode** (`str`): Embedding aggregation mode. `"cell"` computes only pooled cell-level similarities;
  `"gene"` computes token-wise similarities; `"cell_and_gene"` records both.
- **forward_batch_size** (`int`): Batch size used for forward passes during perturbation.
- **nproc** (`int`): Number of processes used for dataset filtering and mapping.
- **token_dictionary_file** (`str`): Path to a pickled mapping of ENSEMBL IDs to token IDs used to
  resolve `genes_to_perturb`, `genes_to_keep`, and `anchor_gene`.
- **cell_inds_to_perturb** (`dict` or `None`): Optional slice dictionary of form
  `{"start": int, "end": int}` used to restrict perturbations to a specific index range after filtering.
- **kwargs**: Additional legacy keyword arguments are accepted and ignored to preserve backward
  compatibility with older APIs.

**Raises**

- **ValueError**: If configuration values are incompatible (e.g. invalid `perturb_type`, negative
  `num_samples`, or incorrect `cell_inds_to_perturb` bounds).


### perturb_dataset

```python
def perturb_dataset(
    self,
    model_directory: str,
    input_data_file: str,
    output_directory: str,
    output_prefix: str,
) -> str
```

Run the full perturbation workflow: load model, filter and slice the dataset, apply group
or single perturbations, compute cosine similarity shifts, and write raw results to disk.

**Parameters**

- **model_directory** (`str`): Path to a pretrained or classifier model checkpoint directory
  compatible with Hugging Face `AutoModel` / `AutoModelForSequenceClassification`.
- **input_data_file** (`str`): Path to a tokenized Hugging Face `Dataset` on disk (as produced
  by the STFormer tokenization pipeline).
- **output_directory** (`str`): Directory where raw perturbation similarity pickles will be saved.
- **output_prefix** (`str`): Prefix for the output filename. The method appends `"_raw.pickle"`.

**Returns**

- **str**: Full path to the raw similarity file (`"{output_directory}/{output_prefix}_raw.pickle"`).

**Description**

1. Loads the requested model on CPU or GPU using `load_model_to_device`.
2. Determines the effective embedding layer index via `ModelAdapter.quant_layers()` and `emb_layer`.
3. Loads and filters the dataset with `load_and_filter_dataset`, including optional start-state,
   token-based, and index range filters.
4. Resolves ENSEMBL IDs to token IDs based on `token_dictionary_file` for `genes_to_perturb` and
   `genes_to_keep`, and configures a `PerturbOps` instance.
5. In **group mode** (when explicit perturbation targets are supplied), creates a perturbed dataset
   via HF `Dataset.map`, applies delete or overexpression operations (`delete_indices`,
   `overexpress_tokens`, `overexpress_tokens_extended`), and aligns original vs. perturbed sequences
   for token-wise cosine similarity using `quant_cos_sims_tokenwise`.
6. In **single mode**, iterates over distinct tokens present in each batch and deletes each token
   individually, computing cell-level cosine shifts for each target.
7. Aggregates all similarity scores in a `defaultdict` keyed by `(perturbed_token, "cell_emb")`
   and `(perturbed_token, affected_token)` (for gene-level sims), then writes the dictionary using
   `write_perturbation_dictionary`.


---

## Core Dataset & Model Utilities (`perturb_utils.py`)

### load_and_filter_dataset

```python
def load_and_filter_dataset(
    filter_data: Optional[dict],
    nproc: int,
    input_data_file: str
) -> Dataset
```

Load a Hugging Face `Dataset` from disk and apply optional metadata filtering.

**Parameters**

- **filter_data** (`dict` or `None`): Mapping from column names to allowed values; if `None`, no
  filters are applied.
- **nproc** (`int`): Number of processes to use for filtering.
- **input_data_file** (`str`): Directory path of a saved HF `Dataset` (as from `datasets.load_from_disk`).

**Returns**

- **Dataset**: Filtered dataset.


### filter_by_metadata

```python
def filter_by_metadata(
    data: Dataset,
    filter_data: dict,
    nproc: int
) -> Dataset
```

Retain only examples whose metadata columns match specified values.

**Parameters**

- **data** (`Dataset`): Input dataset containing metadata columns.
- **filter_data** (`dict`): Mapping from column name to allowed values (scalar or list).
- **nproc** (`int`): Number of workers for parallel filtering.

**Returns**

- **Dataset**: Subset of `data` containing only rows that satisfy all filter criteria.

**Raises**

- **ValueError**: If no rows remain after filtering.


### filter_by_start_state

```python
def filter_by_start_state(
    data: Dataset,
    state_dict: dict,
    nproc: int
) -> Dataset
```

Filter the dataset to examples that belong to specific start states.

**Parameters**

- **data** (`Dataset`): Input dataset.
- **state_dict** (`dict`): Contains at least `"state_key"` and one or more desired state values
  (e.g. `{"state_key": "Group", "start_state": "Vehicle"}`).
- **nproc** (`int`): Number of processes for filtering.

**Returns**

- **Dataset**: Filtered dataset restricted to the specified start-state values.


### slice_by_indices_to_perturb

```python
def slice_by_indices_to_perturb(
    data: Dataset,
    inds: dict
) -> Dataset
```

Return a contiguous slice of the dataset based on `start` and `end` indices.

**Parameters**

- **data** (`Dataset`): Input dataset after other filters.
- **inds** (`dict`): Dictionary with `"start"` and `"end"` keys specifying the inclusive start
  and exclusive end indices of the slice.

**Returns**

- **Dataset**: Sliced dataset from `start` to `end` (clamped to dataset length).

**Raises**

- **ValueError**: If the slice range is invalid (negative start, end ≤ start, or start beyond length).


### downsample_and_sort

```python
def downsample_and_sort(
    data: Dataset,
    max_ncells: int
) -> Dataset
```

Downsample to at most `max_ncells` examples while preserving original order.

**Parameters**

- **data** (`Dataset`): Input dataset.
- **max_ncells** (`int` or `None`): Maximum number of examples to retain. If `None` or if
  `len(data) <= max_ncells`, the dataset is returned unchanged.

**Returns**

- **Dataset**: Possibly truncated dataset.


### load_model_to_device

```python
def load_model_to_device(
    model_type: str,
    num_classes: int,
    model_directory: str,
    mode: str = "eval"
)
```

Load a model from a checkpoint directory and move it to CPU or GPU.

**Parameters**

- **model_type** (`str`): `"Pretrained"` to load `AutoModel`, `"GeneClassifier"` or `"CellClassifier"`
  to load `AutoModelForSequenceClassification` with `num_labels=num_classes`.
- **num_classes** (`int`): Number of labels for classification heads; ignored if `model_type="Pretrained"`.
- **model_directory** (`str`): Path to the model checkpoint directory.
- **mode** (`str`): `"eval"` to set the model in evaluation mode; other values leave training mode.

**Returns**

- **nn.Module**: Loaded Hugging Face model with `output_hidden_states=True`.


### quant_layers

```python
def quant_layers(model) -> int
```

Infer the number of transformer layers in a model for negative indexing.

**Parameters**

- **model**: Hugging Face model instance.

**Returns**

- **int**: Number of hidden layers (e.g. 12 for BERT-base), used to compute layer offsets.


### get_model_hidden_size

```python
def get_model_hidden_size(model) -> int
```

Return the hidden size (embedding dimension) from the model configuration.

**Parameters**

- **model**: Hugging Face model instance.

**Returns**

- **int**: Hidden size (e.g. 768).

**Raises**

- **AttributeError**: If the hidden size cannot be determined from the config.


### get_model_input_size

```python
def get_model_input_size(model) -> int
```

Determine the maximum input sequence length for a model.

**Parameters**

- **model**: Hugging Face model instance.

**Returns**

- **int**: Maximum sequence length, derived from `max_position_embeddings` or defaulting to `512`.


### quant_cos_sims_tokenwise

```python
def quant_cos_sims_tokenwise(
    hid_orig: torch.Tensor,
    hid_pert: torch.Tensor
) -> torch.Tensor
```

Compute token-wise cosine similarities between original and perturbed hidden states.

**Parameters**

- **hid_orig** (`torch.Tensor`): Original hidden states with shape `[B, L, H]`.
- **hid_pert** (`torch.Tensor`): Perturbed hidden states with the same shape.

**Returns**

- **torch.Tensor**: Cosine similarities for each token `[B, L]`.


### quant_cos_sims

```python
def quant_cos_sims(
    A: torch.Tensor,
    B: torch.Tensor,
    cell_states_to_model=None,
    state_embs_dict=None,
    emb_mode: str = "gene"
) -> torch.Tensor
```

Compute cosine similarities between embeddings at either gene or cell level.

**Parameters**

- **A** (`torch.Tensor`): First embedding tensor. `[B, L, H]` for `"gene"` mode or `[B, H]` for `"cell"` mode.
- **B** (`torch.Tensor`): Second embedding tensor with matching shape.
- **cell_states_to_model** (`dict` or `None`): Reserved for state-aware comparisons (not used directly here).
- **state_embs_dict** (`dict` or `None`): Optional state embedding dictionary (not used directly here).
- **emb_mode** (`str`): `"gene"` for per-token similarities or `"cell"` for per-cell similarities.

**Returns**

- **torch.Tensor**: `[B, L]` in gene mode or `[B]` in cell mode.


### remove_front_per_example

```python
def remove_front_per_example(
    hid: torch.Tensor,
    k_vec: torch.Tensor
) -> torch.Tensor
```

Remove a variable number of tokens from the front of each example and pad to a common length.

**Parameters**

- **hid** (`torch.Tensor`): Hidden states of shape `[B, L, H]`.
- **k_vec** (`torch.Tensor`): Per-example counts of tokens to remove from the front; shape `[B]`.

**Returns**

- **torch.Tensor**: New tensor `[B, L', H]` where `L'` is the maximum remaining length in the batch.


### remove_front_per_example_2d

```python
def remove_front_per_example_2d(
    ids: torch.Tensor,
    k_vec: torch.Tensor
) -> torch.Tensor
```

Remove a variable number of token IDs from the front of each sequence and pad to a common length.

**Parameters**

- **ids** (`torch.Tensor`): Token IDs of shape `[B, L]`.
- **k_vec** (`torch.Tensor`): Per-example counts of tokens to drop from the front; shape `[B]`.

**Returns**

- **torch.Tensor**: New token ID tensor `[B, L']` with padding appended as zeros.


### pad_tensor_list

```python
def pad_tensor_list(
    tensors: List[torch.Tensor],
    max_len: int,
    pad_token_id: int,
    model_input_size: int,
    dim_to_pad: int = 1,
    pad_fn=None,
) -> torch.Tensor
```

Pad a list of tensors along the sequence dimension and stack them into a batch.

**Parameters**

- **tensors** (`List[torch.Tensor]`): List of 1D or 2D tensors containing token IDs or embeddings.
- **max_len** (`int`): Maximum length to pad/truncate to in this batch.
- **pad_token_id** (`int`): Token ID used for padding `input_ids`.
- **model_input_size** (`int`): Global maximum model input size (used by custom pad functions).
- **dim_to_pad** (`int`): Dimension index corresponding to sequence length (default `1`).
- **pad_fn** (callable or `None`): Custom padding function; if `None`, a simple right-padding scheme is used.

**Returns**

- **torch.Tensor**: Stacked padded tensor batch.


### pad_3d_tensor

```python
def pad_3d_tensor(
    tensors: List[torch.Tensor],
    max_len: int,
    pad_token_id: int,
    model_input_size: int,
    dim_to_pad: int = 1,
) -> torch.Tensor
```

Pad a list of 2D or 3D tensors to a common length and stack into a 3D batch tensor.

**Parameters**

- **tensors** (`List[torch.Tensor]`): Each element is `[L, H]` or `[B, L, H]`.
- **max_len** (`int`): Target sequence length.
- **pad_token_id** (`int`): Unused for embedding tensors; kept for API compatibility.
- **model_input_size** (`int`): Global maximum model input size (unused here).
- **dim_to_pad** (`int`): Sequence dimension index (unused; assumed `1`).

**Returns**

- **torch.Tensor**: Concatenated tensor of shape `[B, max_len, H]`.


### gen_attention_mask

```python
def gen_attention_mask(
    batch: Dataset
) -> torch.Tensor
```

Build an attention mask from a dataset batch that has `length` and `input_ids` fields.

**Parameters**

- **batch** (`Dataset`): Mini-batch from a HF `Dataset` containing `length` for each example.

**Returns**

- **torch.Tensor**: Attention mask of shape `[batch_size, max_len]` filled with ones.


### gen_attention_mask_from_lengths

```python
def gen_attention_mask_from_lengths(
    lengths: int,
    batch_size: int
) -> torch.Tensor
```

Generate an all-ones attention mask given a maximum length and batch size.

**Parameters**

- **lengths** (`int`): Maximum sequence length for the current batch.
- **batch_size** (`int`): Number of examples.

**Returns**

- **torch.Tensor**: Mask tensor of ones with shape `[batch_size, lengths]`.


### mean_nonpadding_embs

```python
def mean_nonpadding_embs(
    embs: torch.Tensor,
    lengths: torch.Tensor
) -> torch.Tensor
```

Mean-pool embeddings over non-padded positions in each sequence.

**Parameters**

- **embs** (`torch.Tensor`): Embedding tensor of shape `[B, L, H]`.
- **lengths** (`torch.Tensor`): True sequence lengths for each example; shape `[B]`.

**Returns**

- **torch.Tensor**: Mean-pooled embeddings `[B, H]`.


### class: BatchMaker

```python
@dataclass
class BatchMaker:
    pad_token_id: int
    model_input_size: int
    batch_size: int = 64

    def iter(
        self,
        data: Dataset,
        with_indices: bool = False,
        progress_desc: Optional[str] = None
    )
```

Create padded mini-batches from a sorted HF `Dataset` for model forward passes.

**Parameters (init)**

- **pad_token_id** (`int`): Padding token ID.
- **model_input_size** (`int`): Maximum model sequence length.
- **batch_size** (`int`): Number of examples per batch.

**Methods**

- **iter(data, with_indices=False, progress_desc=None)**: Iterate over the dataset in contiguous
  slices, returning dictionaries with `input_ids`, `attention_mask`, and `lengths` tensors.

**Returns (iter)**

- **Iterator[dict]**: Each item is a batch dictionary compatible with `ModelAdapter.forward`.


### class: ModelAdapter

```python
class ModelAdapter:
    @staticmethod
    def get_pad_token_id(model) -> int

    @staticmethod
    def get_model_input_size(model) -> int

    @staticmethod
    def quant_layers(model) -> int

    @staticmethod
    def forward(model, batch: Dict[str, torch.Tensor])

    @staticmethod
    def pick_layer(outputs, layer_index: int) -> torch.Tensor

    @staticmethod
    def pool_mean(
        embs: torch.Tensor,
        lengths: torch.Tensor,
        exclude_cls: bool = False,
        exclude_eos: bool = False
    ) -> torch.Tensor
```

Lightweight adapter to standardize model interaction (device placement, layer selection,
and pooling) across different Hugging Face architectures.

**Methods**

- **get_pad_token_id(model)**: Return the padding token ID from model config (default `0`).
- **get_model_input_size(model)**: Proxy to `get_model_input_size` utility.
- **quant_layers(model)**: Proxy to `quant_layers` utility.
- **forward(model, batch)**: Move `input_ids` and `attention_mask` to the model device and
  call the model with `output_hidden_states=True`.
- **pick_layer(outputs, layer_index)**: Extract the hidden state tensor at the given layer index.
- **pool_mean(embs, lengths, exclude_cls=False, exclude_eos=False)**: Mean-pool over token
  embeddings, optionally excluding CLS and/or final EOS positions.


### class: PerturbOps

```python
@dataclass
class PerturbOps:
    genes_to_keep: Optional[List[Union[int, str]]] = None
    genes_to_perturb: Optional[List[Union[int, str]]] = None
    perturb_fraction: Optional[float] = None
    rank_shift: Optional[str] = None
    top_k: Optional[int] = None
    pad_token_id: int = 0
```

Utility class encapsulating group and single-token perturbation rules.

**Parameters (init)**

- **genes_to_keep** (`list[int | str]` or `None`): Genes that must not be perturbed.
- **genes_to_perturb** (`list[int | str]` or `None`): Explicit perturbation targets; if `None` in
  single mode, all non-keep tokens are candidates.
- **perturb_fraction** (`float` or `None`): Fraction of eligible token positions to perturb in group mode.
- **rank_shift** (`str` or `None`): Perturbation operation label (e.g. `"delete"`).
- **top_k** (`int` or `None`): Maximum number of positions to perturb in group mode.
- **pad_token_id** (`int`): Padding ID used for deletions when `rank_shift == "delete"`.

**Methods**

- **iter_single_tokens(input_ids)**: Yield distinct tokens in the batch that should be perturbed
  (excluding `genes_to_keep` and restricted to `genes_to_perturb` if provided).
- **_mask_keep(ids)**: Internal helper to build a boolean mask of perturbable positions.
- **apply_group(input_ids)**: Replace a subset of non-keep positions with `genes_to_perturb` in a
  deterministic order, respecting `top_k` and `perturb_fraction`.
- **apply_single(input_ids, gene_tok)**: Apply a per-token perturbation (currently delete by setting
  `pad_token_id` when `rank_shift == "delete"`).


### validate_gene_token_mapping

```python
def validate_gene_token_mapping(ens2tok: dict) -> dict
```

Normalize and validate a gene→token dictionary loaded from pickle.

**Parameters**

- **ens2tok** (`dict`): Raw mapping of gene IDs to token IDs.

**Returns**

- **dict**: Cleaned mapping with string keys and integer token IDs; entries with non-integer token IDs
  are dropped and token collisions are logged.


### delete_indices

```python
def delete_indices(example: dict) -> dict
```

Delete tokens at positions in `example["perturb_index"]` and update sequence length.

**Parameters**

- **example** (`dict`): Contains `input_ids` and `perturb_index` list.

**Returns**

- **dict**: Modified example with updated `input_ids` and `length`.


### overexpress_tokens

```python
def overexpress_tokens(
    example: dict,
    max_len: int
) -> dict
```

Simulate overexpression by moving selected tokens to the front of a sequence.

**Parameters**

- **example** (`dict`): Contains `input_ids` and `tokens_to_perturb`.
- **max_len** (`int`): Maximum length to retain after reordering.

**Returns**

- **dict**: Example with tokens rearranged (overexpressed tokens at front) and updated `length`.


### calc_n_overflow

```python
def calc_n_overflow(
    max_len: int,
    length: int,
    tokens_to_perturb: list,
    indices_to_perturb: list
) -> int
```

Compute how many tokens will overflow (be pushed off the end) after overexpression.

**Parameters**

- **max_len** (`int`): Maximum sequence length.
- **length** (`int`): Original sequence length.
- **tokens_to_perturb** (`list`): Tokens being overexpressed.
- **indices_to_perturb** (`list`): Original positions of perturbed tokens.

**Returns**

- **int**: Number of tokens that would overflow past `max_len`.


### truncate_by_n_overflow

```python
def truncate_by_n_overflow(example: dict) -> dict
```

If `example["n_overflow"] > 0`, drop that many tokens from the end of `input_ids`.

**Parameters**

- **example** (`dict`): Contains `input_ids`, `length`, and `n_overflow`.

**Returns**

- **dict**: Example with truncated `input_ids` and updated `length`.


### remove_perturbed_indices_set

```python
def remove_perturbed_indices_set(
    full_original_emb: torch.Tensor,
    perturb_type: str,
    indices_to_perturb: list,
    tokens_to_perturb: list,
    lengths: list
) -> torch.Tensor
```

Remove perturbed positions from original embeddings for alignment with perturbed sequences.

**Parameters**

- **full_original_emb** (`torch.Tensor`): Original embeddings `[B, L, H]`.
- **perturb_type** (`str`): Perturbation operation type (`"delete"` or `"overexpress"`).
- **indices_to_perturb** (`list[list[int]]`): Indices per example that were perturbed.
- **tokens_to_perturb** (`list`): Tokens affected (used for overexpression semantics).
- **lengths** (`list[int]`): Original sequence lengths.

**Returns**

- **torch.Tensor**: Embeddings with perturbed positions removed and padded to equal length.


### compute_nonpadded_cell_embedding

```python
def compute_nonpadded_cell_embedding(
    full_emb: torch.Tensor,
    style: str = "mean_pool"
) -> torch.Tensor
```

Compute a cell-level embedding given a token-level embedding tensor with padding already removed.

**Parameters**

- **full_emb** (`torch.Tensor`): Embedding tensor `[B, L, H]` without padding tokens.
- **style** (`str`): Pooling style; `"mean_pool"` computes mean over the sequence dimension.

**Returns**

- **torch.Tensor**: Cell embeddings `[B, H]`.


### remove_indices_per_example

```python
def remove_indices_per_example(
    full_emb: torch.Tensor,
    lengths: torch.Tensor,
    indices_to_remove: List[List[int]]
) -> torch.Tensor
```

Remove specified token positions per example from a `[B, L, H]` tensor and repad rows.

**Parameters**

- **full_emb** (`torch.Tensor`): Input embeddings `[B, L, H]`.
- **lengths** (`torch.Tensor`): True lengths per example `[B]`.
- **indices_to_remove** (`List[List[int]]`): List of index lists; one per example.

**Returns**

- **torch.Tensor**: A new tensor `[B, L', H]` with removed positions and zero-padding as needed.


### write_cosine_sim_dict

```python
def write_cosine_sim_dict(
    cos_sims: Dict[Tuple[Union[int, Tuple[int, ...]], str], List[float]],
    out_dir: str,
    prefix: str
) -> str
```

Write a cosine similarity dictionary as newline-delimited JSON for inspection or reuse.

**Parameters**

- **cos_sims** (`dict`): Mapping from `(token or token tuple, metric)` to list of similarity values.
- **out_dir** (`str`): Output directory.
- **prefix** (`str`): Filename prefix.

**Returns**

- **str**: Path to the written JSONL file.


### read_cosine_sims

```python
def read_cosine_sims(path)
```

Load a pickled cosine similarity dictionary from disk.

**Parameters**

- **path** (`str`): Path to a pickle file created by perturbation routines.

**Returns**

- **dict**: Loaded similarity dictionary.


### gene_sims_to_dict

```python
def gene_sims_to_dict(
    input_ids: torch.Tensor,
    sims_tokenwise: torch.Tensor,
    pad_token_id: int
) -> Dict[Tuple[int, str], List[float]]
```

Aggregate token-wise similarities into a gene-level dictionary keyed by token ID.

**Parameters**

- **input_ids** (`torch.Tensor`): Token IDs `[B, L]`.
- **sims_tokenwise** (`torch.Tensor`): Similarities `[B, L]`.
- **pad_token_id** (`int`): Token ID treated as padding (ignored).

**Returns**

- **dict**: Mapping `(token_id, "gene_emb") → list[float]` of similarity values.


### write_perturbation_dictionary

```python
def write_perturbation_dictionary(
    cos_sims_dict: defaultdict,
    output_path_prefix: str
)
```

Save the perturbation cosine similarity dictionary as a pickle.

**Parameters**

- **cos_sims_dict** (`defaultdict`): Raw similarity accumulator.
- **output_path_prefix** (`str`): Prefix for the output file; `"_raw.pickle"` is appended.

**Returns**

- `None`


### Extended model support

The following helpers are used to support extended models where sequences are split
into two halves (e.g. spot vs neighbor) and overexpression is applied to both halves.

#### nonpad_len_1d

```python
def nonpad_len_1d(ids_row: list[int], pad_token_id: int) -> int
```

Count non-padding tokens in a single sequence.

- **ids_row** (`list[int]`): Token IDs (possibly padded).
- **pad_token_id** (`int`): Padding token ID.

Returns the index of the first pad token or the full length if no padding is found.


#### overexpress_tokens_extended

```python
def overexpress_tokens_extended(
    example,
    half_size: int,
    model_input_size: int
)
```

Insert `example["tokens_to_perturb"]` at the front of both spot and neighbor halves,
tracking overflow per half.

- **example** (`dict`): Contains `input_ids`, optional `pad_token_id`, and `tokens_to_perturb`.
- **half_size** (`int`): Number of positions allocated to each half.
- **model_input_size** (`int`): Global maximum sequence length.

Updates `example["input_ids"]`, `example["length"]`, and `example["n_overflow_halves"]`.


#### truncate_by_n_overflow_extended

```python
def truncate_by_n_overflow_extended(
    example,
    half_size: int
)
```

Truncate the original sequence to remove overflow positions at the end of each half.

- **example** (`dict`): Contains `input_ids` and `n_overflow_halves`.
- **half_size** (`int`): Half-length used to split the sequence.

Returns the modified example with updated `input_ids` and `length`.


#### remove_front_per_example_halves

```python
def remove_front_per_example_halves(
    hid: torch.Tensor,
    k_spot_vec: torch.Tensor,
    k_neig_vec: torch.Tensor,
    half_size: int
) -> torch.Tensor
```

Drop `k_spot_vec[b]` tokens from the front of the first half and `k_neig_vec[b]` tokens from
the front of the second half for each example.

- **hid** (`torch.Tensor`): Hidden states `[B, L, H]`.
- **k_spot_vec** (`torch.Tensor`): Number of spot tokens to drop per example `[B]`.
- **k_neig_vec** (`torch.Tensor`): Number of neighbor tokens to drop per example `[B]`.
- **half_size** (`int`): Split point between halves.

Returns a new tensor `[B, L', H]` with halves trimmed and repadded.


#### remove_front_per_example_halves_2d

```python
def remove_front_per_example_halves_2d(
    ids: torch.Tensor,
    k_spot_vec: torch.Tensor,
    k_neig_vec: torch.Tensor,
    half_size: int
) -> torch.Tensor
```

Same as `remove_front_per_example_halves`, but applied to token IDs instead of embeddings.

- **ids** (`torch.Tensor`): Token IDs `[B, L]`.
- **k_spot_vec** (`torch.Tensor`): Spot-side drops per example.
- **k_neig_vec** (`torch.Tensor`): Neighbor-side drops per example.
- **half_size** (`int`): Split point between halves.

Returns a new token ID tensor `[B, L']` with zeros as padding.


---

## Perturbation Statistics (`perturb_stats.py`)

This module aggregates raw cosine similarity pickles, performs statistical tests, and
writes CSV summaries for different analysis modes.

### Low-level helpers

These functions are intended for internal use but are documented here for completeness.

#### _is_pickle_path

```python
def _is_pickle_path(p: Union[str, os.PathLike]) -> bool
```

Return `True` if the path ends with `.pkl` or `.pickle`.

#### _load_raw_pickle

```python
def _load_raw_pickle(path: Union[str, os.PathLike]) -> dict
```

Load a single pickled raw similarity dictionary from `path`.


#### _iter_raw_pickles

```python
def _iter_raw_pickles(path_or_dir: Union[str, os.PathLike]) -> Iterable[dict]
```

Yield raw dictionaries from either a single pickle file or all `*_raw.pickle` files
within a directory.


#### _bh_fdr

```python
def _bh_fdr(pvals: Sequence[float]) -> List[float]
```

Apply Benjamini–Hochberg FDR correction to a list of p-values.


#### _mannwhitney_u_vs_zero

```python
def _mannwhitney_u_vs_zero(x: np.ndarray) -> float
```

Approximate a two-sided Mann–Whitney U test comparing `x` against a degenerate
distribution at zero, returning a p-value in `[0, 1]`.


#### _safe_mean_std

```python
def _safe_mean_std(vals: Sequence[float]) -> Tuple[float, float]
```

Compute mean and standard deviation, returning `NaN` for empty input.


#### _merge_raw_dicts

```python
def _merge_raw_dicts(dicts: List[dict]) -> dict
```

Merge a list of raw similarity dictionaries by concatenating lists of values for identical keys.


#### _split_cell_gene_subdicts

```python
def _split_cell_gene_subdicts(raw: dict) -> Tuple[dict, dict]
```

Split a raw dictionary into two sub-dictionaries: cell-level entries and gene-level entries.

- Cell dict keys: `(pert_token, "cell_emb")`.
- Gene dict keys: `(pert_token, affected_token)`.


#### _token_to_name

```python
def _token_to_name(tok: int, tok2name: Optional[Dict[int, str]]) -> str
```

Map a token ID to a human-readable gene name if `tok2name` is provided, otherwise return
the stringified token ID.


### Mode-specific aggregation functions

These helpers implement the concrete CSV outputs for each analysis mode.

#### _mode_aggregate_data

```python
def _mode_aggregate_data(
    raw_merged: dict,
    tok2name: Optional[Dict[int, str]],
    out_csv: Union[str, os.PathLike]
) -> str
```

Aggregate cell-level cosine shift statistics for a single perturbation.

Produces a CSV with columns `['Perturbed','Cosine_sim_mean','Cosine_sim_stdev','N_Detections']`.

#### _mode_aggregate_gene_shifts

```python
def _mode_aggregate_gene_shifts(
    raw_merged: dict,
    tok2name: Optional[Dict[int, str]],
    out_csv: Union[str, os.PathLike],
    tok2ens: Optional[Dict[int, str]] = None
) -> str
```

Aggregate gene-level cosine shift statistics for all `(perturbed, affected)` token pairs.

Outputs a CSV describing per-gene means, standard deviations, and detection counts, including
both gene names and ENSEMBL IDs.


#### _mode_vs_null

```python
def _mode_vs_null(
    raw_merged: dict,
    null_merged: dict,
    tok2name: Optional[Dict[int, str]],
    out_csv: Union[str, os.PathLike]
) -> str
```

Compare cell-level perturbation shifts to a null distribution built from separate runs.

Outputs per-perturbation means and standard deviations for test and null distributions,
as well as p-values and FDR-adjusted q-values.


#### _mode_goal_state_shift

```python
def _mode_goal_state_shift(
    raw_by_state: Dict[str, dict],
    state_cfg: dict,
    tok2name: Optional[Dict[int, str]],
    out_csv: Union[str, os.PathLike]
) -> str
```

Compare perturbation shifts between start and goal cell states (and optional alternative states)
using nonparametric tests, generating start/goal/alt summaries with p- and q-values.


#### _mode_mixture_model

```python
def _mode_mixture_model(
    raw_merged: dict,
    tok2name: Optional[Dict[int, str]],
    out_csv: Union[str, os.PathLike]
) -> str
```

Fit a two-component Gaussian mixture model to per-gene mean shifts and label which component
represents the high-impact (stronger negative) shift cluster, along with the fraction of genes
in that component.


### class: PerturberStats

```python
@dataclass
class PerturberStats:
    mode: str
    top_k: Optional[int] = None
    nperms: Optional[int] = None
    fdr_alpha: float = 0.05
    min_cells: int = 1
    token_dictionary_file: Optional[Dict[str, int]] = None
    gene_id_name_dict: Optional[Dict[str, str]] = None
    cell_states_to_model: Optional[dict] = None
```

Configure the type of perturbation statistics to compute and provide optional dictionaries
for mapping tokens to gene names and cell states.

**Parameters (init)**

- **mode** (`str`): Statistical analysis mode. Supported values include:
  - `"aggregate_data"`: Aggregate cell-level cosine shifts.
  - `"aggregate_gene_shifts"`: Aggregate per-gene cosine shifts.
  - `"vs_null"`: Compare perturbation shifts vs. a null distribution.
  - `"goal_state_shift"`: Compare start vs goal states (and alt states) defined in
    `cell_states_to_model`.
  - `"mixture_model"`: Fit a Gaussian mixture model to per-gene mean shifts.
- **top_k** (`int` or `None`): Optional limit on the number of strongest results to retain (reserved
  for future use; not enforced in current implementation).
- **nperms** (`int` or `None`): Number of permutations for permutation-based modes (currently unused).
- **fdr_alpha** (`float`): Target FDR threshold for significance; used to interpret q-values.
- **min_cells** (`int`): Minimum number of cells required to include a perturbation in outputs.
- **token_dictionary_file** (`dict` or `None`): Mapping ENSEMBL ID → token ID for token name resolution.
- **gene_id_name_dict** (`dict` or `None`): Mapping ENSEMBL ID → gene symbol used in outputs.
- **cell_states_to_model** (`dict` or `None`): Configuration for state-aware statistics containing keys
  such as `"state_key"`, `"start_state"`, `"goal_state"`, and optional `"alt_states"`.


#### _tok_maps

```python
def _tok_maps(self)
```

Build mapping dictionaries from token IDs to gene names and ENSEMBL IDs based on
`token_dictionary_file` and `gene_id_name_dict`. Returns `(tok2name, tok2ens)` or `(None, None)`
if required dictionaries are missing.


#### _load_and_merge

```python
def _load_and_merge(self, path_or_dir: Union[str, os.PathLike]) -> dict
```

Load one or more raw similarity pickles from a file or directory and return a merged dictionary.


#### _load_by_state

```python
def _load_by_state(self, base_dir: Union[str, os.PathLike]) -> Dict[str, dict]
```

Load state-specific raw dictionaries from subdirectories under `base_dir`, where each subdirectory
name corresponds to a state listed in `cell_states_to_model`.


### compute_stats

```python
def compute_stats(
    self,
    input_data_dir: str,
    null_data_dir: Optional[str],
    output_dir: str,
    output_prefix: str
) -> str
```

Main entry point for computing perturbation statistics and writing CSV files.

**Parameters**

- **input_data_dir** (`str`): Directory containing one or more raw pickle files produced by `Perturber`.
- **null_data_dir** (`str` or `None`): Directory containing null-distribution pickles, required for
  `"vs_null"` mode.
- **output_dir** (`str`): Directory where CSV output will be written.
- **output_prefix** (`str`): Prefix for output CSV filenames.

**Returns**

- **str**: Path to the generated CSV file for the selected `mode`.

**Description**

Depending on `self.mode`, this method:

- Loads and merges raw similarity dictionaries from `input_data_dir` (and optionally `null_data_dir`).
- Converts token IDs to gene names and ENSEMBL IDs if mapping dictionaries are available.
- Delegates to an appropriate mode-specific helper:
  - `"aggregate_data"` → `_mode_aggregate_data`
  - `"aggregate_gene_shifts"` → `_mode_aggregate_gene_shifts`
  - `"vs_null"` → `_mode_vs_null`
  - `"goal_state_shift"` → `_mode_goal_state_shift`
  - `"mixture_model"` → `_mode_mixture_model`
- Creates `output_dir` if necessary and writes a CSV summarizing the perturbation effects.
