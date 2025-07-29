## Module: Perturbation 

```python
class InSilicoPerturber:
    def __init__(
        self,
        mode: str = 'spot',
        perturb_type: str = 'delete',
        perturb_rank_shift: Optional[int] = None,
        genes_to_perturb: Union[str, List[str]] = 'all',
        combos: int = 0,
        anchor_gene: Optional[str] = None,
        model_type: str = 'Pretrained',
        num_classes: int = 0,
        emb_mode: str = 'cell',
        cell_emb_style: str = 'mean_pool',
        filter_data: Optional[Dict[str, List[Any]]] = None,
        cell_states_to_model: Optional[Dict[str, Any]] = None,
        state_embs_dict: Optional[Dict[str, torch.Tensor]] = None,
        max_ncells: Optional[int] = None,
        cell_inds_to_perturb: Union[str, Dict[str, int]] = 'all',
        emb_layer: int = -1,
        forward_batch_size: int = 100,
        nproc: int = 4,
        token_dictionary_file: str
    )
```

Instantiate an in silico perturber to simulate gene-level or cell-level perturbations.

**Parameters**

- **mode** (`str`): Dataset tokenization mode: `'spot'` or `'neighborhood'`.
- **perturb\_type** (`str`): Perturbation operation: `'delete'`, `'overexpress'`, `'inhibit'`, or `'activate'`.
- **perturb\_rank\_shift** (`int` or `None`): Rank shift for `'inhibit'`/`'activate'` modes (ignored for delete/overexpress).
- **genes\_to\_perturb** (`'all'` or `List[str]`): Gene(s) to perturb; `'all'` perturbs entire input.
- **combos** (`int`): 0 for individual perturbations; 1 for pairwise combinations with `anchor_gene`.
- **anchor\_gene** (`str` or `None`): ENSEMBL ID for anchor in combination perturbations.
- **model\_type** (`str`): `'Pretrained'`, `'GeneClassifier'`, or `'CellClassifier'`.
- **num\_classes** (`int`): Number of classes if using a classifier model; 0 otherwise.
- **emb\_mode** (`str`): `'cell'` or `'cell_and_gene'`, output embeddings scope.
- **cell\_emb\_style** (`str`): Pooling method for cell embeddings (currently `'mean_pool'`).
- **filter\_data** (`dict` or `None`): Column-to-values map to pre-filter dataset.
- **cell\_states\_to\_model** (`dict` or `None`): Dict with keys `state_key`, `start_state`, `goal_state`, `alt_states` for state modeling.
- **state\_embs\_dict** (`dict` or `None`): Map from state names to target embeddings (`torch.Tensor`).
- **max\_ncells** (`int` or `None`): Max number of cells to process.
- **cell\_inds\_to\_perturb** (`'all'` or `{'start':int,'end':int}`): Slice indices after filtering.
- **emb\_layer** (`int`): Layer offset for embedding extraction (`-1` for penultimate, `0` for final).
- **forward\_batch\_size** (`int`): Batch size for forward passes.
- **nproc** (`int`): CPU processes for data operations.
- **token\_dictionary\_file** (`str`): Path to pickle mapping gene IDs to token IDs.

**Returns**

- Instance of `InSilicoPerturber`.

---

```python
def validate_options(self) -> None
```

Validate constructor options for type, compatibility, and development status.

**Parameters**

- **self**: Perturber instance.

**Raises**

- **ValueError**: On invalid or incompatible options.

---

```python
def perturb_data(
    self,
    model_directory: str,
    input_data_file: str,
    output_directory: str,
    output_prefix: str
) -> None
```

Run the full perturbation workflow: load model, filter data, apply perturbs.

**Parameters**

- **model\_directory** (`str`): Path to pretrained or classifier model.
- **input\_data\_file** (`str`): Path to input HF Dataset.
- **output\_directory** (`str`): Directory for output files.
- **output\_prefix** (`str`): Prefix for result filenames.

**Returns**

- `None`

---

```python
def apply_additional_filters(
    self,
    filtered_input_data: Dataset
) -> Dataset
```

Apply state-based, token-based, downsampling, and slicing filters.

**Parameters**

- **filtered\_input\_data** (`Dataset`): Dataset after initial metadata filters.

**Returns**

- **Dataset**: Further filtered, sorted, and sliced dataset.

---

```python
def isp_perturb_set(
    self,
    model: torch.nn.Module,
    dataset: Dataset,
    layer_to_quant: int,
    prefix: str
) -> None
```

Perform group perturbations on the dataset and compute cosine similarities.

**Parameters**

- **model** (`nn.Module`): Model with `output_hidden_states=True`.
- **dataset** (`Dataset`): Filtered input dataset.
- **layer\_to\_quant** (`int`): Layer index offset for embeddings.
- **prefix** (`str`): Filename prefix for saving results.

**Returns**

- `None`

---

```python
def isp_perturb_all(
    self,
    model: torch.nn.Module,
    dataset: Dataset,
    layer_to_quant: int,
    prefix: str
) -> None
```

Perturb each cell individually, compute embeddings, and save results in batches.

**Parameters**

- **model** (`nn.Module`): Model for embedding extraction.
- **dataset** (`Dataset`): Filtered input dataset.
- **layer\_to\_quant** (`int`): Layer index offset.
- **prefix** (`str`): Filename prefix for batch outputs.

**Returns**

- `None`

---

```python
def update_perturbation_dictionary(
    self,
    cos_sims: defaultdict,
    sims: torch.Tensor,
    _data: Any,
    _indices: Any,
    genes: Optional[List[int]] = None
) -> defaultdict
```

Append cosine similarity values to a results dictionary.

**Parameters**

- **cos\_sims** (`defaultdict`): Accumulator mapping keys to lists of similarity values.
- **sims** (`torch.Tensor`): Similarity scores tensor.
- **\_data** (`Any`): Unused; for signature compatibility.
- **\_indices** (`Any`): Unused; for signature compatibility.
- **genes** (`List[int]` or `None`): Token IDs of genes corresponding to `sims` rows.

**Returns**

- **defaultdict**: Updated similarity accumulator.

## Module: perturb_utils.py

```python
def load_and_filter(
    filter_data: Optional[Dict[str, List[Any]]],
    nproc: int,
    input_data_file: str
) -> Dataset
```

Load a Hugging Face dataset and apply metadata filters.

**Parameters**

* **filter\_data** (`Optional[Dict[str, List[Any]]]`): Metadata filters (column names to allowed values). If `None`, no filtering.
* **nproc** (`int`): Number of parallel processes for filtering.
* **input\_data\_file** (`str`): Path to a saved HF Dataset directory.

**Returns**

* **Dataset**: Filtered dataset.

---

```python
def filter_by_dict(
    data: Dataset,
    filter_data: Dict[str, List[Any]],
    nproc: int
) -> Dataset
```

Retain only examples matching specified metadata criteria.

**Parameters**

* **data** (`Dataset`): Input dataset.
* **filter\_data** (`Dict[str, List[Any]]`): Mapping from metadata column to allowed values.
* **nproc** (`int`): Number of workers.

**Returns**

* **Dataset**: Filtered dataset.

---

```python
def filter_data_by_tokens(
    filtered_input_data: Dataset,
    tokens: Sequence[int],
    nproc: int
) -> Dataset
```

Keep examples containing all specified token IDs in their `input_ids`.

**Parameters**

* **filtered\_input\_data** (`Dataset`): Dataset with `input_ids` lists.
* **tokens** (`Sequence[int]`): Token IDs required.
* **nproc** (`int`): Number of processes.

**Returns**

* **Dataset**: Subset containing required tokens.

---

```python
def filter_data_by_tokens_and_log(
    filtered_input_data: Dataset,
    tokens: Sequence[int],
    nproc: int,
    filtered_tokens_categ: str
) -> Dataset
```

Filter by tokens and log the remaining count, raising an error if none remain.

**Parameters**

* **filtered\_input\_data** (`Dataset`): Dataset before token filtering.
* **tokens** (`Sequence[int]`): Token IDs to require.
* **nproc** (`int`): Number of workers.
* **filtered\_tokens\_categ** (`str`): Description for logging.

**Returns**

* **Dataset**: Filtered dataset.

---

```python
def filter_data_by_start_state(
    filtered_input_data: Dataset,
    cell_states_to_model: Dict[str, Any],
    nproc: int
) -> Dataset
```

Restrict data to a specified starting state.

**Parameters**

* **filtered\_input\_data** (`Dataset`): Pre-filtered dataset.
* **cell\_states\_to\_model** (`Dict[str, Any]`): Must include:

  * `state_key` (`str`): Metadata column name.
  * `start_state` (`Any`): Desired value in that column.
* **nproc** (`int`): Number of processes.

**Returns**

* **Dataset**: Subset matching the start state.

---

```python
def slice_by_inds_to_perturb(
    dataset: Dataset,
    inds: Dict[str, int]
) -> Dataset
```

Select a contiguous slice of examples by start/end indices.

**Parameters**

* **dataset** (`Dataset`): Input dataset.
* **inds** (`Dict[str, int]`): Contains `start` (inclusive) and `end` (exclusive) indices.

**Returns**

* **Dataset**: Sliced dataset.

---

```python
def load_model(
    model_type: str,
    num_classes: int,
    model_directory: str,
    mode: str
) -> nn.Module
```

Load a transformer model for evaluation or classification.

**Parameters**

* **model\_type** (`str`): `'Pretrained'`, `'GeneClassifier'`, or `'CellClassifier'`.
* **num\_classes** (`int`): Number of labels (for classifier types).
* **model\_directory** (`str`): Path or HF model ID.
* **mode** (`str`): `'eval'` to enable hidden states, else `'train'`.

**Returns**

* **nn.Module**: Loaded model on CUDA with appropriate config.

---

```python
def quant_layers(
    model: nn.Module
) -> int
```

Count transformer encoder layers in the model.

**Parameters**

* **model** (`nn.Module`): Transformer model.

**Returns**

* **int**: Number of layers.

---

```python
def get_model_emb_dims(
    model: nn.Module
) -> int
```

Get the hidden size (embedding dimension) of the model.

**Parameters**

* **model** (`nn.Module`): Transformer model.

**Returns**

* **int**: Size of embeddings.

---

```python
def get_model_input_size(
    model: nn.Module
) -> int
```

Retrieve the maximum sequence length of the model.

**Parameters**

* **model** (`nn.Module`): Transformer model.

**Returns**

* **int**: Max position embeddings.

---

```python
def measure_length(
    example: Dict[str, Any]
) -> Dict[str, Any]
```

Compute and set the `length` field for an example.

**Parameters**

* **example** (`Dict[str, Any]`): Must include `input_ids`.

**Returns**

* **Dict\[str, Any]**: Example with updated `length`.

---

```python
def downsample_and_sort(
    data: Dataset,
    max_ncells: Optional[int]
) -> Dataset
```

Randomly downsample up to `max_ncells` and sort by sequence length.

**Parameters**

* **data** (`Dataset`): Input dataset.
* **max\_ncells** (`Optional[int]`): Max examples; if `None`, no downsampling.

**Returns**

* **Dataset**: Downsampled, length-sorted dataset.

---

```python
def stratified_downsample_and_sort(
    data: Dataset,
    max_ncells: int,
    stratify_by: str
) -> Dataset
```

Perform stratified sampling to reach `max_ncells`, then sort by length.

**Parameters**

* **data** (`Dataset`): Dataset with grouping column.
* **max\_ncells** (`int`): Target total samples.
* **stratify\_by** (`str`): Column for stratification.

**Returns**

* **Dataset**: Stratified, downsampled, sorted dataset.

---

```python
def forward_pass_single_cell(
    model: nn.Module,
    example_cell: Dict[str, Any],
    layer_to_quant: int
) -> torch.Tensor
```

Extract embeddings for a single example.

**Parameters**

* **model** (`nn.Module`): Model with `output_hidden_states=True`.
* **example\_cell** (`Dict[str, Any]`): Must include `input_ids` and `attention_mask`.
* **layer\_to\_quant** (`int`): Index of layer to return.

**Returns**

* **torch.Tensor**: Embedding tensor.

---

```python
def delete_indices(
    example: Dict[str, Any]
) -> Dict[str, Any]
```

Remove tokens specified in `example['perturb_index']`.

**Parameters**

* **example** (`Dict[str, Any]`): Contains `input_ids` and `perturb_index`.

**Returns**

* **Dict\[str, Any]**: Example with tokens removed and updated `length`.

---

```python
def overexpress_indices(
    example: Dict[str, Any]
) -> Dict[str, Any]
```

Simulate overexpression by duplicating tokens in `perturb_index`.

**Parameters**

* **example** (`Dict[str, Any]`): Contains `input_ids` and `perturb_index`.

**Returns**

* **Dict\[str, Any]**: Example with tokens rearranged and updated `length`.

---

```python
def make_perturbation_batch(
    example_cell: Dict[str, Any],
    perturb_type: str,
    tokens_to_perturb: Union[str, List[int]],
    anchor_token: Any,
    combo_lvl: int,
    num_proc: int
) -> Tuple[Dataset, List[List[int]]]
```

Generate a dataset of perturbed examples and corresponding perturbation indices.

**Parameters**

* **example\_cell** (`Dict[str, Any]`): Single-example batch with `input_ids` and `length`.
* **perturb\_type** (`str`): `'delete'` or `'overexpress'`.
* **tokens\_to\_perturb** (`str` or `List[int]`): Token IDs or `'all'`.
* **anchor\_token** (`Any`): Anchor ID for combo perturbations.
* **combo\_lvl** (`int`): Combination level (0=single, >0=group).
* **num\_proc** (`int`): Processes for mapping.

**Returns**

* **Dataset**: Perturbation examples.
* **List\[List\[int]]**: Indices used for each perturbation.

---

```python
def make_comparison_batch(
    original_emb_batch: torch.Tensor,
    indices_to_perturb: List[List[int]],
    perturb_group: bool
) -> torch.Tensor
```

Create aligned batches for comparing original and perturbed embeddings.

**Parameters**

* **original\_emb\_batch** (`torch.Tensor`): Batch of original embeddings.
* **indices\_to\_perturb** (`List[List[int]]`): Token indices removed/added.
* **perturb\_group** (`bool`): Group perturb across multiple cells.

**Returns**

* **torch.Tensor**: Embeddings aligned for comparison.

---

```python
def pad_tensor_list(
    tensor_list: List[torch.Tensor],
    dynamic_or_constant: Union[str, int],
    pad_token_id: float,
    model_input_size: int,
    dim: Optional[int] = None,
    padding_func: Callable = pad_2d_tensor
) -> torch.Tensor
```

Pad a list of tensors to a common length.

**Parameters**

* **tensor\_list** (`List[torch.Tensor]`): Tensors to pad.
* **dynamic\_or\_constant** (`str` or `int`): `'dynamic'` or fixed length.
* **pad\_token\_id** (`float`): Value for padding.
* **model\_input\_size** (`int`): Default max length.
* **dim** (`int`, optional): Dimension to pad.
* **padding\_func** (`Callable`): Specific pad function.

**Returns**

* **torch.Tensor**: Stacked tensor batch.

---

```python
def gen_attention_mask(
    minibatch_encoding: Dict[str, List[int]],
    max_len: Optional[int] = None
) -> torch.Tensor
```

Generate attention masks from example lengths.

**Parameters**

* **minibatch\_encoding** (`Dict[str, List[int]]`): Contains `length` list.
* **max\_len** (`Optional[int]`): Override maximum length.

**Returns**

* **torch.Tensor**: Attention mask tensor.

---

```python
def mean_nonpadding_embs(
    embs: torch.Tensor,
    original_lens: torch.Tensor,
    dim: int = 1
) -> torch.Tensor
```

Compute mean embeddings excluding padded positions.

**Parameters**

* **embs** (`torch.Tensor`): Embedding batch.
* **original\_lens** (`torch.Tensor`): True lengths per example.
* **dim** (`int`): Dimension of sequence.

**Returns**

* **torch.Tensor**: Mean pooled embeddings.

---

```python
def quant_cos_sims(
    perturbation_emb: torch.Tensor,
    original_emb: torch.Tensor,
    cell_states_to_model: Optional[Dict[str, Any]] = None,
    state_embs_dict: Optional[Dict[str, torch.Tensor]] = None,
    emb_mode: str = 'gene'
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]
```

Compute cosine similarity shifts between perturbed and original embeddings.

**Parameters**

* **perturbation\_emb** (`torch.Tensor`): Perturbed embeddings.
* **original\_emb** (`torch.Tensor`): Original embeddings.
* **cell\_states\_to\_model** (`Optional[Dict]`): Dict with state comparison info.
* **state\_embs\_dict** (`Optional[Dict]`): Embeddings for target states.
* **emb\_mode** (`str`): `'gene'` or `'cell'` mode.

**Returns**

* **torch.Tensor**: Cosine similarities for `'gene'` mode.
* **Dict\[str, torch.Tensor]**: Shift maps for `'cell'` mode.

---

```python
def write_perturbation_dictionary(
    cos_sims_dict: defaultdict,
    output_path_prefix: str
) -> None
```

Save raw cosine similarity results to a pickle.

**Parameters**

* **cos\_sims\_dict** (`defaultdict`): Similarity accumulator.
* **output\_path\_prefix** (`str`): Prefix for output file.

**Returns**

* `None`

---

```python
class GeneIdHandler:
    def __init__(
        self,
        raise_errors: bool = False,
        token_dictionary_file: str,
        gene_id_name_dict: str
    )
```

Utility to convert between ENSEMBL IDs, tokens, and gene symbols.

**Parameters**

* **raise\_errors** (`bool`): If `True`, missing keys raise KeyError; else return input.
* **token\_dictionary\_file** (`str`): Path to pickle mapping gene→token.
* **gene\_id\_name\_dict** (`str`): Path to pickle mapping gene ID→symbol.

**Methods**

* **ens\_to\_token(ens\_id)**: Map ENSEMBL to token ID.
* **token\_to\_ens(token)**: Reverse map.
* **ens\_to\_symbol(ens\_id)**: Map ENSEMBL to gene symbol.
* **symbol\_to\_ens(symbol)**: Reverse map.
* **token\_to\_symbol(token)**: Token → symbol.
* **symbol\_to\_token(symbol)**: Symbol → token.

**Returns**

* Instance of `GeneIdHandler`.

## Module: perturb\_stats.py

```python
def invert_dict(
    dictionary: Dict[Any, Any]
) -> Dict[Any, Any]
```

Swap keys and values in a dictionary.

**Parameters**

* **dictionary** (`Dict[Any, Any]`): Original mapping from keys to values.

**Returns**

* **Dict\[Any, Any]**: A new dictionary where each original value is now a key, and each original key is its corresponding value.

---

```python
def read_dict(
    cos_sims_dict: Dict[Any, Any],
    cell_or_gene_emb: str,
    anchor_token: Any
) -> List[Dict[Any, Any]]
```

Extract structured cell or gene embedding entries from a raw cosine-similarity dictionary.

**Parameters**

* **cos\_sims\_dict** (`Dict[Any, Any]`): Raw similarity results mapping tokens or token pairs to score lists.
* **cell\_or\_gene\_emb** (`str`): Mode flag, either `'cell'` or `'gene'`, to select which entries to extract.
* **anchor\_token** (`Any`): Reference token used when filtering gene-mode entries; if `None`, returns all non-empty gene entries.

**Returns**

* **List\[Dict\[Any, Any]]**: A list containing a single filtered dictionary of embeddings.

---

```python
def read_dictionaries(
    input_data_directory: str,
    cell_or_gene_emb: str,
    anchor_token: Any,
    cell_states_to_model: Optional[Dict[str, Any]],
    pickle_suffix: str
) -> Union[List[Dict[Any, Any]], Dict[Any, Dict[Any, Any]]]
```

Load and aggregate pickled perturbation result dictionaries.

**Parameters**

* **input\_data\_directory** (`str`): Path to directory containing raw `.pickle` files.
* **cell\_or\_gene\_emb** (`str`): `'cell'` or `'gene'` mode for entry extraction.
* **anchor\_token** (`Any`): Token reference for gene-mode filtering; may be `None`.
* **cell\_states\_to\_model** (`Dict[str, Any]`, optional): Specifies state-key and state values for aggregation; if `None`, returns a flat list.
* **pickle\_suffix** (`str`): Filename suffix (e.g., `"_raw.pickle"`) to identify relevant files.

**Returns**

* **List\[Dict\[Any, Any]]**: If `cell_states_to_model` is `None`, list of per-file dictionaries.
* **Dict\[Any, Dict\[Any, Any]]**: If modeling states, maps each state to its aggregated dictionary.

---

```python
def get_gene_list(
    dict_list: Union[List[Dict[Any, Any]], Dict[Any, Dict[Any, Any]]],
    mode: str
) -> List[Any]
```

Compile a sorted list of unique gene identifiers from aggregated entries.

**Parameters**

* **dict\_list** (`List[Dict]` or `Dict`): Output from `read_dictionaries` containing similarity records.
* **mode** (`str`): `'cell'` or `'gene'`, to select token position (0 or 1) when extracting keys.

**Returns**

* **List\[Any]**: Sorted list of unique gene tokens or cell identifiers.

---

```python
def token_tuple_to_ensembl_ids(
    token_tuple: Union[Tuple[int, ...], int],
    gene_token_id_dict: Dict[int, str]
) -> Union[Tuple[Optional[str], ...], Optional[str]]
```

Map one or more token IDs to Ensembl gene IDs using a lookup.

**Parameters**

* **token\_tuple** (`int` or `Tuple[int, ...]`): Single token ID or tuple of token IDs.
* **gene\_token\_id\_dict** (`Dict[int, str]`): Mapping from token ID to Ensembl ID.

**Returns**

* **Tuple\[str, ...]** or **str**: Corresponding Ensembl IDs, or `None` where no mapping exists.

---

```python
def n_detections(
    token: int,
    dict_list: List[Dict[Any, Any]],
    mode: str,
    anchor_token: Any
) -> int
```

Count total similarity entries for a given token across all dictionaries.

**Parameters**

* **token** (`int`): Token ID to tally.
* **dict\_list** (`List[Dict]`): List of similarity dictionaries.
* **mode** (`str`): `'cell'` counts `(token, 'cell_emb')`; `'gene'` counts `(anchor_token, token)`.
* **anchor\_token** (`Any`): Reference token for gene mode.

**Returns**

* **int**: Total number of detections.

---

```python
def get_fdr(
    pvalues: Sequence[float]
) -> List[float]
```

Adjust a list of p-values for false discovery rate via Benjamini–Hochberg.

**Parameters**

* **pvalues** (`Sequence[float]`): Raw p-values from statistical tests.

**Returns**

* **List\[float]**: FDR-corrected p-values, same order as input.

---

```python
def get_impact_component(
    test_value: float,
    gaussian_mixture_model: GaussianMixture
) -> int
```

Determine which component (impact vs. non-impact) a value belongs to in a 2-component GMM.

**Parameters**

* **test\_value** (`float`): Observed shift statistic.
* **gaussian\_mixture\_model** (`GaussianMixture`): Fitted 2-component model.

**Returns**

* **int**: Component index (`0` or `1`) indicating impact group.

---

```python
def isp_aggregate_grouped_perturb(
    cos_sims_df: pd.DataFrame,
    dict_list: List[Dict[Any, Any]],
    genes_perturbed: List[Any]
) -> pd.DataFrame
```

Aggregate cosine-shift data for a specific gene or group perturbation across cells.

**Parameters**

* **cos\_sims\_df** (`pd.DataFrame`): DataFrame with columns `'Gene'`, `'Gene_name'`, `'Ensembl_ID'`.
* **dict\_list** (`List[Dict]`): Similarity dictionaries for each cell.
* **genes\_perturbed** (`List[Any]`): Gene IDs in the perturbation group.

**Returns**

* **pd.DataFrame**: Concatenated DataFrame with `'Cosine_sim'` and `'Gene'` columns for each perturbation.

---

```python
def find(
    variable: Any,
    x: Any
) -> bool
```

Utility to test membership or equality robustly.

**Parameters**

* **variable** (`Any`): Value or iterable to search.
* **x** (`Any`): Element to test for.

**Returns**

* **bool**: `True` if `x` equals or is contained in `variable`, else `False`.

---

```python
def isp_aggregate_gene_shifts(
    cos_sims_df: pd.DataFrame,
    dict_list: List[Dict[Any, Any]],
    gene_token_id_dict: Dict[int, str],
    gene_id_name_dict: Dict[str, str]
) -> pd.DataFrame
```

Summarize mean, standard deviation, and count of cosine shifts per gene.

**Parameters**

* **cos\_sims\_df** (`pd.DataFrame`): Initial gene list DataFrame.
* **dict\_list** (`List[Dict]`): Similarity dictionaries.
* **gene\_token\_id\_dict** (`Dict[int, str]`): Token→Ensembl ID mapping.
* **gene\_id\_name\_dict** (`Dict[str, str]`): Ensembl ID→gene name mapping.

**Returns**

* **pd.DataFrame**: Table with columns `['Perturbed','Gene_name','Ensembl_ID','Affected','Affected_gene_name','Affected_Ensembl_ID','Cosine_sim_mean','Cosine_sim_stdev','N_Detections']`.

---

```python
def isp_stats_to_goal_state(
    cos_sims_df: pd.DataFrame,
    result_dict: Dict[Any, List[float]],
    cell_states_to_model: Dict[str, Any],
    genes_perturbed: Union[str, List[Any]]
) -> pd.DataFrame
```

Compute shifts for perturbations toward goal vs. alternate or random states.

**Parameters**

* **cos\_sims\_df** (`pd.DataFrame`): Base gene list DataFrame.
* **result\_dict** (`Dict`): Aggregated similarity lists keyed by state.
* **cell\_states\_to\_model** (`Dict[str, Any]`): Contains `'state_key','start_state','goal_state','alt_states'`.
* **genes\_perturbed** (`'all'` or `List[Any]`): Genes tested.

**Returns**

* **pd.DataFrame**: Shifts and p-values (and FDR) for goal and alternate states.

---

```python
def isp_stats_vs_null(
    cos_sims_df: pd.DataFrame,
    dict_list: List[Dict[Any, Any]],
    null_dict_list: List[Dict[Any, Any]]
) -> pd.DataFrame
```

Compare perturbation shifts against a null distribution.

**Parameters**

* **cos\_sims\_df** (`pd.DataFrame`): Base gene list DataFrame.
* **dict\_list** (`List[Dict]`): Test similarity dictionaries.
* **null\_dict\_list** (`List[Dict]`): Null similarity dictionaries.

**Returns**

* **pd.DataFrame**: Test vs. null shifts, p-values, and FDR in columns `['Test_avg_shift','Null_avg_shift','Test_vs_null_avg_shift','Test_vs_null_pval','Test_vs_null_FDR','Sig']`.

---

```python
def isp_stats_mixture_model(
    cos_sims_df: pd.DataFrame,
    dict_list: List[Dict[Any, Any]],
    combos: int,
    anchor_token: Any
) -> pd.DataFrame
```

Fit a 2-component GMM to identify impact vs. non-impact perturbations.

**Parameters**

* **cos\_sims\_df** (`pd.DataFrame`): Base gene list DataFrame.
* **dict\_list** (`List[Dict]`): Similarity dictionaries.
* **combos** (`int`): 0 for single-gene, 1 for pairwise perturbations.
* **anchor\_token** (`Any`): Reference token for combos.

**Returns**

* **pd.DataFrame**: GMM component assignments and related statistics, including `['Impact_component','Impact_component_percent','N_Detections']`.

---

### Class: InSilicoPerturberStats

```python
class InSilicoPerturberStats:
    def __init__(
        self,
        mode: str = 'mixture_model',
        genes_perturbed: Union[str, List[Any]] = 'all',
        combos: int = 0,
        anchor_gene: Optional[str] = None,
        cell_states_to_model: Optional[Dict[str, Any]] = None,
        pickle_suffix: str = '_raw.pickle',
        token_dictionary_file: str | None = None,
        gene_name_id_dictionary_file: str | None = None
    )
```

Configure the type of perturbation statistics to compute and load necessary mappings.

**Parameters**

* **mode** (`str`): One of `{'goal_state_shift','vs_null','mixture_model','aggregate_data','aggregate_gene_shifts'}`.
* **genes\_perturbed** (`'all'` or `List[Any]`): Genes to evaluate.
* **combos** (`int`): Combination level (0=single, 1=pairwise, 2=triplet).
* **anchor\_gene** (`str`, optional): ENSEMBL ID for anchor gene in combos.
* **cell\_states\_to\_model** (`Dict[str, Any]`, optional): Dict with keys `'state_key','start_state','goal_state','alt_states'`.
* **pickle\_suffix** (`str`): Suffix to identify raw pickle files.
* **token\_dictionary\_file** (`str`, optional): Path to token→ID pickle.
* **gene\_name\_id\_dictionary\_file** (`str`, optional): Path to gene symbol→ID pickle.

```python
    def validate_options(self) -> None
```

Validate initialization parameters and enforce compatibility.

**Raises**

* **ValueError**: On invalid or incompatible options.

```python
    def get_stats(
        self,
        input_data_directory: str,
        null_dist_data_directory: Optional[str],
        output_directory: str,
        output_prefix: str,
        null_dict_list: Optional[List[Dict[Any, Any]]] = None
    ) -> None
```

Compute chosen statistics and write results to CSV.

**Parameters**

* **input\_data\_directory** (`str`): Directory of raw pickle inputs.
* **null\_dist\_data\_directory** (`str`, optional): Directory of null distribution pickles.
* **output\_directory** (`str`): Output directory for CSV.
* **output\_prefix** (`str`): Prefix for CSV filename.
* **null\_dict\_list** (`List[Dict]`, optional): Preloaded null dictionaries.

```python
    def token_to_gene_name(self, item: Union[int, Tuple[int, ...]]) -> Union[str, Tuple[str, ...]]
```

Map a token ID or tuple of IDs to gene symbol(s).

**Parameters**

* **item** (`int` or `Tuple[int, ...]`): Token(s) to convert.

**Returns**

* **str** or **Tuple\[str, ...]**: Corresponding gene symbol(s).
