# Gene Regulatory Network
Module to take self attention weights throughout model pretraining and construct a gene regulatory network graph

## GeneRegulatoryNetwork
```python
class GeneRegulatoryNetwork:
    def __init__(
        self,
        model_dir: str,
        dataset_path: str,
        model_type: Literal["CellClassifier","GeneClassifier","Pretrained"],
        metadata_column: Optional[str] = None,
        metadata_value: Optional[str] = None,
        num_classes: int = 0,
        threshold: float = 0.01,
        device: Optional[str] = None,
        batch_size: int = 16,
        nproc: int = 4
    )
```

Initialize a gene regulatory network extractor using attention from a pretrained BERT model.

**Parameters**

- **model\_dir** (`str`): Path to the pretrained model directory.
- **dataset\_path** (`str`): Path to a saved Hugging Face dataset directory containing an `input_ids` column.
- **model\_type** (`Literal[...]`): One of:
  - `'Pretrained'`: masked language model
  - `'GeneClassifier'`: token classification model
  - `'CellClassifier'`: sequence classification model
- **metadata\_column** (`Optional[str]`): Column name in dataset to filter on (e.g., cell type).
- **metadata\_value** (`Optional[str]`): Value in `metadata_column` to filter (e.g., `'Tcell'`).
- **num\_classes** (`int`): Number of labels (required if `model_type` is classifier).
- **threshold** (`float`): Default attention cutoff used when building the graph.
- **device** (`str`, optional): Compute device (e.g., `'cuda'` or `'cpu'`); auto-detected if `None`.
- **batch\_size** (`int`): Number of examples per batch when computing attention.
- **nproc** (`int`): Number of worker processes for data loading and filtering.

**Returns**

- Instance of `GeneRegulatoryNetwork` with model, tokenizer, and dataset loaded.

---

### load_model_and_tokenizer
```python
    def _load_model_and_tokenizer(self) -> None
```

Load model and tokenizer from `model_dir`, enabling attention outputs.

**Behavior**

- Loads `AutoConfig` with `output_attentions=True` and `output_hidden_states=True`.
- Instantiates one of:
  - `BertForMaskedLM` if `model_type=='Pretrained'`.
  - `BertForTokenClassification` if `model_type=='GeneClassifier'`.
  - `BertForSequenceClassification` if `model_type=='CellClassifier'`.
- Sends model to `device` and sets `eval()` mode.
- Loads tokenizer via `AutoTokenizer.from_pretrained`.
- Builds an `id2token` mapping from tokenizer vocabulary.

**Returns**

- None

---
### load_dataset
```python
    def _load_dataset(self) -> None
```

Load and preprocess the Hugging Face dataset from `dataset_path`.

**Behavior**

- Uses `load_from_disk` to load dataset.
- If `metadata_column` and `metadata_value` are set, filters examples accordingly.
- Ensures `attention_mask` exists; if missing, adds all-ones mask.
- Computes `seq_len` as the maximum sequence length across `input_ids`.
- Pads all `input_ids` and `attention_mask` to `seq_len` using pad token ID.
- Stores the processed dataset in `self.dataset` and sequence length in `self.seq_len`.

**Returns**

- None

---
### compute_attention
```python
    def compute_attention(self) -> None
```

Compute the average attention weight matrix across all examples.

**Behavior**

- Optionally casts model to half precision (`.half()`) if `torch.cuda.amp` available.
- Converts dataset to PyTorch tensors (`input_ids`, `attention_mask`).
- Defines a custom `collate_fn` to pad each batch to its own maximum length.
- Iterates batches through the model, collects `outputs.attentions`.
- Averages over heads and layers to produce per-example `(seq, seq)` matrices.
- Accumulates a sum over all examples, then divides by total examples.
- Stores resulting `(seq_len, seq_len)` NumPy array in `self.attention_matrix`.

**Returns**

- None

---
### build_graph
```python
    def build_graph(
        self,
        cutoff: Optional[float] = None,
        top_k: Optional[int] = None,
        percentile: Optional[float] = None,
        min_cooccurrence: Optional[int] = None
    ) -> None
```

Construct a directed graph from the averaged attention matrix.

**Parameters**

- **cutoff** (`float`, optional): Include edges where attention ≥ cutoff.
- **top\_k** (`int`, optional): For each source token, include top\_k targets by weight.
- **percentile** (`float`, optional): Include edges above the given percentile of weights.
- **min\_cooccurrence** (`int`, optional): Remove edges where source and target co-occur in fewer than this many samples.

**Behavior**

- Validates exactly one of `cutoff`, `top_k`, or `percentile` is set.
- Computes `sample_presence`: for each token ID, set of example indices in which it appears.
- Builds a token-position counts matrix, then aggregates attention via matrix multiplications.
- Normalizes aggregated scores by occurrence counts.
- Selects edges per the chosen mode.
- Optionally filters edges by co-occurrence count.
- Builds a `networkx.DiGraph` mapping token strings to edges with `weight` attributes.
- Stores graph in `self.graph`.

**Returns**

- None

---
### save_edge_list
```python
    def save_edge_list(
        self,
        output_path: str,
        gene_name_id_dictionary_file: Optional[str] = None
    ) -> None
```

Export the graph as a CSV edge list.

**Parameters**

- **output\_path** (`str`): Path to write CSV file.
- **gene\_name\_id\_dictionary\_file** (`Optional[str]`): Path to pickle mapping gene symbols to IDs; used to add human-readable names.

**Behavior**

- Writes header: `source,source_gene,target,target_gene,weight`.
- Iterates `self.graph.edges`, writing token IDs and optional gene names.

**Returns**

- None

---
### plot_network
```python
    def plot_network(
        self,
        output_path: str
    ) -> None
```

Visualize the regulatory network using a spring layout.

**Parameters**

- **output\_path** (`str`): Path to save the PNG plot.

**Behavior**

- Computes `spring_layout` positions.
- Draws nodes (size 50), edges (width scaled by weight), and labels (font size 6).
- Saves figure with DPI 300.

**Returns**

- None

