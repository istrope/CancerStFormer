## Classifier


### class GenericClassifier

```python
class GenericClassifier:
    def __init__(
        self,
        metadata_column: str,
        classifier_type: str = 'sequence',
        gene_class_dict: dict | None = None,
        label_mapping: dict[str, int] | None = None,
        token_dictionary_file: str | None = None,
        filter_data: dict | None = None,
        rare_threshold: float = 0.0,
        max_examples: int | None = None,
        max_examples_per_class: int | None = None,
        training_args: dict | None = None,
        ray_config: dict | None = None,
        freeze_layers: int = 0,
        forward_batch_size: int = 48,
        nproc: int = 4
    )
```
Configure and instantiate a sequence or token classification pipeline.

**Description**  
Stores settings for data filtering, label mapping, training arguments (with defaults), and optional Ray Tune hyperparameter search.

**Parameters**  
- **metadata_column** (`str`): Column name in dataset holding original labels.  
- **classifier_type** (`str`, default `'sequence'`): `'sequence'` for one label per example; `'token'` for per-token labels.  
- **gene_class_dict** (`dict`, optional): Required if `classifier_type` is `'token'`; maps class names to gene token lists.  
- **label_mapping** (`dict[str,int]`, optional): Predefined mapping from label values to integer IDs.  
- **token_dictionary_file** (`str`, optional): Path to token vocabulary pickle (required for token classification).  
- **filter_data** (`dict`, optional): Metadata filters to apply before labeling.  
- **rare_threshold** (`float`, default `0.0`): Drop labels whose frequency falls below this fraction.  
- **max_examples** (`int`, optional): Global cap on number of examples after filtering.  
- **max_examples_per_class** (`int`, optional): Cap per-class when downsampling.  
- **training_args** (`dict`, optional): Overrides for Hugging Face `TrainingArguments`.  
- **ray_config** (`dict`, optional): Ranges for hyperparameters to tune via Ray Tune.  
- **freeze_layers** (`int`, default `0`): Number of initial transformer layers to freeze during training.  
- **forward_batch_size** (`int`, default `48`): Batch size used for evaluation and prediction.  
- **nproc** (`int`, default `4`): Number of processes for dataset operations.

---

#### prepare_data

```python
prepare_data(
    input_data: str | Dataset,
    output_directory: str,
    output_prefix: str
) -> tuple[str, str]
```
Filter, label, and save a Hugging Face dataset for classification.

**Description**  
Applies metadata filters, removes rare classes, downsamples, maps original labels to integer `label`, and writes the labeled dataset and label mapping to disk.

**Parameters**  
- **input_data** (`str` or `Dataset`): Path or in-memory Hugging Face dataset.  
- **output_directory** (`str`): Directory where outputs will be saved.  
- **output_prefix** (`str`): Filename prefix for saved dataset and mapping.

**Returns**  
- Tuple of (`dataset_path`, `label_map_path`), where:
  - **dataset_path** (`str`): Path to `{output_prefix}_labeled.dataset`.  
  - **label_map_path** (`str`): Path to `{output_prefix}_id_class_dict.pkl`.

---

#### _load_tokenizer

```python
_load_tokenizer(
    name_or_path: str
) -> PreTrainedTokenizerFast | AutoTokenizer
```
Load a tokenizer from a path or pretrained name.

**Description**  
Determines if `name_or_path` points to a pickle (custom vocab) or a standard pretrained identifier, returning the appropriate tokenizer.

**Parameters**  
- **name_or_path** (`str`): Path to tokenizer pickle or Hugging Face model name.

**Returns**  
- **PreTrainedTokenizerFast** or **AutoTokenizer** instance.

---

#### train

```python
train(
    model_checkpoint: str,
    dataset_path: str,
    output_directory: str,
    eval_dataset: str | Dataset | None = None,
    n_trials: int = 0,
    test_size: float = 0.2,
    tokenizer_name_or_path: str | None = None
) -> Trainer
```
Train or hyperparameter-tune a classification model.

**Description**  
Auto-splits train/test if needed, loads datasets, configures tokenizer and collator, initializes model and `Trainer`, optionally runs Ray Tune, and trains. Saves final model and predictions.

**Parameters**  
- **model_checkpoint** (`str`): Pretrained model name or local path.  
- **dataset_path** (`str`): Directory of labeled dataset from `prepare_data()`.  
- **output_directory** (`str`): Directory to save checkpoints and outputs.  
- **eval_dataset** (`str` | `Dataset`, optional): Separate eval set; if None, auto-splits by `test_size`.  
- **n_trials** (`int`, default `0`): Number of Ray Tune trials (requires `ray_config`).  
- **test_size** (`float`, default `0.2`): Fraction of data for test split.  
- **tokenizer_name_or_path** (`str`, optional): Override tokenizer source.

**Returns**  
- **Trainer**: A Hugging Face `Trainer` instance after training.

---

#### _ray_tune

```python
_ray_tune(
    model_checkpoint: str,
    dataset_path: str,
    output_directory: str,
    eval_dataset_path: str | None,
    n_trials: int,
    test_size: float,
    tokenizer_name_or_path: str | None
) -> ExperimentAnalysis
```
Run Ray Tune hyperparameter search and save best model.

**Description**  
Sets up Ray, defines search space from `ray_config`, performs hyperparameter search, and saves the best checkpoint and tokenizer.

**Parameters**  
- **model_checkpoint** (`str`): Pretrained model name or path.  
- **dataset_path** (`str`): Path to labeled dataset.  
- **output_directory** (`str`): Directory for tune results and best model.  
- **eval_dataset_path** (`str` | None): Optional eval dataset path.  
- **n_trials** (`int`): Number of search trials.  
- **test_size** (`float`): Fraction for auto-split if no eval dataset.  
- **tokenizer_name_or_path** (`str` | None): Tokenizer override.

**Returns**  
- **ExperimentAnalysis** from the Ray Tune run.

---

#### evaluate

```python
evaluate(
    model_directory: str,
    eval_dataset_path: str,
    id_class_dict_file: str,
    output_directory: str,
    tokenizer_name_or_path: str | None = None
) -> dict
```
Evaluate a trained classification model.

**Description**  
Loads model, tokenizer, and dataset, sets up a `Trainer`, runs evaluation, and logs metrics.

**Parameters**  
- **model_directory** (`str`): Directory containing the saved model and tokenizer.  
- **eval_dataset_path** (`str`): Path to the dataset for evaluation.  
- **id_class_dict_file** (`str`): Pickle file mapping class labels to IDs.  
- **output_directory** (`str`): Directory to write evaluation logs.  
- **tokenizer_name_or_path** (`str`, optional): Override tokenizer path.

**Returns**  
- **dict**: Evaluation metrics from `Trainer.evaluate()`.

---

#### plot_confusion_matrix

```python
plot_confusion_matrix(
    self,
    conf_mat: Any,
    output_directory: str,
    output_prefix: str,
    class_order: list[str]
) -> None
```
Delegate confusion matrix plotting to utility functions.

**Parameters**  
- **conf_mat** (`Any`): Confusion matrix array or dict.  
- **output_directory** (`str`): Directory to save the plot.  
- **output_prefix** (`str`): Filename prefix for the saved plot.  
- **class_order** (`list[str]`): Ordered list of class names for axes.

**Returns**  
- `None`

---

#### plot_predictions

```python
plot_predictions(
    self,
    predictions_file: str,
    id_class_dict_file: str,
    title: str,
    output_directory: str,
    output_prefix: str,
    class_order: list[str]
) -> None
```
Delegate prediction summary plotting to utility functions.

**Parameters**  
- **predictions_file** (`str`): Path to pickle file with prediction outputs.  
- **id_class_dict_file** (`str`): Pickle mapping class IDs to names.  
- **title** (`str`): Title for the plot.  
- **output_directory** (`str`): Directory to save the plot.  
- **output_prefix** (`str`): Filename prefix for the saved plot.  
- **class_order** (`list[str]`): Ordered class names for axes.

**Returns**  
- `None`

---
### build_custom_tokenizer
```python
build_custom_tokenizer(
    token_dict_path: str,
    pad_token: str = "<pad>",
    mask_token: str = "<mask>"
) -> PreTrainedTokenizerFast
```
Build a Hugging Face tokenizer from a pickled vocabulary.

**Description**  
Loads a token-to-ID mapping from a pickle file and instantiates a `PreTrainedTokenizerFast` with custom special tokens.

**Parameters**  
- **token_dict_path** (`str`): Path to a pickle file containing a `dict[str, int]` vocabulary.  
- **pad_token** (`str`, default `"<pad>"`): Token used for padding sequences.  
- **mask_token** (`str`, default `"<mask>"`): Token used to mark masked positions.

**Returns**  
- **PreTrainedTokenizerFast**: Tokenizer configured with `<unk>`, `<cls>`, `<sep>`, plus the provided pad/mask tokens.


## Classifier Utility Functions

### load_and_filter
```python
load_and_filter(
    filter_data: Optional[dict],
    nproc: int,
    input_data: Union[str, Dataset]
) -> Dataset
```

Load a Hugging Face dataset from disk or memory and apply metadata filters.

**Description**
If `input_data` is a string path, loads the dataset via `load_from_disk`; otherwise assumes it is already a `Dataset`. Applies filters for each key–value pair in `filter_data`.

**Parameters**

* **filter\_data** (`dict` or `None`): Mapping from column names to allowed values.
* **nproc** (`int`): Number of processes to use for filtering.
* **input\_data** (`str` or `Dataset`): Path to a saved dataset or an in-memory `Dataset` object.

**Returns**

* **Dataset**: Filtered Hugging Face `Dataset`.

---
### remove_rare
```python
remove_rare(
    data: Dataset,
    rare_threshold: float,
    state_key: str,
    nproc: int
) -> Dataset
```

Filter out examples whose label frequency falls below a threshold.

**Description**
Computes the frequency of each value in `state_key`. Drops values whose count divided by total examples is below `rare_threshold`.

**Parameters**

* **data** (`Dataset`): Input dataset.
* **rare\_threshold** (`float`): Minimum fraction for label retention (0–1).
* **state\_key** (`str`): Column name containing labels to evaluate.
* **nproc** (`int`): Number of processes for filtering.

**Returns**

* **Dataset**: Dataset with rare-label examples removed.

---
### downsample_and_shuffle
```python
downsample_and_shuffle(
    data: Dataset,
    max_ncells: Optional[int],
    max_ncells_per_class: Optional[int],
    cell_state_dict: dict
) -> Dataset
```

Shuffle and downsample a dataset globally and per class.

**Description**
Shuffles examples with a fixed seed, selects up to `max_ncells` total examples, then subsamples each class to at most `max_ncells_per_class` using indices from `cell_state_dict`.

**Parameters**

* **data** (`Dataset`): Input dataset to shuffle and downsample.
* **max\_ncells** (`int` or `None`): Maximum total examples to retain.
* **max\_ncells\_per\_class** (`int` or `None`): Maximum examples per class.
* **cell\_state\_dict** (`dict`): Must contain key `"state_key"` specifying label column name.

**Returns**

* **Dataset**: Downsampled and shuffled dataset.

---
### subsample_by_class
```python
subsample_by_class(
    labels: Sequence,
    N: int
) -> List[int]
```

Select up to N indices per unique label.

**Description**
Groups indices by label, then randomly samples up to N for each group.

**Parameters**

* **labels** (`Sequence`): List of labels corresponding to each example.
* **N** (`int`): Maximum number of indices to select per label.

**Returns**

* **List\[int]**: Selected example indices.

---
### rename_cols
```python
rename_cols(
    data: Dataset,
    state_key: str
) -> Dataset
```

Rename the label column to a standardized name.

**Description**
Renames the column named by `state_key` to `"label"`.

**Parameters**

* **data** (`Dataset`): Input dataset.
* **state\_key** (`str`): Original column name to rename.

**Returns**

* **Dataset**: Dataset with `state_key` renamed to `label`.

---
### flatten_list
```python
flatten_list(
    l: Sequence[Sequence]
) -> List[Any]
```

Flatten a list of lists into a single list.

**Parameters**

* **l** (`Sequence` of sequences): Nested list structure.

**Returns**

* **List\[Any]**: Flat list containing all items.

---
### label_classes
```python
label_classes(
    classifier: str,
    data: Dataset,
    class_dict: dict[str, Sequence],
    token_dict_path: str,
    nproc: int
) -> Tuple[Dataset, dict]
```

Map string labels to integer class IDs for cell- or gene-level classification.

**Description**

* For `classifier="cell"`, enumerates unique `data["label"]` values and remaps.
* For `classifier="gene"`, loads a token-to-gene mapping, inverts it, builds a token-to-class map from `class_dict`, labels sequences per token, filters out sequences with no valid labels.

**Parameters**

* **classifier** (`"cell"` or `"gene"`): Determines labeling strategy.
* **data** (`Dataset`): Input dataset with a `label` or `input_ids` field.
* **class\_dict** (`dict[str, Sequence]`): Mapping from class names to lists of gene identifiers (for gene classifier).
* **token\_dict\_path** (`str`): Path to pickle of token-to-gene vocabulary.
* **nproc** (`int`): Number of processes for dataset mapping/filtering.

**Returns**

* **Tuple\[Dataset, dict]**: Labeled dataset and mapping of class names to integer IDs.

---
### predict_from_checkpoint
```python
predict_from_checkpoint(
    model_dir: str,
    dataset_path: str,
    classifier_type: str = "sequence",
    batch_size: int = 32,
    num_workers: int = 4,
    return_logits: bool = False
) -> Union[List[int], Tuple[List[int], np.ndarray]]
```

Load a trained model and run predictions on a saved dataset.

**Description**
Loads tokenizer and model (sequence or token), constructs a `Trainer`, and returns predicted class indices, optionally with raw logits.

**Parameters**

* **model\_dir** (`str`): Path to directory containing a saved Hugging Face model and tokenizer.
* **dataset\_path** (`str`): Path to the saved HF dataset.
* **classifier\_type** (`str`): `'sequence'` or `'token'` to select model class.
* **batch\_size** (`int`): Number of examples per inference batch.
* **num\_workers** (`int`): DataLoader workers.
* **return\_logits** (`bool`): If True, also return raw logits array.

**Returns**

* **List\[int]** of predicted labels, or `(List[int], np.ndarray)` if `return_logits`.

---

## Data Collators

### class: DataCollatorForCellClassification
```python
class DataCollatorForCellClassification(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding: str = "max_length", max_length: int = 2048)
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]
```

Pad cell-level batches and extract labels.

**Description**
Inherits from `DataCollatorWithPadding`; pops each example’s `label`, pads inputs, and stacks labels into `batch["labels"]`.

**Parameters**

* **tokenizer**: Hugging Face tokenizer instance.
* **padding** (`str`): Padding strategy (`'max_length'` or `'longest'`).
* **max\_length** (`int`): Maximum sequence length for padding.

**Returns**

* **Dict\[str, torch.Tensor]**: Batch dict with `input_ids`, `attention_mask`, and `labels`.

---
### class DataCollatorForGeneClassification
```python
class DataCollatorForGeneClassification:
    def __init__(
        self,
        tokenizer,
        padding: str = "longest",
        max_length: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt"
    )
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]
```

Pad gene-level batches with per-token labels.

**Description**
Uses `tokenizer.pad()` to pad `input_ids` and `attention_mask`, then pads each feature’s `labels` list to `max_length` with `label_pad_token_id` and returns a tensor batch.

**Parameters**

* **tokenizer**: Hugging Face tokenizer instance.
* **padding** (`str`): Padding strategy (`'max_length'` or `'longest'`).
* **max\_length** (`int` or `None`): Maximum sequence length.
* **label\_pad\_token\_id** (`int`): ID to use for padded label positions (`-100`).
* **return\_tensors** (`str`): Tensor type (`'pt'` for PyTorch).

**Returns**

* **Dict\[str, torch.Tensor]**: Batch dict with padded `input_ids`, `attention_mask`, and `labels`.

