# Classification

This documentation describes the current classification API used with STFormer-style
models. It covers:

- The high-level `Classifier` orchestration class (`Classifier.py`)
- Data utilities and collators (`classifier_utils.py`)
- Evaluation helpers (`evaluation_utils.py`)

The goal is to give a single place to understand how to prepare datasets, train
sequence- or gene-level classifiers, and evaluate their performance.

---

## Classifier

```python
from Classifier import Classifier
```

```python
class Classifier:
    def __init__(
        self,
        metadata_column,
        mode: str = "spot",
        classifier_type: str = "sequence",
        gene_class_dict=None,
        label_mapping=None,
        token_dictionary_file=None,
        filter_data=None,
        rare_threshold: float = 0.0,
        max_examples=None,
        max_examples_per_class=None,
        training_args=None,
        ray_config=None,
        freeze_layers: int = 0,
        forward_batch_size: int = 32,
        nproc: int = 4,
    )
```

A flexible orchestration class that:

1. Loads and filters a Hugging Face dataset.
2. Builds label columns for either **sequence-level** or **gene-level** classification.
3. Configures and launches Hugging Face `Trainer` (or Ray Tune for hyperparameter search).
4. Saves labeled datasets, trained models, and predictions.

### Parameters (init)

- **metadata_column** (`str`):  
  Name of the column in the dataset that contains the original labels
  (e.g. `"cell_type"`, `"treatment"`).

- **mode** (`str`, default `"spot"`):  
  Tokenization mode, relevant for gene-level classification:
  - `"spot"` – sequences are treated as single spots.
  - `"extended"` – sequences are spot + neighbor concatenations. For gene-level
    classification, labels in the neighbor half are masked out with `-100` so that
    only the first half is supervised.

- **classifier_type** (`str`, default `"sequence"`):  
  Type of classifier to train:
  - `"sequence"` – one label per example (cell/spot classification).
  - `"gene"` – per-token labels (gene-level classification, implemented as a
    token classification head).

- **gene_class_dict** (`dict | None`):  
  Required when `classifier_type="gene"`. A mapping from class name to a **list of
  gene IDs or tokens** belonging to that class. Used by `label_classes` to build
  token-level labels.

- **label_mapping** (`dict | None`):  
  Optional mapping from **label value → class ID**. If `None`, the mapping is
  inferred automatically from the data during `prepare_data` and stored on
  `self.label_mapping`.

- **token_dictionary_file** (`str | None`):  
  Path to a pickle file containing a token vocabulary dictionary. Required for
  gene-level classification so gene identifiers in `gene_class_dict` can be
  mapped to token IDs.

- **filter_data** (`dict | None`):  
  Optional metadata filters applied before labeling. The dict should map
  `column_name → allowed_values`.

- **rare_threshold** (`float`, default `0.0`):  
  Minimum fraction for a label to be kept. Any label whose frequency is below
  `rare_threshold` is dropped via `remove_rare`.

- **max_examples** (`int | None`):  
  Global cap on the number of examples after filtering. If `None`, all remaining
  examples are kept.

- **max_examples_per_class** (`int | None`):  
  Per-class cap when downsampling. If provided, at most this many examples per
  class are retained after shuffling.

- **training_args** (`dict | None`):  
  Keyword arguments merged into `Classifier.default_training_args` to construct
  Hugging Face `TrainingArguments`.

- **ray_config** (`dict | None`):  
  Hyperparameter search space for Ray Tune. If set and `n_trials > 0` is passed
  to `train`, the training will be delegated to `_ray_tune`.

- **freeze_layers** (`int`, default `0`):  
  Number of initial transformer encoder layers to freeze.

- **forward_batch_size** (`int`, default `32`):  
  Batch size used during evaluation and for scaling the effective training batch
  size.

- **nproc** (`int`, default `4`):  
  Number of worker processes used in dataset filtering and mapping operations.

---

### build_custom_tokenizer

```python
from Classifier import build_custom_tokenizer

tokenizer = build_custom_tokenizer(
    token_dict_path: str,
    pad_token: str = "<pad>",
    mask_token: str = "<mask>",
)
```

Build a `PreTrainedTokenizerFast` from a custom token vocabulary stored as a
pickle.

#### Parameters

- **token_dict_path** (`str`):  
  Path to a pickle file containing a `dict[str, int]` vocabulary.

- **pad_token** (`str`, default `"<pad>"`):  
  Token used for padding sequences.

- **mask_token** (`str`, default `"<mask>"`):  
  Token used to indicate masked positions.

#### Returns

- **PreTrainedTokenizerFast** – ready-to-use tokenizer instance.

---

### _load_tokenizer

```python
tokenizer = classifier._load_tokenizer(
    name_or_path: str,
)
```

Internal helper to load a tokenizer from:

- A Hugging Face model name (e.g. `"bert-base-uncased"`),
- A local tokenizer directory,
- Or a pickled token dictionary (delegating to `build_custom_tokenizer`).

You usually do **not** call this directly; it is used inside `train` and
`evaluate`.

---

### prepare_data

```python
dataset_path, label_map_path = classifier.prepare_data(
    input_data,
    output_directory: str,
    output_prefix: str,
)
```

Filter, downsample, and label a dataset for classification, then save it to disk.

The method:

1. Loads the dataset via `classifier_utils.load_and_filter`.
2. Drops rare labels using `classifier_utils.remove_rare`.
3. Optionally down-samples using `downsample_and_shuffle`.
4. Builds label columns:
   - `"sequence"` classifier: one `label` column.
   - `"gene"` classifier: per-token `labels` sequences via `label_classes`. In
     `"extended"` mode, only the first half is supervised (second half set to
     `-100`).
5. Saves the dataset (`DatasetDict`) with `save_to_disk`.
6. Saves the label mapping to a pickle.

#### Parameters

- **input_data** (`str | Dataset | DatasetDict`):  
  Either a path to a dataset saved with `load_from_disk` or an in-memory
  dataset.

- **output_directory** (`str`):  
  Directory to write the labeled dataset and label map.

- **output_prefix** (`str`):  
  Prefix used for the saved dataset and mapping filenames.

#### Returns

- **dataset_path** (`str`):  
  Path to the labeled dataset directory.

- **label_map_path** (`str`):  
  Path to the pickled `id_class_dict` mapping.

---

### train

```python
trainer_or_result = classifier.train(
    model_checkpoint: str,
    dataset_path: str,
    output_directory: str,
    eval_dataset=None,
    n_trials: int = 0,
    test_size: float = 0.2,
    tokenizer_name_or_path: str | None = None,
)
```

Train a classifier (sequence or gene-level) or run Ray Tune hyperparameter
search, depending on `ray_config` and `n_trials`.

#### Behavior

- If `ray_config` is set **and** `n_trials > 0`:
  - Delegates to `_ray_tune`, which runs Ray Tune over the provided hyperparameter
    space and returns the tuning result.

- Otherwise:
  1. Loads the labeled dataset from `dataset_path`.
  2. If `eval_dataset` is `None`, creates a train/eval split using `test_size`.
  3. Loads a tokenizer (model checkpoint or `tokenizer_name_or_path`).
  4. Chooses collator:
     - `DataCollatorForCellClassification` when `classifier_type="sequence"`.
     - `DataCollatorForGeneClassification` when `classifier_type="gene"`.
  5. Builds a classification model using `AutoModelForSequenceClassification` or
     `AutoModelForTokenClassification`.
  6. Instantiates a `Trainer` with:
     - training args (`TrainingArguments`),
     - `compute_metrics` from `evaluation_utils` as `compute_metrics`,
     - the chosen collator and datasets.
  7. Calls `trainer.train()`.
  8. Saves:
     - final model and tokenizer in `{output_directory}/final_model`,
     - predictions on eval set to `{output_directory}/predictions.pkl`.

#### Parameters

- **model_checkpoint** (`str`):  
  Hugging Face model name or local path to the backbone to fine-tune.

- **dataset_path** (`str`):  
  Path to the labeled dataset created by `prepare_data`.

- **output_directory** (`str`):  
  Directory where checkpoints, final models, and prediction files are stored.

- **eval_dataset** (`str | Dataset | DatasetDict | None`, default `None`):  
  Optional separate evaluation dataset. If `None`, the dataset at `dataset_path`
  is split into train/eval.

- **n_trials** (`int`, default `0`):  
  Number of Ray Tune trials to run. If `> 0` and `ray_config` is set, triggers
  hyperparameter search.

- **test_size** (`float`, default `0.2`):  
  Fraction of data reserved for evaluation when `eval_dataset` is not provided.

- **tokenizer_name_or_path** (`str | None`):  
  Optional explicit tokenizer source. If `None`, `model_checkpoint` is used.

#### Returns

- **Trainer** – trained Hugging Face `Trainer` instance, or  
- Ray Tune result object when `_ray_tune` is used.

---

### evaluate

```python
metrics = classifier.evaluate(
    model_directory: str,
    eval_dataset_path: str,
    id_class_dict_file: str,
    output_directory: str,
    tokenizer_name_or_path: str | None = None,
)
```

Evaluate a trained model on a labeled dataset.

#### Behavior

1. Loads model + tokenizer from `model_directory`.
2. Loads eval dataset from `eval_dataset_path`.
3. Loads the class mapping from `id_class_dict_file`.
4. Recreates the correct collator (sequence vs gene).
5. Instantiates a `Trainer` with `compute_metrics`.
6. Calls `trainer.evaluate()` and returns the metrics.

#### Parameters

- **model_directory** (`str`):  
  Directory containing the trained model and tokenizer (e.g. `final_model`).

- **eval_dataset_path** (`str`):  
  Path to the evaluation dataset (same format as produced by `prepare_data`).

- **id_class_dict_file** (`str`):  
  Pickled mapping between class names and class IDs.

- **output_directory** (`str`):  
  Directory used for any logs or plots (if produced).

- **tokenizer_name_or_path** (`str | None`):  
  Optional tokenizer override; if `None`, the tokenizer in `model_directory` is
  used.

#### Returns

- **dict** – metrics dictionary as returned by `Trainer.evaluate()` plus the
  extra statistics computed by `compute_metrics`.

---

### plot_confusion_matrix

```python
classifier.plot_confusion_matrix(
    conf_mat,
    output_directory: str,
    output_prefix: str,
    class_order: list[str],
)
```

Thin wrapper around `evaluation_utils.plot_confusion_matrix` that saves a
heatmap of the confusion matrix.

#### Parameters

- **conf_mat** (`np.ndarray` or array-like):  
  Confusion matrix of shape `(num_classes, num_classes)`.

- **output_directory** (`str`):  
  Directory where the figure is saved.

- **output_prefix** (`str`):  
  Prefix for the output filename.

- **class_order** (`list[str]`):  
  Ordered list of class names for axes labeling.

---

### plot_predictions

```python
classifier.plot_predictions(
    predictions_file: str,
    id_class_dict_file: str,
    title: str,
    output_directory: str,
    output_prefix: str,
    class_order: list[str],
)
```

Wrapper around `evaluation_utils.plot_predictions` to visualize prediction
outputs (saved from `Trainer.predict`).

#### Parameters

- **predictions_file** (`str`):  
  Path to the pickled predictions dict.

- **id_class_dict_file** (`str`):  
  Pickled mapping between class IDs and names.

- **title** (`str`):  
  Title for the plot.

- **output_directory** (`str`):  
  Directory where the prediction plot is saved.

- **output_prefix** (`str`):  
  Prefix for the output filename.

- **class_order** (`list[str]`):  
  Ordered list of class names for x/y axes.

---

## Dataset Utilities (`classifier_utils.py`)

The `classifier_utils.py` helper module provides a set of functions and collators
used by `Classifier`.

### load_and_filter

```python
from classifier_utils import load_and_filter

data = load_and_filter(
    filter_data,
    nproc: int,
    input_data_file,
)
```

Load a dataset (from disk or in-memory) and apply metadata filters.

- **filter_data** (`dict | None`):  
  Mapping from column name to allowed values.

- **nproc** (`int`):  
  Number of worker processes used for filtering.

- **input_data_file** (`str | Dataset | DatasetDict`):  
  Path to a dataset saved with `load_from_disk` or a `Dataset`/`DatasetDict`.

Returns a filtered `Dataset` or `DatasetDict`.

---

### remove_rare

```python
from classifier_utils import remove_rare

data = remove_rare(
    data,
    rare_threshold: float,
    state_key: str,
    nproc: int,
)
```

Drop examples whose label frequency is below a threshold.

- **data** (`Dataset | DatasetDict`): Input dataset.
- **rare_threshold** (`float`): Minimum frequency (0–1) for label retention.
- **state_key** (`str`): Column containing label values.
- **nproc** (`int`): Number of worker processes.

---

### downsample_and_shuffle

```python
from classifier_utils import downsample_and_shuffle

data = downsample_and_shuffle(
    data,
    max_ncells: int | None,
    max_ncells_per_class: int | None,
    cell_state_dict: dict,
)
```

Shuffle and optionally downsample the dataset, globally and per class.

- **data** (`Dataset | DatasetDict`): Dataset to downsample.
- **max_ncells** (`int | None`): Global maximum number of examples.
- **max_ncells_per_class** (`int | None`): Max per-class examples after shuffling.
- **cell_state_dict** (`dict`): Mapping from class ID to list of example indices.

---

### subsample_by_class

```python
from classifier_utils import subsample_by_class

data = subsample_by_class(
    data,
    cell_state_dict: dict,
    max_ncells_per_class: int,
)
```

Subsample examples per class according to `max_ncells_per_class`.

---

### rename_cols

```python
from classifier_utils import rename_cols

data = rename_cols(
    data,
    mapping: dict[str, str],
)
```

Rename dataset columns according to a mapping.

---

### flatten_list

```python
from classifier_utils import flatten_list

flat = flatten_list(nested_list)
```

Flatten a nested list of lists into a single list.

---

### label_classes

```python
from classifier_utils import label_classes

labeled_data, id_class_dict = label_classes(
    data,
    gene_class_dict: dict,
    label_mapping: dict | None,
    token_dictionary_file: str | None,
    mode: str = "spot",
)
```

Create per-token class labels for gene-level classification.

- **data** (`Dataset | DatasetDict`):  
  Input tokenized dataset.

- **gene_class_dict** (`dict`):  
  Mapping from class name → list of gene IDs/tokens.

- **label_mapping** (`dict | None`):  
  Optional mapping from class name → integer ID. If `None`, it is created.

- **token_dictionary_file** (`str | None`):  
  Path to token dictionary used to map gene IDs to token IDs.

- **mode** (`str`, default `"spot"`):  
  `"spot"` for single sequences; `"extended"` for spot + neighbor, where only
  the first half is supervised.

Returns the labeled dataset and the `id_class_dict` mapping.

---

### DataCollatorForCellClassification

```python
from classifier_utils import DataCollatorForCellClassification
```

A custom data collator for **sequence-level** classification. It:

- Pads input IDs and attention masks,
- Collects `labels` as integer class IDs,
- Returns a batch suitable for `AutoModelForSequenceClassification`.

You normally do not instantiate this directly; it is constructed inside
`Classifier.train`.

---

### DataCollatorForGeneClassification

```python
from classifier_utils import DataCollatorForGeneClassification
```

A custom data collator for **gene-level** (token) classification. It:

- Pads `input_ids` and `labels`,
- Preserves `-100` values to mark ignored positions,
- Returns a batch suitable for `AutoModelForTokenClassification`.

---

## Evaluation Utilities (`evaluation_utils.py`)

### py_softmax

```python
from evaluation_utils import py_softmax

probs = py_softmax(x, axis=-1)
```

Pure-Python softmax utility over a NumPy array.

---

### compute_metrics

```python
from evaluation_utils import compute_metrics

metrics = compute_metrics(eval_pred)
```

Metric function intended for use with `Trainer`. It typically computes:

- overall accuracy,
- macro/micro F1,
- precision and recall,
- confusion matrix and per-class stats (if labels are available).

It accepts the standard `(predictions, labels)` tuple used by `Trainer`.

---

### evaluate_model

```python
from evaluation_utils import evaluate_model

metrics = evaluate_model(
    model,
    dataloader,
    device,
)
```

Standalone evaluation helper that runs a model over a dataloader and aggregates
metrics using `compute_metrics`.

---

### plot_confusion_matrix

```python
from evaluation_utils import plot_confusion_matrix

plot_confusion_matrix(
    conf_mat,
    class_order: list[str],
    output_directory: str,
    output_prefix: str,
)
```

Plot and save a confusion matrix heatmap.

---

### plot_predictions

```python
from evaluation_utils import plot_predictions

plot_predictions(
    predictions_file: str,
    id_class_dict_file: str,
    title: str,
    output_directory: str,
    output_prefix: str,
    class_order: list[str],
)
```

Load saved predictions and class mappings, reconstruct the confusion matrix, and
save a labeled heatmap to disk.
