# stFormer Pretraining API

## Overview

This API provides tools for pretraining stFormer models on spatial transcriptomics data. It includes custom data collators, a specialized trainer class, and utility functions for environment setup, configuration, and hyperparameter tuning.

---

## Utility Functions

### `load_example_lengths`

```python
load_example_lengths(file_path: Union[str, Path]) -> List[int]
```

Load example lengths for length-based sampling from a pickle file.

**Parameters:**

file_path : Path | str
Path to the pickle file containing example lengths.

Returns:

List[int] — A list of example lengths.

setup_environment

```python
setup_environment(seed: int) -> None
```

Configure environment variables and random seeds for reproducible training.

**Parameters:**

seed : int
The random seed to use.

Returns:

None

make_output_dirs

```python
make_output_dirs(output_dir: Path, run_name: str) -> Dict[str, Path]
```

Create and return paths for training, logging, and model outputs.

**Parameters:**

output_dir : Path
The root output directory.

run_name : str
A unique name for the current training run.

Returns:

Dict[str, Path] — A dictionary containing paths for 'training', 'logging', and 'model' outputs.

choose_closest

```python
choose_closest(name: str, supported: Sequence[str], unsupported: Optional[Sequence[str]] = None, cutoff: float = 0.4) -> str
```

Finds the closest match for a given name within a sequence of supported strings, optionally checking against unsupported strings.

**Parameters:**

name : str
The name to find a match for.

supported : Sequence[str]
A sequence of supported strings to match against.

unsupported : Sequence[str] | None, default: None
An optional sequence of unsupported strings. If name is found here (case-insensitive), a ValueError is raised.

cutoff : float, default: 0.4
A similarity threshold for fuzzy matching.

Returns:

str — The closest supported match.

Raises:

ValueError — If name is in unsupported or no close match is found.

check_deepspeed_optimizer

```python
check_deepspeed_optimizer(optimizer_type: str) -> str
```

Checks if the given optimizer type is supported by DeepSpeed and returns the canonical name.

**Parameters:**

optimizer_type : str
The type of optimizer (e.g., 'Adam', 'AdamW').

Returns:

str — The validated optimizer type.

Raises:

ValueError — If the optimizer type is not supported or a close match cannot be found.

check_deepspeed_scheduler

```python
check_deepspeed_scheduler(lr_scheduler_type: str) -> str
```
Checks if the given learning rate scheduler type is supported by DeepSpeed and returns the canonical name.

**Parameters:**

lr_scheduler_type : str
The type of learning rate scheduler (e.g., 'WarmupLR').

Returns:

str — The validated scheduler type.

Raises:

ValueError — If the scheduler type is not supported or a close match cannot be found.

build_bert_config
```python
build_bert_config(
    model_type: str,
    num_layers: int,
    num_heads: int,
    embed_dim: int,
    max_input: int,
    pad_id: int,
    vocab_size: int,
    activ_fn: str = 'relu',
    initializer_range: float = 0.02,
    layer_norm_eps: float = 1e-12,
    attention_dropout: float = 0.02,
    hidden_dropout: float = 0.02,
) -> BertConfig
```
Constructs a BertConfig object for the stFormer model.

**Parameters:**

model_type : str
The type of BERT model.

num_layers : int
Number of hidden layers in the transformer.

num_heads : int
Number of attention heads.

embed_dim : int
Dimension of the embedding and hidden layers.

max_input : int
Maximum sequence length.

pad_id : int
The ID of the padding token.

vocab_size : int
Size of the vocabulary.

activ_fn : str, default: 'relu'
Activation function for the hidden layers.

initializer_range : float, default: 0.02
Standard deviation of the truncated normal initializer.

layer_norm_eps : float, default: 1e-12
Epsilon for layer normalization.

attention_dropout : float, default: 0.02
Dropout probability for attention.

hidden_dropout : float, default: 0.02
Dropout probability for hidden layers.

Returns:

BertConfig — A HuggingFace BertConfig object.

get_training_arguments

```python
get_training_arguments(
    output_dir: Path,
    logging_dir: Path,
    train_dataset_length: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    warmup_steps: int,
    lr_scheduler_type: str,
    optimizer_type: str,
    do_train: bool,
    do_eval: bool,
    length_column_name: str,
    disable_tqdm: bool,
    fp16: bool,
    group_by_length: bool,
    overrides: Optional[Dict] = None,
) -> TrainingArguments
```

Builds a TrainingArguments object for standard pretraining, merging defaults with any provided overrides.

**Parameters:**

output_dir : Path
Directory for model checkpoints.

logging_dir : Path
Directory for training logs.

train_dataset_length : int
Number of examples in the training dataset.

batch_size : int
Per-device training batch size.

learning_rate : float
Initial learning rate.

epochs : int
Number of training epochs.

weight_decay : float
Strength of weight decay.

warmup_steps : int
Number of warmup steps for the learning rate scheduler.

lr_scheduler_type : str
Type of learning rate scheduler.

optimizer_type : str
Type of optimizer.

do_train : bool
Whether to run training.

do_eval : bool
Whether to run evaluation.

length_column_name : str
Name of the column containing sequence lengths in the dataset.

disable_tqdm : bool
Whether to disable the tqdm progress bar.

fp16 : bool
Whether to use mixed precision training (FP16).

group_by_length : bool
Whether to group samples by length for more efficient batching.

overrides : Dict | None, default: None
Optional dictionary of arguments to override defaults.

Returns:

TrainingArguments — A HuggingFace TrainingArguments object.

build_deepspeed

```python
build_deepspeed(
    output_dir: Path,
    logging_dir: Path,
    train_dataset_length: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    warmup_steps: int,
    lr_scheduler_type: str,
    optimizer_type: str,
    do_train: bool,
    do_eval: bool,
    length_column_name: str,
    disable_tqdm: bool,
    fp16: bool,
    group_by_length: bool,
    overrides: Optional[Dict] = None,
) -> TrainingArguments
```

Builds a TrainingArguments object specifically configured for DeepSpeed, including DeepSpeed-specific parameters.

**Parameters:**

output_dir : Path
Directory for model checkpoints.

logging_dir : Path
Directory for training logs.

train_dataset_length : int
Number of examples in the training dataset.

batch_size : int
Per-device training batch size.

learning_rate : float
Initial learning rate.

epochs : int
Number of training epochs.

weight_decay : float
Strength of weight decay.

warmup_steps : int
Number of warmup steps for the learning rate scheduler.

lr_scheduler_type : str
Type of learning rate scheduler (validated against DeepSpeed supported types).

optimizer_type : str
Type of optimizer (validated against DeepSpeed supported types).

do_train : bool
Whether to run training.

do_eval : bool
Whether to run evaluation.

length_column_name : str
Name of the column containing sequence lengths in the dataset.

disable_tqdm : bool
Whether to disable the tqdm progress bar.

fp16 : bool
Whether to use mixed precision training (FP16).

group_by_length : bool
Whether to group samples by length for more efficient batching.

overrides : Dict | None, default: None
Optional dictionary of arguments to override defaults.

Returns:

TrainingArguments — A HuggingFace TrainingArguments object with DeepSpeed configuration.

STFormerPreCollator
Module: pretrainer_refactored

Data collator for single-cell transcriptomics, providing padding and token conversion.

__init__

```python
STFormerPreCollator(token_dict: Dict[str, int])
```

**Parameters:**

token_dict : Dict[str, int]
A dictionary mapping tokens (e.g., gene names, special tokens) to their integer IDs.

convert_tokens_to_ids

```python
convert_tokens_to_ids(tokens: Union[str, List[str]]) -> Union[int, List[int]]
```
Convert token(s) to their numerical IDs.

**Parameters:**

tokens : str | List[str]
A single token string or a list of token strings.

Returns:

int | List[int] — The numerical ID(s) corresponding to the input token(s).

convert_ids_to_tokens

```python
convert_ids_to_tokens(ids: Union[int, List[int]]) -> Union[str, List[str]]
```
Inverse of convert_tokens_to_ids, required by DataCollatorForLanguageModeling.

**Parameters:**

ids : int | List[int]
A single integer ID or a list of integer IDs.

Returns:

str | List[str] — The token string(s) corresponding to the input ID(s).

__call__

```python
__call__(examples: List[Dict[str, List[int]]]) -> BatchEncoding
```
Prepare a batch for masked language modeling.

**Parameters:**

examples : List[Dict[str, List[int]]]
A list of examples, each a dictionary containing 'input_ids'.

Returns:

BatchEncoding — A batch prepared for masked language modeling.

pad
```python
pad(
    encoded_inputs: Union[List[Dict[str, List[int]]], BatchEncoding],
    padding: Union[bool, str] = True,
    max_length: Optional[int] = None,
    return_tensors: Optional[str] = None,
    **kwargs
) -> BatchEncoding
```
Pads input sequences to a uniform length and generates an attention mask.

**Parameters:**

encoded_inputs : List[Dict[str, List[int]]] | BatchEncoding
Input sequences, either as a list of dictionaries with 'input_ids' or a BatchEncoding object.

padding : bool | str, default: True
Whether to pad the inputs. Can be True, 'longest', or 'max_length'.

max_length : int | None, default: None
The maximum length to pad to. If None, pads to the longest sequence in the batch.

return_tensors : str | None, default: None
The type of tensors to return ('pt' for PyTorch).

kwargs
Additional keyword arguments.

Returns:

BatchEncoding — A BatchEncoding object containing padded input_ids and attention_mask.

get_special_tokens_mask

```python
get_special_tokens_mask(
    token_ids_0: List[int],
    token_ids_1: Optional[List[int]] = None,
    already_has_special_tokens: bool = False
) -> List[int]
```
Marks pad and mask tokens as "special" so they won’t get masked again during masked language modeling.

**Parameters:**

token_ids_0 : List[int]
A list of token IDs for the first sequence.

token_ids_1 : List[int] | None, default: None
An optional list of token IDs for the second sequence.

already_has_special_tokens : bool, default: False
Whether the input already contains special tokens.

Returns:

List[int] — A list of integers (0 or 1) indicating whether a token is special.

STFormerPretrainer
Module: pretrainer_refactored

Custom Trainer for masked pretraining on single-cell data. This class extends transformers.Trainer.

__init__

```python
STFormerPretrainer(
    args: TrainingArguments,
    train_dataset: Dataset,
    token_dictionary: Dict[str, int],
    example_lengths_file: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    model_init: Optional[Callable[[], torch.nn.Module]] = None,
    mlm_probability: float = 0.15,
)
```
**Parameters:**

args : TrainingArguments
The training arguments.

train_dataset : Dataset
The dataset for training.

token_dictionary : Dict[str, int]
A dictionary mapping tokens to their integer IDs.

example_lengths_file : Path | str
Path to a pickle file containing example lengths for length grouping.

model : torch.nn.Module | None, default: None
The model to train.

model_init : Callable[[], torch.nn.Module] | None, default: None
A function that returns an initialized model. Used for hyperparameter search.

mlm_probability : float, default: 0.15
The probability of masking tokens for Masked Language Model training.

Raises:

ValueError — If neither model nor model_init is provided.

get_train_sampler

```python
get_train_sampler() -> Optional[torch.utils.data.Sampler]
```
Returns a sampler for the training dataset, grouping by length if args.group_by_length is True.

Returns:

torch.utils.data.Sampler | None — A sampler object or None if the dataset has no length.

PretrainML
Module: stformer_pretrainer

Module to configure and run stFormer pretraining as importable functions.

__init__

```python
PretrainML(
    dataset_path: str,
    token_dict_path: str,
    example_lengths_path: str,
    mode: Literal['spot', 'neighborhood'],
    output_dir: str,
    deepspeed: bool = False,
    seed: int = 42,
    model_type: str = 'bert',
    num_layers: int = 6,
    num_heads: int = 4,
    embed_dim: int = 256,
    max_input: int = 2048,
    activ_fn: str = 'relu',
    initializer_range: float = 0.02,
    layer_norm_eps: float = 1e-12,
    attention_dropout: float = 0.02,
    hidden_dropout: float = 0.02,
    batch_size: int = 12,
    learning_rate: float = 1e-3,
    lr_scheduler_type: str = 'linear',
    optimizer_type: str = 'adamw_torch',
    warmup_steps: int = 10000,
    epochs: int = 3,
    weight_decay: float = 0.001,
    do_train: bool = True,
    do_eval: bool = False,
    length_column_name: str = 'length',
    disable_tqdm: bool = False,
    group_by_length: bool = False,
    fp16: bool = True,
    training_args_overrides: Optional[Dict] = None,
)
Initializes the pretraining setup for a Bert Masked Language Model.
```
**Parameters:**

dataset_path : str
Path to the HuggingFace dataset.

token_dict_path : str
Path to the token dictionary pickle file.

example_lengths_path : str
Path to the example lengths pickle file.

mode : 'spot' | 'neighborhood'
Tokenization mode. If 'neighborhood', max_input is doubled.

output_dir : str
Root directory for all outputs (models, logs).

deepspeed : bool, default: False
Whether to enable DeepSpeed for distributed training.

seed : int, default: 42
Random seed for reproducibility.

model_type : str, default: 'bert'
Type of the HuggingFace model.

num_layers : int, default: 6
Number of transformer layers.

num_heads : int, default: 4
Number of attention heads.

embed_dim : int, default: 256
Dimension of the model embeddings.

max_input : int, default: 2048
Maximum input sequence length.

activ_fn : str, default: 'relu'
Activation function for the model.

initializer_range : float, default: 0.02
Standard deviation of the weight initializer.

layer_norm_eps : float, default: 1e-12
Epsilon for layer normalization.

attention_dropout : float, default: 0.02
Dropout rate for attention layers.

hidden_dropout : float, default: 0.02
Dropout rate for hidden layers.

batch_size : int, default: 12
Per-device training batch size.

learning_rate : float, default: 1e-3
Initial learning rate.

lr_scheduler_type : str, default: 'linear'
Type of learning rate scheduler.

optimizer_type : str, default: 'adamw_torch'
Type of optimizer.

warmup_steps : int, default: 10000
Number of warmup steps.

epochs : int, default: 3
Number of training epochs.

weight_decay : float, default: 0.001
Weight decay rate.

do_train : bool, default: True
Whether to perform training.

do_eval : bool, default: False
Whether to perform evaluation.

length_column_name : str, default: 'length'
Column name in the dataset indicating sequence length.

disable_tqdm : bool, default: False
Whether to disable the progress bar.

group_by_length : bool, default: False
Whether to group training samples by length.

fp16 : bool, default: True
Whether to enable mixed precision training.

training_args_overrides : Dict | None, default: None
Optional dictionary to override TrainingArguments parameters.

run_pretraining
Python

run_pretraining()
Executes the stFormer pretraining process. This function handles setting up directories, building the model configuration, preparing training arguments (with or without DeepSpeed), initializing the trainer, and saving the final model and tokenizer.

Returns:

None

run_hyperparameter_train

```python
run_hyperparameter_train(
    search_space: Dict,
    resources_per_trial,
    n_trials: int = 8,
    backend: Literal['ray','optuma'] = 'ray'
)
```
Runs a hyperparameter search for stFormer pretraining.

**Parameters:**

search_space : Dict
A dictionary defining the hyperparameter search space. Each key is a hyperparameter name, and its value is a dictionary specifying 'type' (e.g., 'loguniform', 'categorical', 'int', 'uniform') and range/values.

resources_per_trial
Resources to allocate per trial (e.g., {"cpu": 4, "gpu": 1}).

n_trials : int, default: 8
Number of hyperparameter trials to run.

backend : 'ray' | 'optuma', default: 'ray'
The backend to use for hyperparameter optimization.

Returns:

None (prints the best trial results to console)