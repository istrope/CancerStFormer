# Pretraining

## Pretrainer

```python
class STFormerPretrainer(Trainer):
    def __init__(
        self,
        args: TrainingArguments,
        train_dataset: Dataset,
        token_dictionary: Dict[str,int],
        example_lengths_file: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        model_init: Optional[Callable[[], torch.nn.Module]] = None,
        mlm_probability: float = 0.15
    )
```
Custom Hugging Face Trainer subclass for masked pretraining on single-cell data.

**Parameters**
- **args** (`TrainingArguments`): HF training configuration.
- **train_dataset** (`Dataset`): Dataset containing `input_ids` fields.
- **token_dictionary** (`Dict[str,int]`): Vocabulary mapping.
- **example_lengths_file** (`str` | `Path`): Pickle path for example lengths for length grouping.
- **model** (`nn.Module`, optional): Model instance to train.
- **model_init** (`Callable`, optional): Function to create model for hyperparameter search.
- **mlm_probability** (`float`): Masking probability for MLM.

**Raises**
- **ValueError**: If neither `model` nor `model_init` is provided.

---
### get_train_sampler
```python
def get_train_sampler(self) -> Optional[Sampler]
```
Provide a sampler that groups examples by length if enabled.

**Returns**
- **LengthGroupedSampler** if `args.group_by_length` is `True` and dataset has length info, else `RandomSampler` or `None`.

---
### STFormerPreCollator
```python
class STFormerPreCollator(SpecialTokensMixin):
    def __init__(
        self,
        token_dict: Dict[str, int]
    )
```
Data collator for masked language modeling over single-cell tokens.

**Parameters**
- **token_dict** (`Dict[str,int]`): Mapping from token strings to integer IDs (must include `<pad>` and `<mask>`).

---
### load_example_lengths
```python
def load_example_lengths(
    file_path: Union[str, Path]
) -> List[int]
```
Load a list of example lengths from a pickle file.

**Parameters**
- **file_path** (`str` | `Path`): Path to the pickle containing a Python list of integers.

**Returns**
- **List[int]**: Loaded example lengths for length-based sampling.

---

### __len__
```python
def __len__(self) -> int
```
Get vocabulary size.

**Returns**
- **int**: Number of tokens in `token_dict`.

---
### convert_token_to_ids
```python
def convert_tokens_to_ids(
    self,
    tokens: Union[str, List[str]]
) -> Union[int, List[int]]
```
Map token strings to their numeric IDs.

**Parameters**
- **tokens** (`str` or `List[str]`): Single token or list of tokens.

**Returns**
- **int** or **List[int]**: Corresponding ID(s), or `None` if missing.

---
### convert_ids_to_tokens
```python
def convert_ids_to_tokens(
    self,
    ids: Union[int, List[int]]
) -> Union[str, List[str]]
```
Inverse mapping from IDs to token strings.

**Parameters**
- **ids** (`int` or `List[int]`): Single token ID or list of IDs.

**Returns**
- **str** or **List[str]**: Corresponding token(s), or `None` if missing.

---
### __call__
```python
def __call__(
    self,
    examples: List[Dict[str, List[int]]]
) -> BatchEncoding
```
Create a batch dict for Hugging Face Trainer input.

**Parameters**
- **examples** (`List[Dict[str,List[int]]]`): Each with key `input_ids` holding a list of token IDs.

**Returns**
- **BatchEncoding**: Contains a batch `input_ids` list (no masking applied here; handled by HF MLM collator).

---
### pad
```python
def pad(
    self,
    encoded_inputs: Union[List[Dict[str, List[int]]], BatchEncoding],
    padding: Union[bool, str] = True,
    max_length: Optional[int] = None,
    return_tensors: Optional[str] = None,
    **kwargs
) -> BatchEncoding
```
Pad or truncate batches of token sequences under a single maximum length.

**Parameters**
- **encoded_inputs** (`List[dict]` or `BatchEncoding`): Raw batch or already-encoded batch with `input_ids`.
- **padding** (`bool` or `str`): Whether to pad and how.
- **max_length** (`int`, optional): Target sequence length; if `None`, uses the longest in batch.
- **return_tensors** (`str`, optional): Framework for returned tensors, e.g. `'pt'`.

**Returns**
- **BatchEncoding**: Contains padded `input_ids` and corresponding `attention_mask`.

---
### get_special_tokens_mask
```python
def get_special_tokens_mask(
    self,
    token_ids_0: List[int],
    token_ids_1: Optional[List[int]] = None,
    already_has_special_tokens: bool = False
) -> List[int]
```
Flag special tokens (pad/mask) to avoid masking them in MLM.

**Parameters**
- **token_ids_0** (`List[int]`): First sequence of token IDs.
- **token_ids_1** (`List[int]`, optional): Second sequence (if any).
- **already_has_special_tokens** (`bool`): If `True`, skip marking.

**Returns**
- **List[int]**: Mask where `1` denotes a special token position.

---


## Pretrainer Utility

### setup_environment
```python
def setup_environment(
    seed: int
) -> None
```
Configure reproducible seeds and environment variables.

**Parameters**
- **seed** (`int`): Random seed for Python, NumPy, and PyTorch.

---
### make_output_dirs
```python
def make_output_dirs(
    output_dir: Path,
    run_name: str
) -> Dict[str, Path]
```
Create standardized directories for models, logs, and final outputs.

**Parameters**
- **output_dir** (`Path`): Base path for outputs.
- **run_name** (`str`): Unique identifier for this run.

**Returns**
- **Dict[str,Path]**: Paths for `training`, `logging`, and `model` outputs.

---
### choose_closest
```python
def choose_closest(
    name: str,
    supported: Sequence[str],
    unsupported: Optional[Sequence[str]] = None,
    cutoff: float = 0.4
) -> str
```
Fuzzy‐match a string against supported options, rejecting known unsupported.

**Parameters**
- **name** (`str`): Input choice.
- **supported** (`Seq[str]`): Allowed names.
- **unsupported** (`Seq[str]`, optional): Disallowed names.
- **cutoff** (`float`): Similarity threshold.

**Returns**
- **str**: Closest match from `supported`.

**Raises**
- **ValueError**: If no acceptable match is found.

---

