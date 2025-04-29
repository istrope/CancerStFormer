# stFormer

A flexible framework for transformer-based analysis of spatial transcriptomics and single-cell RNA-seq data. stFormer provides tools for data tokenization, pretraining, embedding extraction, in silico perturbation, and downstream classification.

## Features

- **Data Tokenization**

  - Spot-resolution and neighborhood-resolution tokenizers for Visium and other spatial platforms.
  - Support for both `.h5ad` and `.loom` file formats.
  - Gene median estimation using t-Digest or custom methods.
  - Memory-efficient scanning of large Anndata/loom files.

- **Pretraining**

  - `STFormerPretrainer` class for masked language modeling of gene tokens.
  - Configurable hyperparameters and Ray Tune integration for automated search.

- **Embedding Extraction**

  - `EmbeddingExtractor` module to pull cell- and gene-level embeddings.
  - Options for CLS-token or mean-pooling strategies.
  - Batch-wise, multi-core support for large datasets.

- **In Silico Perturbation**

  - `InSilicoPerturber` for single-gene or combination perturbations.
  - Cosine-similarity shifts in cell and gene embedding spaces.
  - Anchor-gene support for targeted perturbation experiments.
  - `InSilicoPerturberStats` to aggregate and summarize perturbation results.

- **Classification & Fine-Tuning**

  - Utilities for training cell-type or gene classifiers with Hugging Face Transformers.
  - Ray Tune experiments for hyperparameter optimization.
  - Best-model extraction and checkpoint management.

- **Utility Scripts**

  - Dataset inspection: view `input_ids`, `length`, and metadata fields in Hugging Face `.dataset` files.
  - Gene-marker extraction and ranking with Scanpy.
  - Parallelized DE testing and outlier removal for UMAP embeddings.

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/stFormer.git
cd stFormer

# Install dependencies
pip install -r requirements.txt
```

> **Prerequisites:** Python 3.8+, PyTorch, Transformers, Datasets, Scanpy, Anndata, Ray Tune.

## Usage

### 1. Tokenization

```python
from stFormer.tokenization.tokenization import SpatialTokenizer, compute_tdigest_medians

# Estimate gene medians (supports .h5ad or .loom)
compute_tdigest_medians("data/sample.h5ad", output="medians.pickle")

# Build spot-resolution tokenizer
tok = SpatialTokenizer(
    token_dictionary_file="token_dictionary.pickle",
    mode="spot",  # or "neighbor"
    max_cells=100,
    pad_token_id=0,
)
# Tokenize dataset
tok.tokenize("data/visium_spot.h5ad", output_dataset="spot.dataset")
```

### 2. Pretraining

```python
from stFormer.pretrainer_refactored import STFormerPretrainer, load_example_lengths
from transformers import BertConfig, BertForMaskedLM, TrainingArguments

# Prepare model and args
config = BertConfig(vocab_size=len(token_dict), pad_token_id=token_dict["<pad>"])
model = BertForMaskedLM(config)
args = TrainingArguments(
    output_dir="output/models",
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

# Initialize and train
trainer = STFormerPretrainer(
    model=model,
    args=args,
    train_dataset=load_from_disk("spot.dataset"),
    token_dictionary=token_dict,
    example_lengths_file="example_lengths.pickle"
)
trainer.train()
```

### 3. Embedding Extraction

```python
from stFormer.tokenization.embedding_extractor import EmbeddingExtractor

extractor = EmbeddingExtractor(
    token_dict_path="token_dictionary.pickle",
    emb_mode="cell_and_gene",
    emb_layer=1,
    forward_batch_size=32,
)
extractor.extract_embs(
    model_directory="output/models/final",
    dataset_path="spot.dataset",
    output_directory="output/embeddings",
    output_prefix="visium_spot"
)
```

### 4. In Silico Perturbation

```python
from stFormer.perturbation.perturber_2 import InSilicoPerturber

isp = InSilicoPerturber(
    perturb_type="delete",
    genes_to_perturb=["ENSG00000148400", ...],
    model_type="CellClassifier",
    emb_mode="cell_and_gene",
    forward_batch_size=16,
)
isp.perturb_data(
    model_path="output/models/final",
    input_data_file="spot.dataset",
    output_directory="output/perturb",
    output_prefix="perturb_spot"
)
```

### 5. Perturbation Summary

```python
from stFormer.perturbation.in_silico_perturber_stats import InSilicoPerturberStats

stats = InSilicoPerturberStats(
    mode="cell_and_gene",
    genes_perturbed=[...],
    combos=1,
    anchor_gene="ENSG00000148400",
)
stats.compute_summary("output/perturb")
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes and new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


