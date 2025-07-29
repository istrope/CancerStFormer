# Usage

We introduce CancerStFormer, A flexible framework for transformer-based analysis of spatial transcriptomics data. stFormer provides tools for data tokenization, pretraining, embedding extraction, in silico perturbation, and downstream classification.

## Installation

To use cancerstformer, first create environment, install prerequisites, and install package:

```bash
#create conda environment or virtual environment
conda create -n cstformer python=3.10
conda activate cstformer

#Install Depenencies
pip install torch # version compatible with your gpu/cpu
pip install -r requirements.txt
pip install CancerstFormer

# if using deepspeed
pip install mpi4py
```
> **Prerequisites:** Python 3.8+, OpenMPI (for deepspeed only)


## Model Hub
Check out pretrained models at our hugging face repo:  [CancerStFormer](https://huggingface.co/Istrope/stFormer)

**Description:**
- `spot:` single spot resolution tokenized and pretrained model, captures expression in a 55um radius
- `neighborhood:` spot + neighbor cell resolution, captures expression around 165um radius
- `cancer:` pan-cancer pretrained model, can be utilized for cancer specific datasets

| Model | Location |
| --- | --- |
| `spot` |  [spot-model](https://huggingface.co/Istrope/stFormer/tree/main/models/spot) |
| `neighborhood` | [neighborhood-model](https://huggingface.co/Istrope/stFormer/tree/main/models/neighbor)  |
| `sequence-classifier` |  [tissue-model](https://huggingface.co/Istrope/stFormer/tree/main/models/tissue_classifier) |

## Features

- **Data Tokenization**

  - Spot-resolution and neighborhood-resolution tokenizers for Visium and other spatial platforms.
  - Support for both `.h5ad` and `.loom` file formats.

- **Pretraining**

  - `STFormerPretrainer` class for masked language modeling of gene tokens.
  - Configurable hyperparameters and Ray Tune integration for automated search.

- **Embedding Extraction**

  - `EmbeddingExtractor` module to pull cell- and gene-level embeddings.
  - Options for CLS-token or mean-pooling strategies.
  - Batch-wise, multi-core support for large datasets.

- **In Silico Perturbation**

  - `InSilicoPerturber` for single-gene or combination perturbations.
  - `InSilicoPerturberStats` to aggregate and summarize perturbation results.

- **Classification & Fine-Tuning**

  - Utilities for training cell-type or gene classifiers with Hugging Face Transformers.
  - Ray Tune experiments for hyperparameter optimization.
    
- **Network Dynamics**

  - computes attention across layers/heads for all unique token pairs
  - filters node-edges by (weight value, weight percentile, or top n edges)
  - filters noe-edges by number of co-occuring tokens in dataset
