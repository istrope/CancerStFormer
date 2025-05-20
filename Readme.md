# stFormer

A flexible framework for transformer-based analysis of spatial transcriptomics data. stFormer provides tools for data tokenization, pretraining, embedding extraction, in silico perturbation, and downstream classification.

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
pip install torch
pip install -r requirements.txt
pip install stformer
```

> **Prerequisites:** Python 3.8+, PyTorch, Transformers

## Usage

### 1. Tokenization and Median Estimator
```
```

```python

```

### 2. Pretraining

```python

```

```python

```


### 3. Classification

```python
from stFormer.classifier.Classifier import GenericClassifier

# Fine Tune Classifier Based on Metadata Group and Train from a Pretrained MaskedLM Model
classifier = GenericClassifier(
    metadata_column = 'subtype', #what metadata to train classifier on 
    nproc=24)
    
ds_path, map_path = classifier.prepare_data(
    input_data_file = 'output/spot/visium_spot.dataset',
    output_directory = 'tmp',
    output_prefix = 'visium_spot'
    )

trainer = classifier.train(
    model_checkpoint='output/spot/models/250422_102707_stFormer_L6_E3/final',
    dataset_path = ds_path,
    output_directory = 'output/models/classification',
    test_size=0.2 #create test/train split
)
```
```python
from stFormer.classifier.Classifier import GenericClassifier

# Fine-Tuning with Hyperparameter Tuning (saves best model)
classifier = GenericClassifier(
    metadata_column = 'subtype',
    ray_config={ #check BertForSequenceClassification for addional hyperparameters
        "learning_rate":[1e-5,5e-5], #loguniform learning rate
        "num_train_epochs":[2,3], #choice
        "weight_decay": [0.0, 0.3], #tune.uniform across values
        'lr_scheduler_type': ["linear","cosine","polynomial"], #scheduler
        'seed':[0,100]
        },
    nproc = 24
    )

ds_path, map_path = classifier.prepare_data(
    input_data_file = 'output/spot/visium_spot.dataset',
    output_directory = 'tmp',
    output_prefix = 'visium_spot'
    )

best_run = classifier.train(
    model_checkpoint='output/spot/models/250422_102707_stFormer_L6_E3/final',
    dataset_path = ds_path,
    output_directory = 'output/models/tuned_classification',
    n_trials=10,
    test_size=0.2 #Test/Train Split
)
```


### 5. In Silico Perturbation

```python
from stFormer.perturbation.stFormer_perturb import InSilicoPerturber
from stFormer.perturbation.perturb_stats import InSilicoPerturberStats

isp = InSilicoPerturber(
            perturb_type="delete",
            genes_to_perturb=['ENSG00000188389'],
            #perturb_rank_shift = None,
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cell_and_gene",
            cell_emb_style='mean_pool',
            max_ncells=1000,
            emb_layer=0,
            forward_batch_size=10,
            nproc=12,
            token_dictionary_file='output/token_dictionary.pickle',
         )

isp.perturb_data(
    model_directory='output/spot/models/250422_102707_stFormer_L6_E3/final',
    input_data_file='output/spot/visium_spot.dataset',
    output_directory='output/perturb',
    output_prefix='perturb_spot')

ispstats = InSilicoPerturberStats(
    mode='aggregate_gene_shifts',
    genes_perturbed = ['ENSG00000188389'],
    pickle_suffix = '_raw.pickle',
    token_dictionary_file='output/token_dictionary.pickle',
    gene_name_id_dictionary_file='output/ensembl_mapping_dict.pickle'
)
ispstats.get_stats('output/perturb',None,'output/perturb_stats','perturb_spot')
```

### 5. Network Graph

```python
from stFormer.network_dynamics.gene_regulatory_network import GeneRegulatoryNetwork

# Compute Gene Regulatory Network by Capturing Model Attention Weights Across Pairs
grn = GeneRegulatoryNetwork(
    model_dir = 'output/spot/models/250422_102707_stFormer_L6_E3/final',
    dataset_path = 'output/spot/visium_spot.dataset',
    model_type = 'Pretrained',
    metadata_column = 'subtype',
    metadata_value = 'TNBC',
    device='cuda',
    batch_size=24,
    nproc = 12
)

# computes average attention for a gene in a batch from pretrained model
grn.compute_attention()

#Can take a long time, computing average i,j attention across all samples for all token pairs
grn.build_graph(
    percentile = 0.99999, #filter node-edges by attention weight (value above percentile)
    min_cooccurrence=100 #filter node-edges by number of samples expressing pair
)

```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes and new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


