# stFormer

A flexible framework for transformer-based analysis of spatial transcriptomics data. stFormer provides tools for data tokenization, pretraining, embedding extraction, in silico perturbation, and downstream classification.

## Features

- **Data Tokenization**

  - Spot-resolution and neighborhood-resolution tokenizers for Visium and other spatial platforms.
  - Support for both `.h5ad` and `.loom` file formats.
  - Gene median estimation using t-Digest
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
pip install torch torchaudio torchtext
pip install -r requirements.txt
```

> **Prerequisites:** Python 3.8+, PyTorch, Transformers, Datasets, Scanpy, Anndata, Ray Tune.

## Usage

### 1. Tokenization and Median Estimator
``` python
# Compute T-Digests
from stFormer.tokenization.median_estimator import MedianEstimator
estimator = MedianEstimator(
    data_dir = 'data',
    extension = '.h5ad',
    out_path = 'output',
    merge_tdigests = True 
)
estimator.compute_tdigests(
      file_path #optional processes single file, unless loads all in data_dir)
estimator.get_median_dict()
estimator.write_tdigests() # write to file in out_path
estimator.write_medians() # write to file in out_path
```

```python
from stFormer.tokenization.SpatialTokenize import SpatialTokenizer, create_token_dictionary

#create token dictionary
token_dictionary = create_token_dictionary(median_dict=median_dict)
with open('output/token_dictionary.pickle','wb') as file:
    pickle.dump(token_dictionary,file)

# Build spot-resolution tokenizer
tok = SpatialTokenizer(
    token_dict_file="output/token_dictionary.pickle",
    gene_median_file='output/gene_medians.pickle',
    nproc = 12,
    mode="spot",  # or "neighbor"
    custom_meta = {'sample_id' : 'sample', 'class' : 'classification', 'subtype' : 'subtype'}
)
# Tokenize dataset
tok.tokenize(
    data_dir = "data/visium_spot.h5ad",
    out_dir = "output/spot",
    prefix = 'spot',
    file_format = 'h5ad')
```

### 2. Pretraining

```python
from datasets import load_from_disk
import pickle

# Create example lengths file from tokenized data
ds = load_from_disk('output/spot/visium_spot.dataset')
lengths = [len(example['input_ids']) for example in ds]
with open('output/example_lengths.pickle','wb') as file:
    pickle.dump(lengths,file)
```

```python
from stFormer.pretrain.stFormer_pretrainer import run_pretraining

run_pretraining(
   dataset_path='output/spot/visium_spot.dataset',
   token_dict_path='output/token_dictionary.pickle',
   example_lengths_path='output/example_lengths.pickle',
   rootdir='output/spot',
   seed=42,
   num_layers=6,
   num_heads=4,
   embed_dim=256,
   max_input=2048, #2048 for spot and 4096 for neighbor
   batch_size=12,
   lr=1e-3,
   warmup_steps=10000,
   epochs=3,
   weight_decay=0.001
)
```

### 3. Embedding Extraction

```python
from stFormer.tokenization.embedding_extractor import EmbeddingExtractor

extractor = EmbeddingExtractor(
    token_dict_path=Path('output/token_dictionary.pickle'),
    emb_mode='cls',
    emb_layer = -1,
    forward_batch_size=64
    )
embeddings = extractor.extract_embs(
    model_directory='output/spot/models/250422_102707_stFormer_L6_E3/final',
    dataset_path='output/spot/visium_spot.dataset',
    output_directory='output/spot/embeddings',
    output_prefix='visium_spot'

)
```
### 4. Classification

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
    test_size=0.2, #create test/train split
    #
    stratify=False
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
    test_size=0.2, #Test/Train Split
    stratify=False
)
```

```python
#save confusion matrix heatmap with inbuilt plotting functionality

classifier.plot_predictions(
    predictions_file="output/models/classification/predictions.pkl",
    id_class_dict_file=map_path,
    title="Visium Spot Subtype Predictions",
    output_directory="output/models/classification",
    output_prefix="visium_spot",
    class_order=class_order
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

grn.save_edge_list(
    output_path = 'output/spot/gene_network_edges.csv')

grn.plot_network('output/spot/gene_network.png')
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes and new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


