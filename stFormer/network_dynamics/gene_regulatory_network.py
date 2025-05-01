#!/usr/bin/env python3
"""
gene_regulatory_network.py

Extract attention weights from pretrained BERT models and construct gene regulatory networks,
with optional filtering by metadata column.

Example usage:
    from gene_regulatory_network import GeneRegulatoryNetwork

    # Filter to a specific metadata group (e.g. Subtype == 'TNBC' or cell type == 'Epithelial')
    grn = GeneRegulatoryNetwork(
        model_dir="models/pretrained_model",
        dataset_path="data/visium_spot.dataset",
        model_type="Pretrained",
        metadata_column="cell_type",        # Optional: column name in your dataset
        metadata_value="Tcell",             # Optional: value to filter on
        device="cuda",
        batch_size = 16,
        nproc = 4
    )
    grn.compute_attention()
    grn.build_graph(
                    {cutoff = 0.02, percentile = 0.99 )
    grn.save_edge_list("output/tcell_network_edges.csv")
    grn.plot_network("output/tcell_network.png")

    # Or without filtering:
    grn2 = GeneRegulatoryNetwork(
        model_dir="models/pretrained_model",
        dataset_path="data/visium_spot.dataset",
        model_type="classification"
    )
    grn2.compute_attention()
    grn2.build_graph(percentile=90)
    grn2.save_edge_list("output/full_network_edges.csv")

++++++
IDEAS
++++++
Faster Computing Attention:
    1. Incorporate model pruning on attention layers/heads for faster computation
    2. exporting model to ONNX or TensorRT to speedup attention computation

Future Functionality:
    1. Incorporate Transcription Factors Lists to determine TF source -> edges
    2. Utilize motif enrichment in order to incorporate biological priors to filter list
    3. If multiple groups provided, compare networks to each other for differential regulatory analysis
"""

import logging
from pathlib import Path
from typing import Optional, Literal, Dict
from torch.utils.data import DataLoader
import numpy as np
import torch
import pickle
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForSequenceClassification,
    BertForMaskedLM,
    BertForTokenClassification
)
from tqdm import tqdm
from datasets import load_from_disk
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GeneRegulatoryNetwork:
    """
    Class to extract attention weights and build gene regulatory networks.
    """
    def __init__(
        self,
        model_dir: str,
        dataset_path: str,
        model_type: str = Literal["CellClassifier","GeneClassifier","Pretrained"],
        metadata_column: Optional[str] = None,
        metadata_value: Optional[str] = None,
        num_classes : int = 0,
        threshold: float = 0.01,
        device: Optional[str] = None,
        batch_size: int = 16,
        nproc: int = 4
    ):
        self.model_dir = Path(model_dir)
        self.dataset_path = Path(dataset_path)
        self.model_type = model_type
        self.threshold = threshold
        self.num_classes = num_classes
        self.metadata_column = metadata_column
        self.metadata_value = metadata_value
        self.batch_size = batch_size
        self.nproc = nproc
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model_and_tokenizer()
        self._load_dataset()
        

    def _load_model_and_tokenizer(self):
        # Load config with attentions turned on
        config = AutoConfig.from_pretrained(
            str(self.model_dir),
            output_attentions=True,
            output_hidden_states=True,
        )
        #output_hidden_states = (mode == 'eval')
        cls_map = {
            'Pretrained': (BertForMaskedLM,{}),
            'GeneClassifier': (BertForTokenClassification,{'num_labels': self.num_classes}),
            'CellClassifier': (BertForSequenceClassification,{'num_labels': self.num_classes})
        }
        ModelClass,extra_args = cls_map.get(self.model_type,(None,None))
        if ModelClass is None:
            raise ValueError(f'Unknown model type: {self.model_type!r}')
        # Load model
        self.model = ModelClass.from_pretrained(
            str(self.model_dir),
            config=config,
            **extra_args

        )
        self.model.to(self.device)
        self.model.eval()
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True
        )
        # Build id->token map
        self.id2token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        logger.info(f"Loaded model and tokenizer from {self.model_dir}")

    def _load_dataset(self):
        # Load HF dataset, expecting 'input_ids'
        ds = load_from_disk(str(self.dataset_path))
        #optional filter by metadata (group specific grn)
        if self.metadata_column is not None and self.metadata_value is not None:
            if self.metadata_column not in ds.column_names:
                raise ValueError(f"Metadata column '{self.metadata_column}' not found in dataset")
            ds = ds.filter(lambda ex: ex[self.metadata_column] == self.metadata_value)
            logger.info(f"Filtered dataset: {self.metadata_column} == {self.metadata_value} (n={len(ds)})")
        # Ensure attention_mask exists
        if 'attention_mask' not in ds.column_names:
            ds = ds.map(
                lambda ex: {"attention_mask": [1] * len(ex['input_ids'])},
                batched=False
            )
        # Pad all samples to the same length
        max_len = max(len(ids) for ids in ds['input_ids'])
        pad_id = self.tokenizer.pad_token_id
        def pad_fn(ex):
            seq = ex['input_ids']
            mask = ex['attention_mask']
            pad_len = max_len - len(seq)
            return {
                'input_ids': seq + [pad_id] * pad_len,
                'attention_mask': mask + [0] * pad_len
            }
        ds = ds.map(pad_fn, batched=False)
        self.dataset = ds
        self.seq_len = max_len
        logger.info(f"Loaded dataset from {self.dataset_path} (seq_len={self.seq_len})")


    def compute_attention(self):
        """
        Compute average attention weights using mixed-precision and optimized DataLoader settings.
        """
        # Optionally switch to FP16 for faster inference
        try:
            from torch.cuda.amp import autocast
            self.model = self.model.half()
            use_autocast = True
        except ImportError:
            use_autocast = False

        # Format dataset as tensors
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        def collate_fn(batch):
            # Convert tensors to Python lists for flexible padding
            ids = [item['input_ids'].tolist() for item in batch]
            masks = [item['attention_mask'].tolist() for item in batch]
            # Determine max length in this batch
            max_len_b = max(len(x) for x in ids)
            pad_id = self.tokenizer.pad_token_id
            # Pad sequences and masks to batch max length
            padded_ids = [seq + [pad_id] * (max_len_b - len(seq)) for seq in ids]
            padded_masks = [m + [0] * (max_len_b - len(m)) for m in masks]
            return {
                'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                'attention_mask': torch.tensor(padded_masks, dtype=torch.long)
            }


        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.nproc,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            collate_fn=collate_fn
        )

        attn_sum = torch.zeros((self.seq_len, self.seq_len), device=self.device)
        total_examples = 0

        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            with torch.no_grad():
                if use_autocast:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            # aggregate attentions
            attn = torch.stack(outputs.attentions)      # (layers, batch, heads, seq, seq)
            attn = attn.mean(dim=2).mean(dim=0)         # (batch, seq, seq)
            attn_sum[:, :attn.size(2)] += attn.sum(dim=0)
            total_examples += attn.size(0)

        # Restore format and finalize
        self.dataset.reset_format()
        self.attention_matrix = (attn_sum / total_examples).cpu().numpy()
        logger.info("Computed average attention matrix.")

    def build_graph(
        self,
        cutoff: Optional[float] = None,
        top_k: Optional[int] = None,
        percentile: Optional[float] = None,
        min_cooccurrence: Optional[int] = None
    ):
        """
        Build a directed graph from the attention matrix, aggregating over all samples.
        Exactly one of cutoff, top_k, or percentile must be set.
        Optionally filter pairs by minimum sample co-occurrence.
        """
        if sum(x is not None for x in [cutoff, top_k, percentile]) != 1:
            raise ValueError("Specify exactly one of cutoff, top_k, or percentile.")

        W = self.attention_matrix
        pad_id = self.tokenizer.pad_token_id

        # Identify unique tokens and compute sample presence
        logger.info("Computing sample presence per token")
        sample_presence: Dict[int, set] = {}
        for idx, ids in enumerate(self.dataset['input_ids']):
            unique_ids = set(ids)
            for tid in unique_ids:
                if tid == pad_id:
                    continue
                sample_presence.setdefault(tid, set()).add(idx)
        unique_tids = list(sample_presence.keys())
        logger.info(f"Found {len(unique_tids)} unique tokens across samples")

        # Build counts matrix over positions across all samples
        logger.info("Building position-counts matrix across samples")
        # Convert input_ids to a NumPy array: (n_samples, seq_len)
        ids_list = self.dataset['input_ids']
        id_mat = np.array(ids_list, dtype=np.int32)
        unique_arr = np.array(unique_tids, dtype=np.int32)
        counts = np.zeros((len(unique_tids), self.seq_len), dtype=np.float32)
        max_id = int(id_mat.max()) + 1
        for p in tqdm(range(self.seq_len), desc="Counting positions per column"):
            col = id_mat[:, p]  # token IDs at position p for all samples
            binc = np.bincount(col, minlength=max_id)
            counts[:, p] = binc[unique_arr]

        # Vectorized aggregation: average attention across positions: average attention across positions
        logger.info("Aggregating attention matrix via matmul")
        N = counts @ W @ counts.T
        norms = counts.sum(axis=1)
        denom = np.outer(norms, norms)
        with np.errstate(divide='ignore', invalid='ignore'):
            agg_mat = np.divide(N, denom, out=np.zeros_like(N), where=denom!=0)

        # Determine mode and build initial edge list
        modes = ['cutoff', 'top_k', 'percentile']
        vals = [cutoff, top_k, percentile]
        mode = modes[[v is not None for v in vals].index(True)]
        logger.info(f"Selecting edges by {mode}")

        edges = []
        if cutoff is not None:
            idxs = np.argwhere(agg_mat >= cutoff)
            edges = [(unique_tids[i], unique_tids[j], float(agg_mat[i, j])) for i, j in idxs]
        elif percentile is not None:
            thr = np.percentile(agg_mat, percentile)
            idxs = np.argwhere(agg_mat >= thr)
            edges = [(unique_tids[i], unique_tids[j], float(agg_mat[i, j])) for i, j in idxs]
        else:  # top_k per source
            logger.info(f"Selecting top_k={top_k} per source token")
            for i in tqdm(range(len(unique_tids)), desc="Selecting top_k"):
                row = agg_mat[i]
                top_idx = np.argpartition(-row, top_k)[:top_k]
                edges.extend([(unique_tids[i], unique_tids[j], float(row[j])) for j in top_idx])

        # Optional co-occurrence filtering
        if min_cooccurrence is not None:
            logger.info(f"Filtering edges by minimum co-occurrence {min_cooccurrence}")
            filtered = []
            for src_id, tgt_id, w in edges:
                co = len(sample_presence.get(src_id, set()) & sample_presence.get(tgt_id, set()))
                if co >= min_cooccurrence:
                    filtered.append((src_id, tgt_id, w))
            edges = filtered
            logger.info(f"{len(edges)} edges remain after co-occurrence filter")

        # Build graph
        logger.info("Building graph from selected edges")
        G = nx.DiGraph()
        for src_id, tgt_id, w in edges:
            src = self.id2token.get(src_id, str(src_id))
            tgt = self.id2token.get(tgt_id, str(tgt_id))
            G.add_edge(src, tgt, weight=w)
        self.graph = G
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")    
    def save_edge_list(
        self, 
        output_path: str,
        gene_name_id_dictionary_file: Optional[str] = None
    ):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if gene_name_id_dictionary_file is not None:
            with open(gene_name_id_dictionary_file,'rb') as f:
                gene_id_dict = pickle.load(f)
            id_gene_dict = {ens_id:gene for gene,ens_id in gene_id_dict.items()}
        #holders if no id mapping dict
        target_gene = ''
        source_gene = ''
        #write to file
        with p.open('w') as f:
            f.write("source,source_gene,target,target_gene,weight\n")
            for source, target, d in self.graph.edges(data=True):
                if self.tokenizer.pad_token_id in [source,target,d]:
                    pass
                if gene_name_id_dictionary_file is not None:
                    source_gene = id_gene_dict[source]
                    target_gene = id_gene_dict[target]
                
                f.write(f"{source},{source_gene},{target},{target_gene},{d['weight']}\n")
        logger.info(f"Edge list saved to {p}")

    def plot_network(self, output_path: str):
        pos = nx.spring_layout(self.graph, seed=42)
        weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges()]
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_size=50)
        nx.draw_networkx_edges(
            self.graph,
            pos,
            width=[w * 5 for w in weights],
            arrowsize=5
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=6)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        logger.info(f"Network plot saved to {output_path}")
