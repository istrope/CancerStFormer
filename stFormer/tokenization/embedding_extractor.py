#!/usr/bin/env python3
"""
 Provides functionality to extract embeddings from a tokenized dataset using a pretrained model.

Usage (as library):
    from pathlib import Path
    from embedding_extractor import EmbeddingExtractor

    # instantiate extractor (replace paths and params as needed)
    extractor = EmbeddingExtractor(
        token_dict_path=Path("output/token_dictionary.pkl"),
        emb_mode="cls",
        emb_layer=-1,
        forward_batch_size=64,
    )

    # run extraction
    embeddings = extractor.extract_embs(
        model_directory="output/models/my_pretrained_model",
        dataset_path="data/visium_spot.dataset",
        output_directory="embeddings",
        output_prefix="visium_spot"
    )
"""

import logging
import os
import pickle
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel
from datasets import load_from_disk
from tqdm import tqdm

logger = logging.getLogger(__name__)


def pad_sequences(seq_list, max_length, pad_value, model_input_size, extra_dim=0):
    """
    Pads or truncates sequences in seq_list to max_length, then ensures model_input_size
    by further padding if needed. Returns a tensor of shape (batch, model_input_size, ...).
    """
    padded = []
    for seq in seq_list:
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq)
        curr_len = seq.size(0)
        if curr_len < max_length:
            pad_shape = (max_length - curr_len, *seq.shape[1:])
            pad_tensor = torch.full(pad_shape, pad_value, dtype=seq.dtype)
            seq = torch.cat([seq, pad_tensor], dim=0)
        else:
            seq = seq[:max_length]
        padded.append(seq)

    padded_tensor = torch.stack(padded)
    # ensure at least model_input_size
    seq_len = padded_tensor.size(1)
    if seq_len < model_input_size:
        extra = model_input_size - seq_len
        extra_pad_shape = (padded_tensor.size(0), extra, *padded_tensor.shape[2:])
        extra_pad = torch.full(extra_pad_shape, pad_value, dtype=padded_tensor.dtype)
        padded_tensor = torch.cat([padded_tensor, extra_pad], dim=1)
    return padded_tensor


def generate_attention_mask(batch, pad_token_id):
    """
    Generates an attention mask (1 for real tokens, 0 for padding) from the padded batch.
    """
    return (batch != pad_token_id).long()


def mean_pool_embeddings(embeddings, lengths, skip_first=False, skip_last=False):
    """
    Computes mean pooling over token embeddings for each example in the batch.
    Optionally skips the first and/or last token.
    """
    pooled = []
    for i, emb in enumerate(embeddings):
        start = 1 if skip_first else 0
        end = lengths[i] - 1 if skip_last else lengths[i]
        pooled.append(emb[start:end].mean(dim=0))
    return torch.stack(pooled)


class EmbeddingExtractor:
    def __init__(
        self,
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cls",
        filter_data=None,
        max_ncells=1000,
        emb_layer=-1,
        emb_labels=None,
        forward_batch_size=100,
        summary_stat=None,
        token_dict_path: Path = None,
    ):
        self.model_type = model_type
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.filter_data = filter_data
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.emb_labels = emb_labels
        self.forward_batch_size = forward_batch_size
        self.summary_stat = summary_stat

        if token_dict_path is None:
            raise ValueError("A valid token dictionary path must be provided.")
        with open(token_dict_path, "rb") as f:
            self.token_dict = pickle.load(f)

        # reverse mapping from token ID to gene name
        self.token_to_gene = {v: k for k, v in self.token_dict.items()}
        self.pad_token_id = self.token_dict.get("<pad>")

    def _get_model_input_size(self, model):
        # use model's maximum position embeddings if available
        try:
            return model.config.max_position_embeddings
        except Exception:
            return 2048

    def _get_embedding_dim(self, model):
        return model.config.hidden_size

    def _get_target_layer(self, model):
        # return the index of the hidden_states layer to extract
        return self.emb_layer

    def extract_embs(
        self,
        model_directory,
        dataset_path,
        output_directory,
        output_prefix,
    ):
        """
        Extracts embeddings for each example in a tokenized dataset using a pretrained model.

        Args:
            model_directory: path to the pretrained model directory
            dataset_path: path to a HuggingFace dataset (use load_from_disk)
            output_directory: directory to write the output .pt file
            output_prefix: prefix for the saved file name

        Returns:
            Tensor of shape (num_examples, hidden_size)
        """
        # load tokenized dataset
        ds = load_from_disk(dataset_path)
        n_examples = len(ds)

        # check once if dataset has length feature
        has_length = "length" in ds.features

        # prepare output directory
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        # load model and configure hidden states
        config = AutoConfig.from_pretrained(model_directory)
        config.output_hidden_states = True
        model = AutoModel.from_pretrained(
            model_directory,
            config=config,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        all_embs = []
        all_lengths = []

        # batch-wise extraction with progress bar
        num_batches = (n_examples + self.forward_batch_size - 1) // self.forward_batch_size
        for start in tqdm(range(0, n_examples, self.forward_batch_size),
                          desc="Extracting embeddings",
                          total=num_batches):
            end = min(start + self.forward_batch_size, n_examples)
            batch = ds[start:end]

            # get input_ids and lengths
            input_ids = batch["input_ids"]
            if has_length:
                lengths = batch["length"]
            else:
                lengths = [len(x) for x in input_ids]

            # pad sequences & build attention mask
            max_len = max(lengths)
            model_input_size = self._get_model_input_size(model)
            padded_ids = pad_sequences(
                [torch.tensor(x) for x in input_ids],
                max_length=max_len,
                pad_value=self.pad_token_id,
                model_input_size=model_input_size,
            )
            attention_mask = generate_attention_mask(padded_ids, self.pad_token_id)

            # forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=padded_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states

            # pick out your target layer
            layer_output = hidden_states[self._get_target_layer(model)]

            # collapse seq_len → embedding_dim
            if self.emb_mode == "cls":
                batch_embs = layer_output[:, 0, :]
            elif self.emb_mode == "pooler":
                batch_embs = (
                    outputs.pooler_output
                    if hasattr(outputs, "pooler_output")
                    else layer_output[:, 0, :]
                )
            elif self.emb_mode == "mean":
                batch_embs = mean_pool_embeddings(
                    layer_output.cpu(), lengths, skip_first=True
                )
            else:
                raise ValueError(f"Unknown emb_mode: {self.emb_mode}")

            all_embs.append(batch_embs.cpu())
            all_lengths.extend(lengths)

        # concatenate and save
        embs_tensor = torch.cat(all_embs, dim=0)
        output_file = out_dir / f"{output_prefix}.pt"
        torch.save(embs_tensor, output_file)
        logger.info(f"Saved embeddings to {output_file}")

        return embs_tensor
