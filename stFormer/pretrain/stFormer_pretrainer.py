"""
Module to configure and run stFormer pretraining as importable functions.

Example usage:
    from stformer_pretrainer import run_pretraining

    run_pretraining(
        dataset_path="path/to/dataset",
        token_dict_path="path/to/token_dict.pkl",
        example_lengths_path="path/to/lengths.pkl",
        mode='spot',
        rootdir="output/root",
        seed=42,
        num_layers=6,
        num_heads=4,
        embed_dim=256,
        max_input=2048,
        batch_size=12,
        lr=1e-3,
        warmup_steps=10000,
        epochs=3,
        weight_decay=0.001,
    )
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import pytz
import random
import datetime
import os
from typing import Literal
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
import pickle

from stFormer.pretrain.pretrainer import STFormerPretrainer, load_example_lengths


def setup_environment(seed: int) -> None:
    """Configure environment variables and random seeds."""
    os.environ.update({
        'NCCL_DEBUG': 'INFO',
        'OMPI_MCA_opal_cuda_support': 'true',
        'CONDA_OVERRIDE_GLIBC': '2.56',
    })
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_output_dirs(rootdir: Path, run_name: str) -> Dict[str, Path]:
    """Create and return paths for training, logging, and model outputs."""
    dirs = {
        'training': rootdir / 'models' / run_name,
        'logging': rootdir / 'runs' / run_name,
        'model': rootdir / 'models' / run_name / 'final',
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def build_bert_config(
    num_layers: int,
    num_heads: int,
    embed_dim: int,
    max_input: int,
    pad_id: int,
    vocab_size: int,
    activ_fn: str = 'relu',
    init_range: float = 0.02,
    norm_eps: float = 1e-12,
    attn_dropout: float = 0.02,
    hidden_dropout: float = 0.02,
) -> BertConfig:
    """Construct a BertConfig for stFormer."""
    return BertConfig(
        hidden_size=embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=embed_dim * 2,
        hidden_act=activ_fn,
        initializer_range=init_range,
        layer_norm_eps=norm_eps,
        attention_probs_dropout_prob=attn_dropout,
        hidden_dropout_prob=hidden_dropout,
        max_position_embeddings=max_input,
        pad_token_id=pad_id,
        vocab_size=vocab_size,
    )



def run_pretraining(
    dataset_path: str,
    token_dict_path: str,
    example_lengths_path: str,
    rootdir: str,
    mode: Literal['spot', 'neighborhood'] = 'spot',
    seed: int = 42,
    num_layers: int = 6,
    num_heads: int = 4,
    embed_dim: int = 256,
    max_input: int = 2048,
    batch_size: int = 12,
    lr: float = 1e-3,
    warmup_steps: int = 10000,
    epochs: int = 3,
    weight_decay: float = 0.001,
) -> None:
    """
    High‑level function to execute stFormer pretraining.
    """
    setup_environment(seed)

    #  load data & token dict 
    train_ds = load_from_disk(dataset_path)
    token_dict = pickle.load(open(token_dict_path, 'rb'))
    pad_id     = token_dict['<pad>']
    vocab_size = len(token_dict)

    #  name & dirs 
    tz   = pytz.timezone('US/Central')
    now  = datetime.datetime.now(tz)
    stamp = now.strftime('%y%m%d_%H%M%S')
    run_name = f"{stamp}_stFormer_L{num_layers}_E{epochs}"
    dirs = make_output_dirs(Path(rootdir), run_name)

    #  config & model 
    config = build_bert_config(
        num_layers, num_heads, embed_dim, max_input, pad_id, vocab_size
    )
    model = BertForMaskedLM(config)

    #  training args 
    save_steps = max(1, len(train_ds) // (batch_size * 8))
    training_args = TrainingArguments(
        output_dir=str(dirs['training']),
        logging_dir=str(dirs['logging']),
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        group_by_length=True,
        save_strategy='steps',
        save_steps=save_steps,
        logging_steps=1000,
    )

    print(f"\n[INFO] Checkpoints will go to: {dirs['training']}")
    print(f"[INFO] TensorBoard logs will go to: {dirs['logging']}\n")

    #  trainer & train 
    trainer = STFormerPretrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        token_dictionary=token_dict,
        example_lengths_file=example_lengths_path,
    )
    trainer.train()

    #  save the final frozen model 
    final_dir = dirs['model']
    print(f"\n[INFO] Saving final model to: {final_dir}\n")
    trainer.model.save_pretrained(final_dir)
    config.save_pretrained(final_dir)

    from stFormer.tokenization.SpatialTokenize import build_custom_tokenizer

    tokenizer = build_custom_tokenizer(token_dict_path,mode)
    tokenizer.save_pretrained(final_dir)

    print(f"[DONE] Pretraining complete!")
    print("Contents of final model dir:", os.listdir(final_dir))
    print("Contents of checkpoint dir:", os.listdir(dirs['training']), "\n")