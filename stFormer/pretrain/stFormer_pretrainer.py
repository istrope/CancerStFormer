"""
Module to configure and run stFormer pretraining as importable functions.

Example usage:
    from stformer_pretrainer import execute_pretraining

    execute_pretraining(
        dataset_path="path/to/dataset",
        token_dict_path="path/to/token_dict.pkl",
        example_lengths_path="path/to/lengths.pkl",
        mode='spot',
        output_dir="output/root"
    )
"""
from pathlib import Path
from typing import Dict, Optional, Literal
import os
import datetime
import random
import pytz
import numpy as np
import torch
import pickle
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from stFormer.pretrain.pretrainer import STFormerPretrainer


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


def make_output_dirs(output_dir: Path, run_name: str) -> Dict[str, Path]:
    """Create and return paths for training, logging, and model outputs."""
    dirs = {
        'training': output_dir / 'models' / run_name,
        'logging':  output_dir / 'runs'   / run_name,
        'model':    output_dir / 'models' / run_name / 'final',
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_bert_config(
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
) -> BertConfig:
    """Construct a BertConfig for stFormer."""
    return BertConfig(
        model_type=model_type,
        hidden_size=embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=embed_dim * 2,
        hidden_act=activ_fn,
        initializer_range=initializer_range,
        layer_norm_eps=layer_norm_eps,
        attention_probs_dropout_prob=attention_dropout,
        hidden_dropout_prob=hidden_dropout,
        max_position_embeddings=max_input,
        pad_token_id=pad_id,
        vocab_size=vocab_size,
    )


def get_training_arguments(
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
    overrides: Optional[Dict] = None
) -> TrainingArguments:
    """
    Build a TrainingArguments object, merging defaults with any overrides.
    """
    defaults = {
        'output_dir': str(output_dir),
        'logging_dir': str(logging_dir),
        'per_device_train_batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_train_epochs': epochs,
        'weight_decay': weight_decay,
        'warmup_steps': warmup_steps,
        'lr_scheduler_type': lr_scheduler_type,
        'optim': optimizer_type,
        'do_train': do_train,
        'do_eval': do_eval,
        'group_by_length': True,
        'length_column_name': length_column_name,
        'disable_tqdm': disable_tqdm,
        'save_strategy': 'steps',
        'save_steps': max(1, train_dataset_length // (batch_size * 8)),
        'logging_steps': 1000,
    }
    if overrides:
        defaults.update(overrides)
    return TrainingArguments(**defaults)


def run_pretraining(
    dataset_path: str,
    token_dict_path: str,
    example_lengths_path: str,
    mode: Literal['spot', 'neighborhood'],
    output_dir: str,
    # The rest are optional with sensible defaults:
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
    optimizer_type: str = 'adamw_hf',
    warmup_steps: int = 10000,
    epochs: int = 3,
    weight_decay: float = 0.001,
    do_train: bool = True,
    do_eval: bool = False,
    length_column_name: str = 'length',
    disable_tqdm: bool = False,
    training_args_overrides: Optional[Dict] = None,
) -> None:
    """
    High-level function to configure environment and run stFormer pretraining.
    Required: dataset_path, token_dict_path, example_lengths_path, mode, output_dir.
    """
    setup_environment(seed)

    # load datasets and token dictionary
    train_ds = load_from_disk(dataset_path)
    token_dict = pickle.load(open(token_dict_path, 'rb'))
    pad_id = token_dict['<pad>']
    vocab_size = len(token_dict)

    # prepare directories
    tz = pytz.timezone('US/Central')
    now = datetime.datetime.now(tz)
    stamp = now.strftime('%y%m%d_%H%M%S')
    run_name = f"{stamp}_STgeneformer_30M_L{num_layers}_emb{embed_dim}_SL{max_input}_E{epochs}_B{batch_size}_LR{learning_rate}_LS{lr_scheduler_type}_WU{warmup_steps}_O{weight_decay}_DS"
    dirs = make_output_dirs(Path(output_dir), run_name)

    # build model
    config = build_bert_config(
        model_type,
        num_layers,
        num_heads,
        embed_dim,
        max_input,
        pad_id,
        vocab_size,
        activ_fn,
        initializer_range,
        layer_norm_eps,
        attention_dropout,
        hidden_dropout,
    )
    model = BertForMaskedLM(config)
    model = model.train()

    # training arguments
    training_args = get_training_arguments(
        output_dir=dirs['training'],
        logging_dir=dirs['logging'],
        train_dataset_length=len(train_ds),
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        optimizer_type=optimizer_type,
        do_train=do_train,
        do_eval=do_eval,
        length_column_name=length_column_name,
        disable_tqdm=disable_tqdm,
        overrides=training_args_overrides,
    )

    print(f"[INFO] Checkpoints: {dirs['training']}")
    print(f"[INFO] Logs: {dirs['logging']}")

    # run training
    trainer = STFormerPretrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        token_dictionary=token_dict,
        example_lengths_file=example_lengths_path,
    )
    trainer.train()

    # save final model and tokenizer
    final_dir = dirs['model']
    trainer.model.save_pretrained(final_dir)
    config.save_pretrained(final_dir)
    from stFormer.tokenization.SpatialTokenize import build_custom_tokenizer
    tokenizer = build_custom_tokenizer(token_dict_path, mode)
    tokenizer.save_pretrained(final_dir)
    print("[DONE] Pretraining complete.")
