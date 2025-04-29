#!/usr/bin/env python
"""
hpo_pretrainer.py

Optional hyperparameter search wrapper for stFormer pretraining.

Example usage:
"""



training_config = {
    "dataset_path": "path/to/dataset",
    "token_dict_path": "path/to/token_dict.pkl",
    "example_lengths_path": "path/to/lengths.pkl",
    "rootdir": "output/root",
    "seed": 42,
    "num_layers": 6,
    "num_heads": 4,
    "embed_dim": 256,
    "max_input": 2048,
    # HuggingFace TrainingArguments kwargs
    "training_args": {
        "per_device_train_batch_size": 12,
        "learning_rate": 1e-3,
        "num_train_epochs": 3,
        "weight_decay": 0.001,
        "warmup_steps": 10000,
        "group_by_length": True,
        "save_strategy": "steps",
        "save_steps": 1000,
        "logging_steps": 500,
    },
    # Optional evaluation dataset for HPO
    "eval_dataset_path": "path/to/eval_dataset",
    # Hyperparameter search space
    "hpo_search_space": {
        "learning_rate":       {"type": "loguniform",  "low": 1e-5, "high": 1e-3},
        "per_device_train_batch_size": {"type": "categorical", "values": [4, 8, 16]},
        "weight_decay":        {"type": "loguniform",  "low": 1e-6, "high": 1e-2},
    },
    "n_trials": 8,
    "hpo_backend": "optuna",  # or "ray"
    "resources_per_trial": {
        "cpu": 2,
        "gpu": 1
    }
}


import json
from pathlib import Path
import pickle
import random
import numpy as np
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from stFormer.pretrain.pretrainer import STFormerPretrainer, load_example_lengths
from stFormer.pretrain.stFormer_pretrainer import (
    setup_environment, make_output_dirs, build_bert_config
)


def run_training(
    dataset_path: str,
    token_dict_path: str,
    example_lengths_path: str,
    rootdir: str,
    seed: int,
    num_layers: int,
    num_heads: int,
    embed_dim: int,
    max_input: int,
    training_args: dict,
    eval_dataset_path: str = None,
    hpo_search_space: dict = None,
    n_trials: int = 5,
    hpo_backend: str = "optuna",
    resources_per_trial: dict = None,
):
    # 1. Set seed
    setup_environment(seed)

    # 2. Load data
    train_ds = load_from_disk(dataset_path)
    token_dict = pickle.load(open(token_dict_path, 'rb'))
    lengths = example_lengths_path

    # 3. Build config + model
    pad_id = token_dict.get('<pad>')
    vocab_size = len(token_dict)
    config = build_bert_config(
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embed_dim,
        max_input=max_input,
        pad_id=pad_id,
        vocab_size=vocab_size,
    )
    model = BertForMaskedLM(config).train()

    # 4. TrainingArguments
    output_base = Path(rootdir)
    run_name = f"hpo_run_L{num_layers}_E{training_args.get('num_train_epochs')}"
    dirs = make_output_dirs(output_base, run_name)
    args = TrainingArguments(
        output_dir=str(dirs['training']),
        logging_dir=str(dirs['logging']),
        **training_args
    )

    # 5. Set up trainer
    trainer = STFormerPretrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        token_dictionary=token_dict,
        example_lengths_file=lengths,
    )

    # 6. Optionally add eval_dataset
    if eval_dataset_path:
        eval_ds = load_from_disk(eval_dataset_path)
        trainer.eval_dataset = eval_ds

    # 7. Hyperparameter search
    if hpo_search_space:
        def _hp_space(trial):
            params = {}
            for key, spec in hpo_search_space.items():
                if spec['type'] == 'loguniform':
                    params[key] = trial.suggest_loguniform(key, spec['low'], spec['high'])
                elif spec['type'] == 'categorical':
                    params[key] = trial.suggest_categorical(key, spec['values'])
                else:
                    raise ValueError(f"Unknown search type {spec['type']}")
            return params

        resources = resources_per_trial or {"cpu": 4, "gpu": 1}
        best = trainer.hyperparameter_search(
            direction="minimize",
            backend=hpo_backend,
            hp_space=_hp_space,
            n_trials=n_trials,
            resources_per_trial=resources,
        )
        print("Best trial:", best)
    else:
        # 8. Regular training
        trainer.train()
        trainer.save_model(str(dirs['model']))


if __name__ == '__main__':
    # If you want to load config from JSON, uncomment below:
    # import sys
    # cfg = json.load(open(sys.argv[1]))
    # run_training(**cfg)

    # Otherwise, call run_training with the dict above
    run_training(**training_config)
