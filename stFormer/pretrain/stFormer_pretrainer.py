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

    High-level function to configure environment and run stFormer pretraining.
    Required: dataset_path, token_dict_path, example_lengths_path, mode, output_dir.
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

class PretrainML:
    def __init__(
            self,
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
    ):
        """
        Initialize Bert Masked Learning
        """
        self.dataset_path = dataset_path
        self.token_dict_path = token_dict_path
        self.model_type = model_type
        with open(token_dict_path,'rb') as f:
            token_dict = pickle.load(f)
        self.example_lengths_path = example_lengths_path
        self.mode = mode
        self.output_dir = output_dir
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.max_input = max_input
        self.activ_fn = activ_fn
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.optimizer_type = optimizer_type
        self.warmup_steps = warmup_steps
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.do_train = do_train
        self.do_eval = do_eval
        self.length_column_name = length_column_name
        self.disable_tqdm = disable_tqdm
        self.training_args_overrides = training_args_overrides
        
        setup_environment(seed)
        # load datasets and token dictionary
        self.train_ds = load_from_disk(dataset_path)
        self.pad_id = token_dict['<pad>']
        self.vocab_size = len(token_dict)
        

    def run_pretraining(self):
        # prepare directories
        tz = pytz.timezone('US/Central')
        now = datetime.datetime.now(tz)
        stamp = now.strftime('%y%m%d_%H%M%S')
        run_name = f"{stamp}_STgeneformer_30M_L{self.num_layers}_emb{self.embed_dim}_SL{self.max_input}_E{self.epochs}_B{self.batch_size}_LR{self.learning_rate}_LS{self.lr_scheduler_type}_WU{self.warmup_steps}_O{self.weight_decay}_DS"
        dirs = make_output_dirs(Path(self.output_dir), run_name)
        config = build_bert_config(
            self.model_type,
            self.num_layers,
            self.num_heads,
            self.embed_dim,
            self.max_input,
            self.pad_id,
            self.vocab_size,
            self.activ_fn,
            self.initializer_range,
            self.layer_norm_eps,
            self.attention_dropout,
            self.hidden_dropout
        )
        model = BertForMaskedLM(config)
        model = model.train()

        #training arguments
        training_args = get_training_arguments(
            output_dir=dirs['training'],
            logging_dir=dirs['logging'],
            train_dataset_length=len(self.train_ds),
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            optimizer_type=self.optimizer_type,
            do_train=self.do_train,
            do_eval=self.do_eval,
            length_column_name=self.length_column_name,
            disable_tqdm=self.disable_tqdm,
            overrides=self.training_args_overrides,
    )
        print(f"[INFO] Checkpoints: {dirs['training']}")
        print(f"[INFO] Logs: {dirs['logging']}")

        # run training
        trainer = STFormerPretrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_ds,
            token_dictionary=self.token_dict,
            example_lengths_file=self.example_lengths_path,
        )
        trainer.train()

        # save final model and tokenizer
        final_dir = dirs['model']
        trainer.model.save_pretrained(final_dir)
        config.save_pretrained(final_dir)
        from stFormer.tokenization.SpatialTokenize import build_custom_tokenizer
        tokenizer = build_custom_tokenizer(self.token_dict_path, self.mode)
        tokenizer.save_pretrained(final_dir)
        print("[DONE] Pretraining complete.")


    def run_hyperparameter_train(
        self,
        search_space: Dict,
        resources_per_trial,
        n_trials: int = 8,
        backend: Literal['ray','optuma'] = 'ray'
    ):
        from pathlib import Path
        import ray
        from ray import tune

        #  Initialize Ray 
        ray.shutdown()
        
        # prepare directories
        tz = pytz.timezone('US/Central')
        now = datetime.datetime.now(tz)
        stamp = now.strftime('%y%m%d_%H%M%S')
        run_name = f"{stamp}_STgeneformer_30M_L{self.num_layers}_emb{self.embed_dim}_SL{self.max_input}_E{self.epochs}_B{self.batch_size}_LR{self.learning_rate}_LS{self.lr_scheduler_type}_WU{self.warmup_steps}_O{self.weight_decay}_DS"
        dirs = make_output_dirs(Path(self.output_dir), run_name)
        config = build_bert_config(
            self.model_type,
            self.num_layers,
            self.num_heads,
            self.embed_dim,
            self.max_input,
            self.pad_id,
            self.vocab_size,
            self.activ_fn,
            self.initializer_range,
            self.layer_norm_eps,
            self.attention_dropout,
            self.hidden_dropout
        )
        model = BertForMaskedLM(config)
        model = model.train()
        
        training_args = get_training_arguments(
            output_dir=dirs['training'],
            logging_dir=dirs['logging'],
            train_dataset_length=len(self.train_ds),
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            optimizer_type=self.optimizer_type,
            do_train=self.do_train,
            do_eval=self.do_eval,
            length_column_name=self.length_column_name,
            disable_tqdm=self.disable_tqdm,
            overrides=self.training_args_overrides
        )
        def _hp_space(trial):
            params = {}
            for key, spec in search_space.items():
                if spec['type'] == 'loguniform':
                    params[key] = trial.suggest_loguniform(key, spec['low'], spec['high'])
                elif spec['type'] == 'categorical':
                    params[key] = trial.suggest_categorical(key, spec['values'])
                else:
                    raise ValueError(f"Unknown search type {spec['type']}")
            return params

        trainer = STFormerPretrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_ds,
            token_dictionary=self.token_dict,
            example_lengths_file=self.example_lengths_path,
            )
        resources = resources_per_trial or {"cpu": 4, "gpu": 1}
        best = trainer.hyperparameter_search(
            direction="minimize",
            backend=backend,
            hp_space=_hp_space,
            n_trials=n_trials,
            resources_per_trial=resources,
        )
        print("Best trial:", best)

        # save final model and tokenizer
        final_dir = dirs['model']
        config.save_pretrained(final_dir)
        from stFormer.tokenization.SpatialTokenize import build_custom_tokenizer
        tokenizer = build_custom_tokenizer(self.token_dict_path, self.mode)
        tokenizer.save_pretrained(final_dir)
        print("[DONE] Pretraining complete.")