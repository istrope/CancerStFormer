#!/usr/bin/env python3

import os
import pickle
import logging
from pathlib import Path
import torch
torch.backends.cudnn.benchmark = True
from datasets import load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)

import classifier_utils as cu
import evaluation_utils as eu
from classifier_utils import DataCollatorForGeneClassification

logger = logging.getLogger(__name__)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)



def build_custom_tokenizer(token_dict_path: str,
                           pad_token: str = "<pad>",
                           mask_token: str = "<mask>") -> PreTrainedTokenizerFast:
    with open(token_dict_path, "rb") as f:
        vocab = pickle.load(f)
    tokenizer = PreTrainedTokenizerFast(
        vocab=vocab,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="<cls>",
        sep_token="<sep>"
    )
    return tokenizer

class GenericClassifier:
    """
    A flexible classification utility for Hugging Face tokenized datasets.

    Args:
        metadata_column (str): Column name containing original labels.
        training_args (dict, optional): kwargs for TrainingArguments.
        ray_config (dict, optional): Hyperparameter ranges for Ray Tune.
        freeze_layers (int, optional): Number of initial BERT layers to freeze.
        forward_batch_size (int, optional): Batch size for evaluation.
        nproc (int, optional): Number of processes for dataset operations.
    """
    default_training_args = {
        # evaluation & logging
        "eval_strategy": "steps",
        "eval_steps": 100,
        "logging_steps": 100,
        "save_steps":    100,
        # learning rate & schedule
        "learning_rate":    5e-5,
        "lr_scheduler_type":"cosine",
        'warmup_ratio': 0.1,
        # training length
        "num_train_epochs": 3,
        "weight_decay":     0.0,
        # precision & performance
        "fp16": True,         # or bf16 if supported
        "gradient_accumulation_steps": 2,
        "dataloader_pin_memory": True,
        # checkpoint
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
    }

    def __init__(
        self,
        metadata_column,
        model_mode: str = 'spot',            # "spot" or "extended"
        classifier_type: str = 'sequence',
        gene_class_dict=None,
        label_mapping=None,
        token_dictionary_file=None,
        filter_data=None,
        rare_threshold: float = 0.0,
        max_examples=None,
        max_examples_per_class=None,
        training_args=None,
        ray_config=None,
        freeze_layers: int = 0,
        forward_batch_size: int = 32,
        nproc: int = 4,
    ):
        self.metadata_column = metadata_column
        self.model_mode = model_mode
        self.classifier_type = classifier_type
        self.gene_class_dict = gene_class_dict
        self.label_mapping = label_mapping or {}
        self.filter_data = filter_data or {}
        self.rare_threshold = rare_threshold
        self.max_examples = max_examples
        self.max_examples_per_class = max_examples_per_class
        self.training_args = training_args if training_args is not None else self.default_training_args
        self.ray_config = ray_config
        self.freeze_layers = freeze_layers
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc

        if token_dictionary_file is None and classifier_type != 'sequence':
            raise Exception('If using gene classifier, please provide token dictionary file for mapping')
        self.token_dictionary_file = token_dictionary_file

    def prepare_data(self, input_data_file, output_directory, output_prefix):
        """
        Prepare dataset: filter, downsample, map metadata to 'label'/'labels', and save.

        - For sequence classification: creates a single 'label' per example.
        - For gene (token) classification:
            * In 'spot' mode: supervise all tokens.
            * In 'extended' mode: mask tokens after the midpoint of each sequence (len//2) via -100.
            (Midpoint masking is implemented inside cu.label_classes based on self.model_mode.)
        Returns:
            tuple(str, str): Paths to saved dataset and label mapping.
        """
        data = cu.load_and_filter(self.filter_data, self.nproc, input_data_file)
        os.makedirs(output_directory, exist_ok=True)

        # Will fill this below and persist to disk
        label_map = None

        if self.classifier_type == 'sequence':
            # remove rare, downsample/shuffle
            data = cu.remove_rare(data, self.rare_threshold, self.metadata_column, self.nproc)
            data = cu.downsample_and_shuffle(
                data,
                self.max_examples,
                self.max_examples_per_class,
                {"state_key": self.metadata_column}
            )

            # Infer or use provided label mapping
            if not self.label_mapping:
                unique = sorted(set(data[self.metadata_column]))
                self.label_mapping = {v: i for i, v in enumerate(unique)}
            label_map = self.label_mapping

            def _map(ex):
                ex['label'] = label_map[ex[self.metadata_column]]
                return ex

            data = data.map(_map, num_proc=self.nproc)

        else:  # token (gene) classification
            data, label_map = cu.label_classes(
                classifier='gene',
                data=data,
                class_dict=self.gene_class_dict,
                token_dict_path=self.token_dictionary_file,
                nproc=self.nproc,
                model_mode=self.model_mode,   # <-- key change: wires in midpoint masking when 'extended'
            )
            self.label_mapping = label_map

        # Persist mapping + dataset
        self.label_mapping = label_map
        map_path = Path(output_directory) / f"{output_prefix}_id_class_dict.pkl"
        with open(map_path, 'wb') as f:
            pickle.dump(label_map, f)

        ds_path = Path(output_directory) / f"{output_prefix}_labeled.dataset"
        data.save_to_disk(str(ds_path))
        logger.info("Data prep complete.")
        return str(ds_path), str(map_path)

    
    def _load_tokenizer(self, name_or_path):
        p = Path(name_or_path)
        if p.suffix in ['.pickle','.pkl'] and p.exists():
            return build_custom_tokenizer(str(p))
        try:
            return AutoTokenizer.from_pretrained(name_or_path)
        except:
            return PreTrainedTokenizerFast.from_pretrained(name_or_path)
        
    def train(
        self,
        model_checkpoint,
        dataset_path,
        output_directory,
        eval_dataset_path=None,
        n_trials: int = 0,
        test_size: float = 0.2,
        tokenizer_name_or_path: str = None
    ):
        """
        Train or tune a classifier.
        If eval_dataset_path is None, auto-split dataset_path using test_size.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        # Ray Tune path
        if self.ray_config and n_trials > 0:
            return self._ray_tune(
                model_checkpoint,
                dataset_path,
                output_directory,
                eval_dataset_path,
                n_trials,
                test_size
            )
        split_args = { 'test_size': test_size, 'seed': 42 }
        # Load primary dataset
        train_ds = load_from_disk(dataset_path)
        # Auto-split if no separate eval dataset
        if eval_dataset_path:
            eval_ds = load_from_disk(eval_dataset_path)
        else:
            ds = train_ds.train_test_split(**split_args)
            train_ds = ds['train']
            eval_ds = ds['test']

        os.makedirs(output_directory, exist_ok=True)
        # Prepare collator and tokenizer
        tok_src = tokenizer_name_or_path or model_checkpoint
        tokenizer = self._load_tokenizer(tok_src)
        if self.classifier_type == 'sequence':
            collator = DataCollatorWithPadding(
                tokenizer,
                padding='max_length',
                max_length=tokenizer.model_max_length
            )
            ModelClass = AutoModelForSequenceClassification
        else: # token classifier
            collator = DataCollatorForGeneClassification(
                tokenizer,
                padding='max_length',
                max_length=tokenizer.model_max_length)
            ModelClass = AutoModelForTokenClassification

        num_labels = len(self.label_mapping)
        assert num_labels > 0, 'No class labels found, did you run prepare_data?'
        cols = ['input_ids'] + (['label'] if self.classifier_type=='sequence' else ['labels'])
        train_ds.set_format(type='torch', columns=cols)
        eval_ds.set_format(type='torch', columns=cols)

        train_batch_size = int(self.forward_batch_size / 2) if (self.forward_batch_size /2) != 0.5 else 1 
        args = TrainingArguments(
            output_dir=output_directory, 
            **self.training_args,
            dataloader_num_workers=self.nproc,
            per_device_eval_batch_size=self.forward_batch_size,
            per_device_train_batch_size= train_batch_size)


        # Model init
        config = AutoConfig.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,          
        )
        model = ModelClass.from_pretrained(
            model_checkpoint,
            config=config,
            ignore_mismatched_sizes=True
        )
        model.to('cpu')

        if self.freeze_layers > 0 and hasattr(model, 'bert'):
            for l in model.bert.encoder.layer[:self.freeze_layers]:
                for p in l.parameters(): p.requires_grad=False

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=eu.compute_metrics,
            data_collator=collator
        )
        trainer.train()
        torch.cuda.empty_cache()
        tokenizer.save_pretrained(os.path.join(output_directory,'final_model'))
        trainer.save_model(os.path.join(output_directory, 'final_model'))
        pred = trainer.predict(eval_ds)
        with open(os.path.join(output_directory,'predictions.pkl'),'wb') as f:
            pickle.dump(pred,f)

        return trainer

    def _ray_tune(
        self,
        model_checkpoint,
        dataset_path,
        output_directory,
        eval_dataset_path,
        n_trials,
        test_size,
        tokenizer_name_or_path: str = None
    ):
        import os
        from pathlib import Path
        import ray
        from ray import tune
        from ray.tune.search.hyperopt import HyperOptSearch
        from ray.tune import CLIReporter
        from datasets import load_from_disk
        import torch
        from transformers import (
            AutoTokenizer,
            BertForSequenceClassification,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments
        )
        import evaluation_utils as eu

        #  Normalize and resolve paths 
        ckpt = Path(model_checkpoint)
        if ckpt.exists():
            model_checkpoint = str(ckpt.resolve())
            local_files_only = True
        else:
            local_files_only = False  

        outd = Path(output_directory)
        outd.mkdir(parents=True, exist_ok=True)
        output_directory = str(outd.resolve())

        #  Environment setup 
        GPU_COUNT = torch.cuda.device_count()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(GPU_COUNT))
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

        #  Initialize Ray 
        ray.shutdown()
        ray.init(ignore_reinit_error=True)

        #  Load & split data 
        split_args = { 'test_size': test_size, 'seed': 42 }
        train_ds = load_from_disk(dataset_path)
        if eval_dataset_path:
            eval_ds = load_from_disk(eval_dataset_path)
        else:
            ds = train_ds.train_test_split(**split_args)
            train_ds, eval_ds = ds["train"], ds["test"]

        #  HyperOpt search space 
        space = {}
        rc = self.ray_config or {}
        if "learning_rate" in rc:
            space["learning_rate"] = tune.loguniform(rc["learning_rate"][0], rc["learning_rate"][1])
        if "num_train_epochs" in rc:
            space["num_train_epochs"] = tune.choice(rc["num_train_epochs"])
        if "weight_decay" in rc:
            space["weight_decay"] = tune.uniform(rc["weight_decay"][0], rc["weight_decay"][1])
        if "lr_scheduler_type" in rc:
            space["lr_scheduler_type"] = tune.choice(rc["lr_scheduler_type"])
        if "warmup_steps" in rc:
            space["warmup_steps"] = tune.uniform(rc["warmup_steps"][0], rc["warmup_steps"][1])
        if "seed" in rc:
            space["seed"] = tune.uniform(rc["seed"][0], rc["seed"][1])

        #  Model init function 
        def model_init():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            config = AutoConfig.from_pretrained(
                model_checkpoint,
                num_labels=len(self.label_mapping),  # override the final head size to match our sequence (different classification task)
            )
            if self.classifier_type == "token": #gene classification
                # token classification head
                config.num_labels = len(self.gene_class_dict)  # or self.num_labels
                model = AutoModelForTokenClassification.from_pretrained(
                    model_checkpoint,
                    config=config,
                    ignore_mismatched_sizes=True,
                )
            else:
                # sequence classification head
                config.num_labels = len(self.gene_class_dict)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_checkpoint,
                    config=config,
                    ignore_mismatched_sizes=True,
                )
            if self.freeze_layers:
                for layer in model.bert.encoder.layer[: self.freeze_layers]:
                    for p in layer.parameters():
                        p.requires_grad = False
            return model.to(device)

        #  Tokenizer & collator
        tok_src = tokenizer_name_or_path or model_checkpoint
        tokenizer = self._load_tokenizer(tok_src)
        if self.classifier_type == 'token':
            from classifier_utils import DataCollatorForGeneClassification
            collator = DataCollatorForGeneClassification(
                tokenizer,
                padding="longest",
                max_length=tokenizer.model_max_length,
                label_pad_token_id=-100,
                return_tensors="pt",
            )
        else:
            from transformers import DataCollatorWithPadding
            collator = DataCollatorWithPadding(tokenizer=tokenizer)

        #  Reporter for CLI progress 
        reporter = CLIReporter(
            parameter_columns=list(space.keys()),
            metric_columns=["eval_loss", "eval_accuracy"],
            max_report_frequency=600,
            max_progress_rows=100,
            sort_by_metric=False,
            mode="min"
        )

        #  Base TrainingArguments 
        base_args = dict(self.training_args)
        base_args.update({
            "output_dir": output_directory,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "eval_steps": base_args.get("logging_steps", 500),
            "save_steps": base_args.get("logging_steps", 500),
            "load_best_model_at_end": True,
            "per_device_eval_batch_size": self.forward_batch_size
        })
        args = TrainingArguments(**base_args)

        #  Build Trainer & HP Search 
        trainer = Trainer(
            model_init=model_init,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=eu.compute_metrics,
        )

        hyperopt_search = HyperOptSearch(metric="eval_loss", mode="min")
        best = trainer.hyperparameter_search(
            hp_space=lambda _: space,
            backend="ray",
            search_alg=hyperopt_search,
            n_trials=n_trials,
            resources_per_trial={"cpu": self.nproc, "gpu": 1},
            progress_reporter=reporter,
            log_to_file=('tune_stdout.log','tune_stderr.log'),
            name='hp_search'
        )

        #Load an re-save the best model and tokenizer in seperate directory
        from ray.tune import ExperimentAnalysis
        analysis = ExperimentAnalysis(os.path.join(output_directory,'hp_search'))
        best_trial = analysis.get_best_trial(metric='eval_loss',mode='min')
        best_ckpt = analysis.get_best_checkpoint(best_trial,metric='eval_loss',mode='min')

        #load and re-save best model and tokenizer
        from transformers import BertForSequenceClassification, AutoTokenizer
        model_best = BertForSequenceClassification.from_pretrained(
            best_ckpt,
            num_labels=len(self.label_mapping),
            local_files_only=local_files_only
        )
        tokenizer_best = AutoTokenizer.from_pretrained(
            model_checkpoint,
            local_files_only=local_files_only
        )
        best_model_dir = os.path.join(output_directory,'best_model')
        os.makedirs(best_model_dir,exist_ok=True)
        model_best.save_pretrained(best_model_dir)
        tokenizer_best.save_pretrained(best_model_dir)

        ray.shutdown()
        return best


    def evaluate(
            self, 
            model_directory, 
            eval_dataset_path, 
            id_class_dict_file, 
            output_directory,
            tokenizer_name_or_path: str = None
            ):
        """
        Evaluate a trained classifier with proper padding.
        """
        ds = load_from_disk(eval_dataset_path)
        tok_src = tokenizer_name_or_path or (Path(model_directory)/'tokenizer')
        tokenizer = self._load_tokenizer(str(tok_src))
        collator = DataCollatorWithPadding(
            tokenizer,
            padding='max_length',
            max_length=tokenizer.model_max_length
        )
        collator = DataCollatorWithPadding(
            tokenizer,
            padding='max_length',
            max_length=tokenizer.model_max_length
        )
        cols = ['input_ids'] + (['label'] if self.classifier_type=='sequence' else ['labels'])
        ds.set_format(type='torch', columns=cols)

        model = AutoModelForSequenceClassification.from_pretrained(model_directory) if self.classifier_type == 'sequence' else AutoModelForTokenClassification.from_pretrained(model_directory)
        args = TrainingArguments(
            output_dir=output_directory,
            per_device_eval_batch_size=self.forward_batch_size,
            eval_strategy='no'
        )
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=ds,
            compute_metrics=eu.compute_metrics,
            data_collator=collator
        )
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def plot_confusion_matrix(self, conf_mat, output_directory, output_prefix, class_order):
        eu.plot_confusion_matrix({'conf_mat': conf_mat}, output_directory, output_prefix, class_order)

    def plot_predictions(self, predictions_file, id_class_dict_file, title, output_directory, output_prefix, class_order):
        eu.plot_predictions(predictions_file, id_class_dict_file, title, output_directory, output_prefix, class_order)
