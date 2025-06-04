"""
stFormer Pretrainer module with modular collator and trainer classes.

Example usage:
    from pretrainer_refactored import STFormerPretrainer, load_example_lengths
    from transformers import BertConfig, BertForMaskedLM, TrainingArguments
    from datasets import load_from_disk

    # Load dataset and token dictionary
    dataset = load_from_disk("path/to/dataset")
    token_dict = load_example_lengths("path/to/token_dictionary.pkl")
    lengths_file = "path/to/example_lengths.pkl"

    # Build model and training args
    config = BertConfig(vocab_size=len(token_dict), pad_token_id=token_dict.get('<pad>'))
    model = BertForMaskedLM(config).train()
    training_args = TrainingArguments(
        output_dir="output/models",
        per_device_train_batch_size=8,
        num_train_epochs=2
    )

    # Initialize trainer and start training
    trainer = STFormerPretrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        token_dictionary=token_dict,
        example_lengths_file=lengths_file
    )
    trainer.train()

    # Save trained model
    trainer.save_model("output/models")
"""
import collections
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Callable

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    SpecialTokensMixin,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import RandomSampler
from transformers.trainer_utils import has_length
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Constants
VERY_LARGE_INT = int(1e30)
LARGE_INT = int(1e20)


def load_example_lengths(file_path: Union[str, Path]) -> List[int]:
    """
    Load example lengths for length-based sampling from a pickle file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


class STFormerPreCollator(SpecialTokensMixin):
    """
    Data collator for single-cell transcriptomics, providing padding and token conversion.
    """
    def __init__(self, token_dict: Dict[str, int]):
        super().__init__(mask_token="<mask>", pad_token="<pad>")
        self.token_dict = token_dict
        self.inv_token_dict = {v:k for k,v in token_dict.items()}
        self.padding_side = 'right'
        self.model_input_names = ['input_ids']
        self.pad_token_id = token_dict[self.pad_token]
        self.mask_token_id = token_dict[self.mask_token]

    def __len__(self):
        return len(self.token_dict)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token(s) to their numerical IDs."""
        if isinstance(tokens, str):
            return self.token_dict.get(tokens)
        return [self.token_dict.get(t) for t in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Inverse of convert_tokens_to_ids, required by DataCollatorForLanguageModeling.
        """
        if isinstance(ids, int):
            return self.inv_token_dict.get(ids, None)
        return [self.inv_token_dict.get(i, None) for i in ids]

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        """
        Prepare a batch for masked language modeling.
        """
        batch = BatchEncoding({'input_ids': [ex['input_ids'] for ex in examples]})
        # MLM masking handled by HF's collator
        return batch
    
    def pad(
        self,
        encoded_inputs: Union[List[Dict[str, List[int]]], BatchEncoding],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        # 1) Extract raw input_id lists
        if isinstance(encoded_inputs, list):
            all_ids = [f["input_ids"] for f in encoded_inputs]
        else:
            all_ids = encoded_inputs.get("input_ids")

        # 2) Determine target length
        target_len = max_length or max(len(seq) for seq in all_ids)
        pad_id = self.pad_token_id

        # 3) Pad input_ids
        padded_ids = [
            seq + [pad_id] * (target_len - len(seq))
            for seq in all_ids
        ]

        # 4) Build attention_mask (1 for real tokens, 0 for pads)
        mask = [
            [1] * len(seq) + [0] * (target_len - len(seq))
            for seq in all_ids
        ]

        # 5) Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            padded_ids = torch.tensor(padded_ids, dtype=torch.long)
            mask       = torch.tensor(mask, dtype=torch.long)

        # 6) Return both input_ids and attention_mask
        return BatchEncoding({
            "input_ids": padded_ids,
            "attention_mask": mask
        }, tensor_type=return_tensors)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Mark pad and mask tokens as “special” so they won’t get masked again.
        """
        mask0 = [1 if t in {self.pad_token_id, self.mask_token_id} else 0
                 for t in token_ids_0]
        if token_ids_1 is not None:
            mask1 = [1 if t in {self.pad_token_id, self.mask_token_id} else 0
                     for t in token_ids_1]
            return mask0 + mask1
        return mask0



class STFormerPretrainer(Trainer):
    """
    Custom Trainer for masked pretraining on single-cell data.
    """

    def __init__(
        self,
        args: TrainingArguments,
        train_dataset: Dataset,
        token_dictionary: Dict[str, int],
        example_lengths_file: Union[str, Path],
        model: Optional[torch.nn.Module] = None,
        model_init: Optional[Callable[[], torch.nn.Module]] = None,
        mlm_probability: float = 0.15,
    ):
        if model is None and model_init is None:
            raise ValueError("You must provide either `model` or `model_init`.")

        self.lengths = load_example_lengths(example_lengths_file)

        # Build custom tokenizer/collator
        collator = DataCollatorForLanguageModeling(
            tokenizer=STFormerPreCollator(token_dictionary),
            mlm=True,
            mlm_probability=mlm_probability,
        )

        # Store model or model_init
        self._model_init = model_init 
        super().__init__(
            model=model,
            model_init=model_init,
            args=args,
            train_dataset=train_dataset,
            data_collator=collator,
        )

    def get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Return a sampler, grouping by length if requested.
        """
        if not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            return LengthGroupedSampler(
                dataset=self.train_dataset,
                lengths=self.lengths,
                batch_size=self.args.train_batch_size,
                model_input_name=self.tokenizer.model_input_names[0],
            )
        return RandomSampler(self.train_dataset)

