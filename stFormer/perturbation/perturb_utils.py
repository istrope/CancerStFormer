import itertools as it
import logging
import pickle
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
)

sns.set()
logger = logging.getLogger(__name__)


# load data and filter by defined criteria
def load_and_filter(filter_data, nproc, input_data_file):
    data = load_from_disk(input_data_file)
    if filter_data is not None:
        data = filter_by_dict(data, filter_data, nproc)
    return data


def filter_by_dict(data, filter_data, nproc):
    for key, value in filter_data.items():
        data = data.filter(lambda example: example[key] in value, num_proc=nproc)
    if len(data) == 0:
        logger.error("No cells remain after filtering. Check filtering criteria.")
        raise
    return data


def filter_data_by_tokens(filtered_input_data, tokens, nproc):
    def if_has_tokens(example):
        return len(set(example["input_ids"]).intersection(tokens)) == len(tokens)
    filtered_input_data = filtered_input_data.filter(if_has_tokens, num_proc=nproc)
    return filtered_input_data


def logging_filtered_data_len(filtered_input_data, filtered_tokens_categ):
    if len(filtered_input_data) == 0:
        logger.error(f"No cells in dataset contain {filtered_tokens_categ}.")
        raise
    else: logger.info(f"# cells with {filtered_tokens_categ}: {len(filtered_input_data)}")

def filter_data_by_tokens_and_log(
    filtered_input_data, tokens, nproc, filtered_tokens_categ
):
    filtered_input_data = filter_data_by_tokens(filtered_input_data, tokens, nproc)
    logging_filtered_data_len(filtered_input_data, filtered_tokens_categ)
    return filtered_input_data


def filter_data_by_start_state(filtered_input_data, cell_states_to_model, nproc):
    # confirm that start state is valid to prevent futile filtering
    state_key = cell_states_to_model["state_key"]
    state_values = filtered_input_data[state_key]
    start_state = cell_states_to_model["start_state"]
    if start_state not in state_values:
        logger.error(
            f"Start state {start_state} is not present "
            f"in the dataset's {state_key} attribute."
        )
        raise
    def filter_for_origin(example):
        return example[state_key] in [start_state]
    filtered_input_data = filtered_input_data.filter(filter_for_origin, num_proc=nproc)
    return filtered_input_data


def slice_by_inds_to_perturb(dataset, inds):
    start, end = inds["start"], inds["end"]
    n = len(dataset)
    if start >= n:
        logger.error(f"Start index ({start}) out of range (dataset size {n}).")
        raise IndexError(f"cell_inds_to_perturb['start'] ({start}) >= dataset length ({n})")
    if end > n:
        logger.warning(f"End index ({end}) exceeds dataset length ({n}); clipping to {n}.")
        end = n
    return dataset.select(range(start, end))


def load_model(model_type, num_classes, model_directory, mode):
    output_hidden_states = (mode == "eval")
    cls_map = {
        "Pretrained": (BertForMaskedLM,{}),
        "GeneClassifier": (BertForTokenClassification,{"num_labels": num_classes}),
        "CellClassifier": (BertForSequenceClassification,{"num_labels": num_classes}),
    }
    ModelClass, extra_args = cls_map.get(model_type, (None, None))
    if ModelClass is None:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    model = ModelClass.from_pretrained(
        model_directory,
        output_hidden_states=output_hidden_states,
        output_attentions=False,
        **extra_args
    )
    if mode == "eval":
        model.eval()
    return model.to("cuda")

def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [int(name.split("layer.")[1].split(".")[0])]
    return int(max(layer_nums)) + 1
def get_model_emb_dims(model):
    return model.config.hidden_size
def get_model_input_size(model):
    return model.config.max_position_embeddings
def flatten_list(megalist):
    return [item for sublist in megalist for item in sublist]
def measure_length(example):
    example["length"] = len(example["input_ids"])
    return example


def downsample_and_sort(data, max_ncells):
    num_cells = len(data)
    if max_ncells is not None:
        if num_cells > max_ncells:
            data = data.shuffle(seed=42)
            num_cells = max_ncells
    data_subset = data.select([i for i in range(num_cells)])
    data_sorted = data_subset.sort("length", reverse=True)
    return data_sorted

def stratified_downsample_and_sort(data, max_ncells, stratify_by):
    num_cells = len(data)
    if max_ncells is not None and num_cells > max_ncells:
        unique_labels = data.unique(stratify_by)
        per_group_count = max_ncells // len(unique_labels)
        sampled_indices = []
        for label in unique_labels:
            label_indices = [i for i, val in enumerate(data[stratify_by]) if val == label]
            np.random.seed(42)
            shuffled_indices = np.random.permutation(label_indices)
            sampled_indices.extend(shuffled_indices[:per_group_count])
        sampled_indices = np.random.permutation(sampled_indices)[:max_ncells]
        data = data.select(sampled_indices)
    data_sorted = data.sort("length", reverse=True)
    return data_sorted

def get_possible_states(cell_states_to_model):
    possible_states = []
    for key in ["start_state", "goal_state"]:
        possible_states += [cell_states_to_model[key]]
    possible_states += cell_states_to_model.get("alt_states", [])
    return possible_states


def forward_pass_single_cell(model, example_cell, layer_to_quant):
    example_cell.set_format(type="torch")
    input_data = example_cell["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_data.to("cuda"))
    emb = torch.squeeze(outputs.hidden_states[layer_to_quant])
    del outputs
    return emb


def perturb_emb_by_index(emb, indices):
    mask = torch.ones(emb.numel(), dtype=torch.bool)
    mask[indices] = False
    return emb[mask]


def delete_indices(example):
    indices = example["perturb_index"]
    if any(isinstance(el, list) for el in indices):
        indices = flatten_list(indices)
    for index in sorted(indices, reverse=True):
        del example["input_ids"][index]

    example["length"] = len(example["input_ids"])
    return example


# for genes_to_perturb = "all" where only genes within cell are overexpressed
def overexpress_indices(example):
    indices = example["perturb_index"]
    if any(isinstance(el, list) for el in indices):
        indices = flatten_list(indices)
    for index in sorted(indices, reverse=True):
        example["input_ids"].insert(0, example["input_ids"].pop(index))

    example["length"] = len(example["input_ids"])
    return example

def check_batch_inputs(orig_batch,pert_batch,start,end,perturb_type,):
    for i in range(len(orig_batch["input_ids"])):
        orig_tokens = orig_batch["input_ids"][i]
        pert_tokens = pert_batch["input_ids"][i]

        # Either full match or (for delete perturbations) perturbed should be a subset
        if perturb_type == "delete":
            if not set(pert_tokens).issubset(set(orig_tokens)):
                print(f"Mismatch in DELETE at index {start + i}")
                print("Original:", orig_tokens)
                print("Perturbed:", pert_tokens)
                raise ValueError("Perturbed tokens are not a subset of original tokens.")
        else:  # Overexpress
            # Overexpress should preserve the original tokens somewhere after the perturb
            missing = [t for t in orig_tokens if t not in pert_tokens]
            if missing:
                print(f"Mismatch in OVEREXPRESS at index {start + i}")
                print("Original:", orig_tokens)
                print("Perturbed:", pert_tokens)
                print("Missing tokens from perturbed:", missing)
                raise ValueError("Original tokens missing from perturbed batch.")
            
def check_embedding_shapes(pert_emb,original_emb,orig_batch,pert_batch,start,end):
    if pert_emb.shape[1] != original_emb.shape[1]:
        print("---- Shape mismatch detected ----")
        print(f"pert_emb shape: {pert_emb.shape}")
        print(f"original_emb shape: {original_emb.shape}")
        print(f"Batch indices {start} to {end}")
        print(f"Original lengths: {orig_batch['length']}")
        print(f"Perturbed lengths: {pert_batch['length']}")
        for i in range(len(orig_batch["input_ids"])):
            print(f"Sample {i} orig tokens ({len(orig_batch['input_ids'][i])}): {orig_batch['input_ids'][i]}")
            print(f"Sample {i} pert tokens ({len(pert_batch['input_ids'][i])}): {pert_batch['input_ids'][i]}")
        raise ValueError("Embedding dimensions do not match. Aborting.")

def align_perturbed_embeddings(full_pert, orig_batch, pert_batch, tokens_to_perturb):
    """
    Align perturbed embeddings to original embeddings by removing the first occurrence 
    of each overexpressed token, if sequence length increased. Pads to batch max length.
    """
    aligned_pert_emb = []

    for i in range(full_pert.shape[0]):
        orig_tokens = orig_batch["input_ids"][i]
        pert_tokens = pert_batch["input_ids"][i]
        pert_emb_sample = full_pert[i]

        orig_len = orig_batch["length"][i]
        pert_len = pert_batch["length"][i]

        pert_tokens_mod = list(pert_tokens)  # Explicitly create new list

        length_increase = pert_len - orig_len

        removed = 0
        for token in tokens_to_perturb:
            if length_increase > 0 and token in pert_tokens_mod:
                index_to_remove = pert_tokens_mod.index(token)

                # Sanity check before removal
                if index_to_remove >= pert_emb_sample.shape[0]:
                    raise ValueError(
                        f"Sample {i}: Token index {index_to_remove} exceeds embedding length {pert_emb_sample.shape[0]}"
                    )

                pert_tokens_mod.pop(index_to_remove)
                pert_emb_sample = torch.cat((
                    pert_emb_sample[:index_to_remove, :],
                    pert_emb_sample[index_to_remove+1:, :]
                ), dim=0)

                length_increase -= 1
                removed += 1

        # Sanity check after all removals
        if len(pert_tokens_mod) != pert_emb_sample.shape[0]:
            raise ValueError(
                f"Sample {i}: Token list length {len(pert_tokens_mod)} does not match embedding length {pert_emb_sample.shape[0]}"
            )

        # Truncate or pad to match original length
        if len(pert_tokens_mod) > orig_len:
            pert_emb_sample = pert_emb_sample[:orig_len, :]
        elif len(pert_tokens_mod) < orig_len:
            pad_size = orig_len - len(pert_tokens_mod)
            padding = pert_emb_sample.new_zeros((pad_size, pert_emb_sample.shape[1]))
            pert_emb_sample = torch.cat([pert_emb_sample, padding], dim=0)

        aligned_pert_emb.append(pert_emb_sample)

    return torch.stack(aligned_pert_emb)


# for genes_to_perturb = list of genes to overexpress that are not necessarily expressed in cell
def overexpress_tokens(example, max_len):
    # -100 indicates tokens to overexpress are not present in rank value encoding
    if example["perturb_index"] != [-100]:
        example = delete_indices(example)
    [
        example["input_ids"].insert(0, token)
        for token in example["tokens_to_perturb"][::-1]
    ]

    # truncate to max input size, must also truncate original emb to be comparable
    if len(example["input_ids"]) > max_len:
        example["input_ids"] = example["input_ids"][0:max_len]

    example["length"] = len(example["input_ids"])
    return example

def calc_n_overflow(max_len, example_len, tokens_to_perturb, indices_to_perturb):
    n_to_add = len(tokens_to_perturb) - len(indices_to_perturb)
    n_overflow = example_len + n_to_add - max_len
    return n_overflow


def truncate_by_n_overflow(example):
    new_max_len = example["length"] - example["n_overflow"]
    example["input_ids"] = example["input_ids"][0:new_max_len]
    example["length"] = len(example["input_ids"])
    return example


def remove_indices_from_emb(emb, indices_to_remove, gene_dim):
    # indices_to_remove is list of indices to remove
    indices_to_keep = [
        i for i in range(emb.size()[gene_dim]) if i not in indices_to_remove
    ]
    num_dims = emb.dim()
    emb_slice = [
        slice(None) if dim != gene_dim else indices_to_keep for dim in range(num_dims)
    ]
    sliced_emb = emb[emb_slice]
    return sliced_emb


def remove_indices_from_emb_batch(emb_batch, list_of_indices_to_remove, gene_dim):
    output_batch_list = [
        remove_indices_from_emb(emb_batch[i, :, :], idxes, gene_dim - 1)
        for i, idxes in enumerate(list_of_indices_to_remove)
    ]
    # add padding given genes are sometimes added that are or are not in original cell
    batch_max = max([emb.size()[gene_dim - 1] for emb in output_batch_list])
    output_batch_list_padded = [
        pad_xd_tensor(emb, 0.000, batch_max, gene_dim - 1) for emb in output_batch_list
    ]
    return torch.stack(output_batch_list_padded)


# removes perturbed indices
def remove_perturbed_indices_set(
    emb,
    perturb_type: str,
    indices_to_perturb: List[List],
    tokens_to_perturb: List[List],
    original_lengths: List[int],
    input_ids=None,
):
    if perturb_type == "overexpress":
        num_perturbed = len(tokens_to_perturb)
        if num_perturbed == 1:
            indices_to_perturb_orig = [
                idx if idx != [-100] else [None] for idx in indices_to_perturb
            ]
            if all(v is [None] for v in indices_to_perturb_orig):
                return emb
        else:
            indices_to_perturb_orig = []
            for idx_list in indices_to_perturb:
                indices_to_perturb_orig.append([idx if idx != [-100] else [None] for idx in idx_list])
    else:
        indices_to_perturb_orig = indices_to_perturb
    emb = remove_indices_from_emb_batch(emb, indices_to_perturb_orig, gene_dim=1)

    return emb


def make_perturbation_batch(
    example_cell, perturb_type, tokens_to_perturb, anchor_token, combo_lvl, num_proc
) -> tuple[Dataset, List[int]]:
    if combo_lvl == 0 and tokens_to_perturb == "all":
        if perturb_type in ["overexpress", "activate"]:
            range_start = 1
        elif perturb_type in ["delete", "inhibit"]:
            range_start = 0
        indices_to_perturb = [
            [i] for i in range(range_start, example_cell["length"][0])
        ]
    # elif combo_lvl > 0 and anchor_token is None:
    ## to implement
    elif combo_lvl > 0 and (anchor_token is not None):
        example_input_ids = example_cell["input_ids"][0]
        anchor_index = example_input_ids.index(anchor_token[0])
        indices_to_perturb = [
            sorted([anchor_index, i]) if i != anchor_index else None
            for i in range(example_cell["length"][0])
        ]
        indices_to_perturb = [item for item in indices_to_perturb if item is not None]
    else:
        example_input_ids = example_cell["input_ids"][0]
        indices_to_perturb = [
            [example_input_ids.index(token)] if token in example_input_ids else None
            for token in tokens_to_perturb
        ]
        indices_to_perturb = [item for item in indices_to_perturb if item is not None]

    # create all permutations of combo_lvl of modifiers from tokens_to_perturb
    if combo_lvl > 0 and (anchor_token is None):
        if tokens_to_perturb != "all":
            if len(tokens_to_perturb) == combo_lvl + 1:
                indices_to_perturb = [
                    list(x) for x in it.combinations(indices_to_perturb, combo_lvl + 1)
                ]
        else:
            all_indices = [[i] for i in range(example_cell["length"][0])]
            all_indices = [
                index for index in all_indices if index not in indices_to_perturb
            ]
            indices_to_perturb = [
                [[j for i in indices_to_perturb for j in i], x] for x in all_indices
            ]

    length = len(indices_to_perturb)
    perturbation_dataset = Dataset.from_dict(
        {
            "input_ids": example_cell["input_ids"] * length,
            "perturb_index": indices_to_perturb,
        }
    )

    if length < 400:
        num_proc_i = 1
    else:
        num_proc_i = num_proc

    if perturb_type == "delete":
        perturbation_dataset = perturbation_dataset.map(
            delete_indices, num_proc=num_proc_i
        )
    elif perturb_type == "overexpress":
        perturbation_dataset = perturbation_dataset.map(
            overexpress_indices, num_proc=num_proc_i
        )

    perturbation_dataset = perturbation_dataset.map(measure_length, num_proc=num_proc_i)

    return perturbation_dataset, indices_to_perturb


# perturbed cell emb removing the activated/overexpressed/inhibited gene emb
# so that only non-perturbed gene embeddings are compared to each other
# in original or perturbed context
def make_comparison_batch(original_emb_batch, indices_to_perturb, perturb_group):
    all_embs_list = []

    # if making comparison batch for multiple perturbations in single cell
    if perturb_group is False:
        # squeeze if single cell
        if original_emb_batch.ndim == 3 and original_emb_batch.size()[0] == 1:
            original_emb_batch = torch.squeeze(original_emb_batch)
        original_emb_list = [original_emb_batch] * len(indices_to_perturb)
    # if making comparison batch for single perturbation in multiple cells
    elif perturb_group is True:
        original_emb_list = original_emb_batch

    for original_emb, indices in zip(original_emb_list, indices_to_perturb):
        if indices == [-100]:
            all_embs_list += [original_emb[:]]
            continue

        emb_list = []
        start = 0
        if any(isinstance(el, list) for el in indices):
            indices = flatten_list(indices)

        # removes indices that were perturbed from the original embedding
        for i in sorted(indices):
            emb_list += [original_emb[start:i]]
            start = i + 1

        emb_list += [original_emb[start:]]
        all_embs_list += [torch.cat(emb_list)]

    len_set = set([emb.size()[0] for emb in all_embs_list])
    if len(len_set) > 1:
        max_len = max(len_set)
        all_embs_list = [pad_2d_tensor(emb, None, max_len, 0) for emb in all_embs_list]
    return torch.stack(all_embs_list)


def pad_list(input_ids, pad_token_id, max_len):
    input_ids = np.pad(
        input_ids,
        (0, max_len - len(input_ids)),
        mode="constant",
        constant_values=pad_token_id,
    )
    return input_ids


def pad_xd_tensor(tensor, pad_token_id, max_len, dim):
    padding_length = max_len - tensor.size()[dim]
    # Construct a padding configuration where all padding values are 0, except for the padding dimension
    # 2 * number of dimensions (padding before and after for every dimension)
    pad_config = [0] * 2 * tensor.dim()
    # Set the padding after the desired dimension to the calculated padding length
    pad_config[-2 * dim - 1] = padding_length
    return torch.nn.functional.pad(
        tensor, pad=pad_config, mode="constant", value=pad_token_id
    )


def pad_tensor(tensor, pad_token_id, max_len):
    tensor = torch.nn.functional.pad(
        tensor, pad=(0, max_len - tensor.numel()), mode="constant", value=pad_token_id
    )

    return tensor


def pad_2d_tensor(tensor, pad_token_id, max_len, dim):
    if dim == 0:
        pad = (0, 0, 0, max_len - tensor.size()[dim])
    elif dim == 1:
        pad = (0, max_len - tensor.size()[dim], 0, 0)
    tensor = torch.nn.functional.pad(
        tensor, pad=pad, mode="constant", value=pad_token_id
    )
    return tensor


def pad_3d_tensor(tensor, pad_token_id, max_len, dim):
    if dim == 0:
        raise Exception("dim 0 usually does not need to be padded.")
    if dim == 1:
        pad = (0, 0, 0, max_len - tensor.size()[dim])
    elif dim == 2:
        pad = (0, max_len - tensor.size()[dim], 0, 0)
    tensor = torch.nn.functional.pad(
        tensor, pad=pad, mode="constant", value=pad_token_id
    )
    return tensor


def pad_or_truncate_encoding(encoding, pad_token_id, max_len):
    if isinstance(encoding, torch.Tensor):
        encoding_len = encoding.size()[0]
    elif isinstance(encoding, list):
        encoding_len = len(encoding)
    if encoding_len > max_len:
        encoding = encoding[0:max_len]
    elif encoding_len < max_len:
        if isinstance(encoding, torch.Tensor):
            encoding = pad_tensor(encoding, pad_token_id, max_len)
        elif isinstance(encoding, list):
            encoding = pad_list(encoding, pad_token_id, max_len)
    return encoding


# pad list of tensors and convert to tensor
def pad_tensor_list(
    tensor_list,
    dynamic_or_constant,
    pad_token_id,
    model_input_size,
    dim=None,
    padding_func=None,
):
    # determine maximum tensor length
    if dynamic_or_constant == "dynamic":
        max_len = max([tensor.squeeze().numel() for tensor in tensor_list])
    elif isinstance(dynamic_or_constant, int):
        max_len = dynamic_or_constant
    else:
        max_len = model_input_size
        logger.warning(
            "If padding style is constant, must provide integer value. "
            f"Setting padding to max input size {model_input_size}."
        )

    # pad all tensors to maximum length
    if dim is None:
        tensor_list = [
            pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list
        ]
    else:
        tensor_list = [
            padding_func(tensor, pad_token_id, max_len, dim) for tensor in tensor_list
        ]
    # return stacked tensors
    if padding_func != pad_3d_tensor:
        return torch.stack(tensor_list)
    else:
        return torch.cat(tensor_list, 0)


def gen_attention_mask(minibatch_encoding, max_len=None):
    if max_len is None:
        max_len = max(minibatch_encoding["length"])
    original_lens = minibatch_encoding["length"]
    attention_mask = [
        [1] * original_len + [0] * (max_len - original_len)
        if original_len <= max_len
        else [1] * max_len
        for original_len in original_lens
    ]
    return torch.tensor(attention_mask, device="cuda")


# get cell embeddings excluding padding
def mean_nonpadding_embs(embs, original_lens, dim=1):
    # create a mask tensor based on padding lengths
    mask = torch.arange(embs.size(dim), device=embs.device) < original_lens.unsqueeze(1)
    if embs.dim() == 3:
        # fill the masked positions in embs with zeros
        masked_embs = embs.masked_fill(~mask.unsqueeze(2), 0.0)

        # compute the mean across the non-padding dimensions
        mean_embs = masked_embs.sum(dim) / original_lens.view(-1, 1).float()

    elif embs.dim() == 2:
        masked_embs = embs.masked_fill(~mask, 0.0)
        mean_embs = masked_embs.sum(dim) / original_lens.float()
    return mean_embs


# get cell embeddings when there is no padding
def compute_nonpadded_cell_embedding(embs, cell_emb_style):
    if cell_emb_style == "mean_pool":
        return torch.mean(embs, dim=embs.ndim - 2)


# quantify shifts for a set of genes
def quant_cos_sims(
    perturbation_emb,
    original_emb,
    cell_states_to_model,
    state_embs_dict,
    emb_mode="gene",
):
    if emb_mode == "gene":
        cos = torch.nn.CosineSimilarity(dim=2)
    elif emb_mode == "cell":
        cos = torch.nn.CosineSimilarity(dim=1)

    # if emb_mode == "gene", can only calculate gene cos sims
    # against original cell anyways
    if cell_states_to_model is None or emb_mode == "gene":
        cos_sims = cos(perturbation_emb, original_emb).to("cuda")
    elif cell_states_to_model is not None and emb_mode == "cell":
        possible_states = get_possible_states(cell_states_to_model)
        cos_sims = dict(zip(possible_states, [[] for _ in range(len(possible_states))]))
        for state in possible_states:
            cos_sims[state] = cos_sim_shift(
                original_emb,
                perturbation_emb,
                state_embs_dict[state].to("cuda"),  # required to move to cuda here
                cos,
            )

    return cos_sims


# calculate cos sim shift of perturbation with respect to origin and alternative cell
def cos_sim_shift(original_emb, perturbed_emb, end_emb, cos):
    origin_v_end = cos(original_emb, end_emb)
    perturb_v_end = cos(perturbed_emb, end_emb)

    return perturb_v_end - origin_v_end


def concatenate_cos_sims(cos_sims):
    if isinstance(cos_sims, list):
        return torch.cat(cos_sims)
    else:
        for state in cos_sims.keys():
            cos_sims[state] = torch.cat(cos_sims[state])
        return cos_sims


def write_perturbation_dictionary(cos_sims_dict: defaultdict, output_path_prefix: str):
    with open(f"{output_path_prefix}_raw.pickle", "wb") as fp:
        pickle.dump(cos_sims_dict, fp)


def tensor_list_to_pd(tensor_list):
    tensor = torch.cat(tensor_list).cpu().numpy()
    df = pd.DataFrame(tensor)
    return df


def validate_cell_states_to_model(cell_states_to_model):
    if cell_states_to_model is not None:
        if len(cell_states_to_model.items()) == 1:
            logger.warning(
                "The single value dictionary for cell_states_to_model will be "
                "replaced with a dictionary with named keys for start, goal, and alternate states. "
                "Please specify state_key, start_state, goal_state, and alt_states "
                "in the cell_states_to_model dictionary for future use. "
                "For example, cell_states_to_model={"
                "'state_key': 'disease', "
                "'start_state': 'dcm', "
                "'goal_state': 'nf', "
                "'alt_states': ['hcm', 'other1', 'other2']}"
            )
            for key, value in cell_states_to_model.items():
                if (len(value) == 3) and isinstance(value, tuple):
                    if (
                        isinstance(value[0], list)
                        and isinstance(value[1], list)
                        and isinstance(value[2], list)
                    ):
                        if len(value[0]) == 1 and len(value[1]) == 1:
                            all_values = value[0] + value[1] + value[2]
                            if len(all_values) == len(set(all_values)):
                                continue
            # reformat to the new named key format
            state_values = flatten_list(list(cell_states_to_model.values()))

            cell_states_to_model = {
                "state_key": list(cell_states_to_model.keys())[0],
                "start_state": state_values[0][0],
                "goal_state": state_values[1][0],
                "alt_states": state_values[2:][0],
            }
        elif set(cell_states_to_model.keys()).issuperset(
            {"state_key", "start_state", "goal_state"}
        ):
            if (
                (cell_states_to_model["state_key"] is None)
                or (cell_states_to_model["start_state"] is None)
                or (cell_states_to_model["goal_state"] is None)
            ):
                logger.error(
                    "Please specify 'state_key', 'start_state', and 'goal_state' in cell_states_to_model."
                )
                raise

            if (
                cell_states_to_model["start_state"]
                == cell_states_to_model["goal_state"]
            ):
                logger.error("All states must be unique.")
                raise

            if "alt_states" in set(cell_states_to_model.keys()):
                if cell_states_to_model["alt_states"] is not None:
                    if not isinstance(cell_states_to_model["alt_states"], list):
                        logger.error(
                            "cell_states_to_model['alt_states'] must be a list (even if it is one element)."
                        )
                        raise
                    if len(cell_states_to_model["alt_states"]) != len(
                        set(cell_states_to_model["alt_states"])
                    ):
                        logger.error("All states must be unique.")
                        raise
            else:
                cell_states_to_model["alt_states"] = []

        else:
            logger.error(
                "cell_states_to_model must only have the following four keys: "
                "'state_key', 'start_state', 'goal_state', 'alt_states'."
                "For example, cell_states_to_model={"
                "'state_key': 'disease', "
                "'start_state': 'dcm', "
                "'goal_state': 'nf', "
                "'alt_states': ['hcm', 'other1', 'other2']}"
            )
            raise

class GeneIdHandler:
    def __init__(
            self, 
            raise_errors=False,
            token_dictionary_file = None,
            gene_id_name_dict = None):
        
        def invert_dict(dict_obj):
            return {v:k for k,v in dict_obj.items()}
        
        self.raise_errors = raise_errors
        
        with open(token_dictionary_file, 'rb') as f:
            self.gene_token_dict = pickle.load(f)
            self.token_gene_dict = invert_dict(self.gene_token_dict)

        with open(gene_id_name_dict, 'rb') as f:
            self.gene_id_dict = pickle.load(f)
            self.id_gene_dict = invert_dict(self.id_gene_dict)
            
    def ens_to_token(self, ens_id):
        if not self.raise_errors:
            return self.gene_token_dict.get(ens_id, ens_id)
        else:
            return self.gene_token_dict[ens_id]
    
    def token_to_ens(self, token):
        if not self.raise_errors:
            return self.token_gene_dict.get(token, token)
        else:
            return self.token_gene_dict[token]

    def ens_to_symbol(self, ens_id):
        if not self.raise_errors:
            return self.gene_id_dict.get(ens_id, ens_id)
        else:
            return self.gene_id_dict[ens_id]
    
    def symbol_to_ens(self, symbol):
        if not self.raise_errors:
            return self.id_gene_dict.get(symbol, symbol)
        else:
            return self.id_gene_dict[symbol]
    
    def token_to_symbol(self, token):
        return self.ens_to_symbol(self.token_to_ens(token))
    
    def symbol_to_token(self, symbol):
        return self.ens_to_token(self.symbol_to_ens(symbol))