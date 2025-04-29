
# imports
import logging
import pandas as pd
import torch
from tqdm.auto import trange

import stFormer.perturbation.geneformer_perturber_utils as pu
logger = logging.getLogger(__name__)

# extract embeddings
def get_embs(
    model,
    filtered_input_data,
    emb_mode,
    layer_to_quant,
    pad_token_id,
    forward_batch_size,
    token_gene_dict,
    special_token=False,
    summary_stat=None,
    silent=False,
):
    model_input_size = pu.get_model_input_size(model)
    total_batch_length = len(filtered_input_data)
    
    if summary_stat is None:
        embs_list = []
    elif summary_stat is not None:
        # get # of emb dims
        emb_dims = pu.get_model_emb_dims(model)
        if emb_mode == "cell":
            # initiate tdigests for # of emb dims
            embs_tdigests = [TDigest() for _ in range(emb_dims)]
        if emb_mode == "gene":
            gene_set = list(
                {
                    element
                    for sublist in filtered_input_data["input_ids"]
                    for element in sublist
                }
            )
            # initiate dict with genes as keys and tdigests for # of emb dims as values
            embs_tdigests_dict = {
                k: [TDigest() for _ in range(emb_dims)] for k in gene_set
            }

    # Check if CLS and EOS token is present in the token dictionary
    cls_present = any("<cls>" in value for value in token_gene_dict.values())
    eos_present = any("<eos>" in value for value in token_gene_dict.values())
    if emb_mode == "cls":
        assert cls_present, "<cls> token missing in token dictionary"
        # Check to make sure that the first token of the filtered input data is cls token
        gene_token_dict = {v:k for k,v in token_gene_dict.items()}
        cls_token_id = gene_token_dict["<cls>"]
        assert filtered_input_data["input_ids"][0][0] == cls_token_id, "First token is not <cls> token value"
    elif emb_mode == "cell":
        if cls_present:
            logger.warning("CLS token present in token dictionary, excluding from average.")    
        if eos_present:
            logger.warning("EOS token present in token dictionary, excluding from average.")
            
    overall_max_len = 0
        
    for i in trange(0, total_batch_length, forward_batch_size, leave=(not silent)):
        max_range = min(i + forward_batch_size, total_batch_length)

        minibatch = filtered_input_data.select([i for i in range(i, max_range)])

        max_len = int(max(minibatch["length"]))
        original_lens = torch.tensor(minibatch["length"], device="cuda")
        minibatch.set_format(type="torch")

        input_data_minibatch = minibatch["input_ids"]
        input_data_minibatch = pu.pad_tensor_list(
            input_data_minibatch, max_len, pad_token_id, model_input_size
        )

        with torch.no_grad():
            outputs = model(
                input_ids=input_data_minibatch.to("cuda"),
                attention_mask=pu.gen_attention_mask(minibatch),
            )

        embs_i = outputs.hidden_states[layer_to_quant]

        if emb_mode == "cell":
            if cls_present:
                non_cls_embs = embs_i[:, 1:, :] # Get all layers except the embs
                if eos_present:
                    mean_embs = pu.mean_nonpadding_embs(non_cls_embs, original_lens - 2)
                else:
                    mean_embs = pu.mean_nonpadding_embs(non_cls_embs, original_lens - 1)
            else:
                mean_embs = pu.mean_nonpadding_embs(embs_i, original_lens)
            if summary_stat is None:
                embs_list.append(mean_embs)
            elif summary_stat is not None:
                # update tdigests with current batch for each emb dim
                accumulate_tdigests(embs_tdigests, mean_embs, emb_dims)
            del mean_embs
        elif emb_mode == "gene":
            if summary_stat is None:
                embs_list.append(embs_i)
            elif summary_stat is not None:
                for h in trange(len(minibatch)):
                    length_h = minibatch[h]["length"]
                    input_ids_h = minibatch[h]["input_ids"][0:length_h]

                    # double check dimensions before unsqueezing
                    embs_i_dim = embs_i.dim()
                    if embs_i_dim != 3:
                        logger.error(
                            f"Embedding tensor should have 3 dimensions, not {embs_i_dim}"
                        )
                        raise

                    embs_h = embs_i[h, :, :].unsqueeze(dim=1)
                    dict_h = dict(zip(input_ids_h, embs_h))
                    for k in dict_h.keys():
                        accumulate_tdigests(
                            embs_tdigests_dict[int(k)], dict_h[k], emb_dims
                        )
                    del embs_h
                    del dict_h
        elif emb_mode == "cls":
            cls_embs = embs_i[:,0,:].clone().detach() # CLS token layer
            embs_list.append(cls_embs)
            del cls_embs
            
        overall_max_len = max(overall_max_len, max_len)
        del outputs
        del minibatch
        del input_data_minibatch
        del embs_i

        torch.cuda.empty_cache()
        
        
    if summary_stat is None:
        if (emb_mode == "cell") or (emb_mode == "cls"):
            embs_stack = torch.cat(embs_list, dim=0)
        elif emb_mode == "gene":
            embs_stack = pu.pad_tensor_list(
                embs_list,
                overall_max_len,
                pad_token_id,
                model_input_size,
                1,
                pu.pad_3d_tensor,
            )

    # calculate summary stat embs from approximated tdigests
    elif summary_stat is not None:
        if emb_mode == "cell":
            if summary_stat == "mean":
                summary_emb_list = tdigest_mean(embs_tdigests, emb_dims)
            elif summary_stat == "median":
                summary_emb_list = tdigest_median(embs_tdigests, emb_dims)
            embs_stack = torch.tensor(summary_emb_list)
        elif emb_mode == "gene":
            if summary_stat == "mean":
                [
                    update_tdigest_dict_mean(embs_tdigests_dict, gene, emb_dims)
                    for gene in embs_tdigests_dict.keys()
                ]
            elif summary_stat == "median":
                [
                    update_tdigest_dict_median(embs_tdigests_dict, gene, emb_dims)
                    for gene in embs_tdigests_dict.keys()
                ]
            return embs_tdigests_dict

    return embs_stack


def label_cell_embs(embs, downsampled_data, emb_labels):
    embs_df = pd.DataFrame(embs.cpu().numpy())
    if emb_labels is not None:
        for label in emb_labels:
            emb_label = downsampled_data[label]
            embs_df[label] = emb_label
    return embs_df


def label_gene_embs(embs, downsampled_data, token_gene_dict):
    gene_set = {
        element for sublist in downsampled_data["input_ids"] for element in sublist
    }
    gene_emb_dict = {k: [] for k in gene_set}
    for i in range(embs.size()[0]):
        length = downsampled_data[i]["length"]
        dict_i = dict(
            zip(
                downsampled_data[i]["input_ids"][0:length],
                embs[i, :, :].unsqueeze(dim=1),
            )
        )
        for k in dict_i.keys():
            gene_emb_dict[k].append(dict_i[k])
    for k in gene_emb_dict.keys():
        gene_emb_dict[k] = (
            torch.squeeze(torch.mean(torch.stack(gene_emb_dict[k]), dim=0), dim=0)
            .cpu()
            .numpy()
        )
    embs_df = pd.DataFrame(gene_emb_dict).T
    embs_df.index = [token_gene_dict[token] for token in embs_df.index]
    return embs_df

