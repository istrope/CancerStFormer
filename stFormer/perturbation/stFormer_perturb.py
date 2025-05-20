"""
     isp = InSilicoPerturber(perturb_type="delete",
                             perturb_rank_shift=None,
                             genes_to_perturb="all",
                             model_type="CellClassifier",
                             num_classes=0,
                             emb_mode="cell",
                             filter_data={"cell_type":["cardiomyocyte"]},
                             cell_states_to_model={"state_key": "disease", "start_state": "dcm", "goal_state": "nf", "alt_states": ["hcm", "other1", "other2"]},
                             state_embs_dict ={"nf": emb_nf, "hcm": emb_hcm, "dcm": emb_dcm, "other1": emb_other1, "other2": emb_other2},
                             max_ncells=None,
                             emb_layer=0,
                             forward_batch_size=100,
                             nproc=16)
     isp.perturb_data("path/to/model",
                      "path/to/input_data",
                      "path/to/output_directory",
                      "output_prefix")
| Performs in silico perturbation (e.g. deletion or overexpression) of defined set of genes or all genes in sample of cells.
| Outputs impact of perturbation on cell or gene embeddings.
| Output files are analyzed with ``in_silico_perturber_stats``.
"""

import logging

# imports
import os
import pickle
from collections import defaultdict
from multiprocess import set_start_method
from typing import List, Optional
import pandas as pd
import torch
#from datasets import Dataset, disable_progress_bars
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from tqdm.auto import trange

import stFormer.perturbation.perturb_utils as pu
from stFormer.perturbation.embs import get_embs
disable_progress_bar()

logger = logging.getLogger(__name__)



class InSilicoPerturber:
    valid_option_dict = {
        'mode': {'spot','neighborhood'},
        "perturb_type": {"delete", "overexpress", "inhibit", "activate"},
        "perturb_rank_shift": {None, 1, 2, 3},
        "genes_to_perturb": {"all", list},
        "combos": {0, 1},
        "anchor_gene": {None, str},
        "model_type": {"Pretrained", "GeneClassifier", "CellClassifier"},
        "num_classes": {int},
        "emb_mode": {"cell", "cell_and_gene"},
        "cell_emb_style": {"mean_pool"},
        "filter_data": {None, dict},
        "cell_states_to_model": {None, dict},
        "state_embs_dict": {None, dict},
        "max_ncells": {None, int},
        "cell_inds_to_perturb": {"all", dict},
        "emb_layer": {-1, 0},
        "forward_batch_size": {int},
        "nproc": {int},
    }

    def __init__(
        self,
        mode ='spot',
        perturb_type="delete",
        perturb_rank_shift=None,
        genes_to_perturb="all",
        combos=0,
        anchor_gene=None,
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        filter_data=None,
        cell_states_to_model=None,
        state_embs_dict=None,
        max_ncells=None,
        cell_inds_to_perturb="all",
        emb_layer=-1,
        forward_batch_size=100,
        nproc=4,
        token_dictionary_file=None,
    ):
        """
        Initialize in silico perturber.
        **Parameters:**
        mode: {'spot','neighborhood'}
            | Type of dataset tokenization option
            | 'spot': if dataset is tokenized genes from only spot
            | 'neighborhood': if dataset is tokenized from spot and surrounding neighbors
        perturb_type : {"delete", "overexpress", "inhibit", "activate"}
            | Type of perturbation.
            | "delete": delete gene from rank value encoding
            | "overexpress": move gene to front of rank value encoding
        combos : {0,1}
            | Whether to perturb genes individually (0) or in pairs (1).
        anchor_gene : None, str
            | ENSEMBL ID of gene to use as anchor in combination perturbations.
            | For example, if combos=1 and anchor_gene="ENSG00000148400":
            |     anchor gene will be perturbed in combination with each other gene.
        model_type : {"Pretrained", "GeneClassifier", "CellClassifier"}
            | Whether model is the pretrained Geneformer or a fine-tuned gene or cell classifier.
        num_classes : int
            | If model is a gene or cell classifier, specify number of classes it was trained to classify.
            | For the pretrained Geneformer model, number of classes is 0 as it is not a classifier.
        emb_mode : {"cell", "cell_and_gene"}
            | Whether to output impact of perturbation on cell and/or gene embeddings.
            | Gene embedding shifts only available as compared to original cell, not comparing to goal state.
        cell_emb_style : "mean_pool"
            | Method for summarizing cell embeddings.
            | Currently only option is mean pooling of gene embeddings for given cell.
        filter_data : None, dict
            | Default is to use all input data for in silico perturbation study.
            | Otherwise, dictionary specifying .dataset column name and list of values to filter by.
        cell_states_to_model : None, dict
            | Cell states to model if testing perturbations that achieve goal state change.
            | Four-item dictionary with keys: state_key, start_state, goal_state, and alt_states
            | state_key: key specifying name of column in .dataset that defines the start/goal states
            | start_state: value in the state_key column that specifies the start state
            | goal_state: value in the state_key column taht specifies the goal end state
            | alt_states: list of values in the state_key column that specify the alternate end states
            | For example: {"state_key": "disease",
            |               "start_state": "dcm",
            |               "goal_state": "nf",
            |               "alt_states": ["hcm", "other1", "other2"]}
        state_embs_dict : None, dict
            | Embedding positions of each cell state to model shifts from/towards (e.g. mean or median).
            | Dictionary with keys specifying each possible cell state to model.
            | Values are target embedding positions as torch.tensor.
            | For example: {"nf": emb_nf,
            |               "hcm": emb_hcm,
            |               "dcm": emb_dcm,
            |               "other1": emb_other1,
            |               "other2": emb_other2}
        max_ncells : None, int
            | Maximum number of cells to test.
            | If None, will test all cells.
        cell_inds_to_perturb : "all", list
            | Default is perturbing each cell in the dataset.
            | Otherwise, may provide a dict of indices of cells to perturb with keys start_ind and end_ind.
            | start_ind: the first index to perturb.
            | end_ind: the last index to perturb (exclusive).
            | Indices will be selected *after* the filter_data criteria and sorting.
            | Useful for splitting extremely large datasets across separate GPUs.
        emb_layer : {-1, 0}
            | Embedding layer to use for quantification.
            | 0: last layer (recommended for questions closely tied to model's training objective)
            | -1: 2nd to last layer (recommended for questions requiring more general representations)
        forward_batch_size : int
            | Batch size for forward pass.
        nproc : int
            | Number of CPU processes to use.
        token_dictionary_file : Path
            | Path to pickle file containing token dictionary (Ensembl ID:token).
        """
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass
        self.perturb_type = perturb_type
        self.mode = mode
        self.perturb_rank_shift = perturb_rank_shift
        self.genes_to_perturb = genes_to_perturb
        self.combos = combos
        self.anchor_gene = anchor_gene
        if self.genes_to_perturb == "all":
            self.perturb_group = False
        else:
            self.perturb_group = True
            if (self.anchor_gene is not None) or (self.combos != 0):
                self.anchor_gene = None
                self.combos = 0
                logger.warning("anchor_gene set to None and combos set to 0. ""If providing list of genes to perturb, ""list of genes_to_perturb will be perturbed together, ""without anchor gene or combinations." )
        self.model_type = model_type
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.cell_emb_style = cell_emb_style
        self.filter_data = filter_data
        self.cell_states_to_model = cell_states_to_model
        self.state_embs_dict = state_embs_dict
        self.max_ncells = max_ncells
        self.cell_inds_to_perturb = cell_inds_to_perturb
        self.emb_layer = emb_layer
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.validate_options()

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)
        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}
        self.pad_token_id = self.gene_token_dict.get("<pad>")
        if self.anchor_gene is None: self.anchor_token = None
        else:
            try: self.anchor_token = [self.gene_token_dict[self.anchor_gene]]
            except KeyError: 
                logger.error(f"Anchor gene {self.anchor_gene} not in token dictionary.")
                raise
        if self.genes_to_perturb == "all":
            self.tokens_to_perturb = "all"
        else:
            missing_genes = [gene for gene in self.genes_to_perturb if gene not in self.gene_token_dict.keys()]
            if len(missing_genes) == len(self.genes_to_perturb):
                logger.error("None of the provided genes to perturb are in token dictionary.")
                raise
            elif len(missing_genes) > 0:
                logger.warning(f"Genes to perturb {missing_genes} are not in token dictionary.")
            self.tokens_to_perturb = [self.gene_token_dict.get(gene) for gene in self.genes_to_perturb]

    def validate_options(self):
        """
        Validate all init options for type, compatibility, and development status.
        Raises ValueError on invalid configuration.
        """
        # disallow options under development
        if self.perturb_type in ("inhibit", "activate"):
            logger.error(
                "In silico 'inhibit' and 'activate' are under development; "
                "valid perturb_type: 'delete' or 'overexpress'."
            )
            raise ValueError("Invalid perturb_type")
        if self.combos > 0 and self.anchor_gene is None:
            logger.error(
                "Combination perturbation without anchor_gene is under development; "
                "must provide anchor_gene for combos > 0."
            )
            raise ValueError("anchor_gene required for combos")

        # literal or type‐based validation
        for name, opts in self.valid_option_dict.items():
            val = getattr(self, name)
            ok = any(
                (val == opt) or (isinstance(opt, type) and isinstance(val, opt))
                for opt in opts
            )
            if not ok:
                logger.error(f"Invalid option for {name!r}: {val!r}. Valid: {opts}")
                raise ValueError(f"Invalid option for {name}")

        # delete/overexpress always clear rank_shift, with warning if set
        if self.perturb_type in ("delete", "overexpress"):
            if self.perturb_rank_shift is not None:
                msg = (
                    "perturb_rank_shift ignored for 'delete' (deletes gene) "
                    "or 'overexpress' (moves to front)."
                )
                logger.warning(msg)
            self.perturb_rank_shift = None

        # anchor_gene only works in cell mode
        if self.anchor_gene and self.emb_mode == "cell_and_gene":
            self.emb_mode = "cell"
            logger.warning(
                "emb_mode forced to 'cell' when using anchor_gene."
            )

        # validate cell_states_to_model block
        if self.cell_states_to_model is not None:
            pu.validate_cell_states_to_model(self.cell_states_to_model)

            # drop anchor when modeling multiple states
            if self.anchor_gene:
                self.anchor_gene = None
                logger.warning(
                    "anchor_gene disabled when modeling multiple cell_states."
                )

            # require state_embs_dict when cell_states_to_model is set
            if self.state_embs_dict is None:
                logger.error(
                    "state_embs_dict required when cell_states_to_model is provided."
                )
                raise ValueError("Missing state_embs_dict")
            # ensure all embeddings are tensors
            for st, emb in self.state_embs_dict.items():
                if not torch.is_tensor(emb):
                    logger.error("state_embs_dict values must be torch.Tensor.")
                    raise ValueError("Invalid state_embs_dict")

            # check that start_state, goal_state, alt_states keys exist in state_embs_dict
            missing = []
            cs = self.cell_states_to_model
            for key in ("start_state", "goal_state"):
                if cs[key] not in self.state_embs_dict:
                    missing.append(cs[key])
            for alt in cs.get("alt_states", []):
                if alt not in self.state_embs_dict:
                    missing.append(alt)
            if missing:
                logger.error(
                    f"cell_states_to_model values missing in state_embs_dict: {missing}"
                )
                raise ValueError("Missing state_embs_dict entries")

        # require perturb_rank_shift for inhibit/activate (if they were allowed)
        if self.perturb_type in ("inhibit", "activate") and self.perturb_rank_shift is None:
            logger.error(
                "perturb_rank_shift must be set when perturb_type is 'inhibit' or 'activate'."
            )
            raise ValueError("Missing perturb_rank_shift")

        # normalize filter_data values to lists
        if self.filter_data is not None:
            for k, v in list(self.filter_data.items()):
                if not isinstance(v, list):
                    self.filter_data[k] = [v]
                    logger.warning(
                        f"filter_data['{k}'] converted to list: {[v]}"
                    )

        # validate cell_inds_to_perturb structure
        if self.cell_inds_to_perturb != "all":
            keys = set(self.cell_inds_to_perturb.keys())
            if keys != {"start", "end"}:
                logger.error("cell_inds_to_perturb keys must be {'start','end'}.")
                raise ValueError("Invalid cell_inds_to_perturb keys")
            if self.cell_inds_to_perturb["start"] < 0 or self.cell_inds_to_perturb["end"] < 0:
                logger.error("cell_inds_to_perturb indices must be non-negative.")
                raise ValueError("Negative cell_inds_to_perturb")

    def perturb_data(
        self, model_directory, input_data_file, output_directory, output_prefix
    ):
        output_path_prefix = os.path.join(
            output_directory, f"in_silico_{self.perturb_type}_{output_prefix}"
        )
        model = pu.load_model(
            self.model_type, self.num_classes, model_directory, mode="eval"
        )
        self.max_len = pu.get_model_input_size(model)
        layer_to_quant = pu.quant_layers(model) + self.emb_layer

        filtered_input_data = pu.load_and_filter(
            self.filter_data, self.nproc, input_data_file
        )
        filtered_input_data = self.apply_additional_filters(filtered_input_data)

        if self.perturb_group is True:
            self.isp_perturb_set(
                model, filtered_input_data, layer_to_quant, output_path_prefix
            )
        else:
            self.isp_perturb_all(
                model, filtered_input_data, layer_to_quant, output_path_prefix
            )

    def apply_additional_filters(self, filtered_input_data):
        # additional filtering of input data dependent on isp mode
        if self.cell_states_to_model is not None:
            # filter for cells with start_state and log result
            filtered_input_data = pu.filter_data_by_start_state(
                filtered_input_data, self.cell_states_to_model, self.nproc
            )

        if (self.tokens_to_perturb != "all") and (self.perturb_type != "overexpress"):
            # filter for cells with tokens_to_perturb and log result
            filtered_input_data = pu.filter_data_by_tokens_and_log(
                filtered_input_data,
                self.tokens_to_perturb,
                self.nproc,
                "genes_to_perturb",
            )

        if self.anchor_token is not None:
            # filter for cells with anchor gene and log result
            filtered_input_data = pu.filter_data_by_tokens_and_log(
                filtered_input_data, self.anchor_token, self.nproc, "anchor_gene"
            )

        # downsample and sort largest to smallest to encounter memory constraints earlier
        filtered_input_data = pu.downsample_and_sort(
            filtered_input_data, self.max_ncells
        )

        # slice dataset if cells_inds_to_perturb is not "all"
        if self.cell_inds_to_perturb != "all":
            filtered_input_data = pu.slice_by_inds_to_perturb(
                filtered_input_data, self.cell_inds_to_perturb
            )

        return filtered_input_data


    def isp_perturb_set(self, model, dataset: Dataset, layer_to_quant: int, prefix: str):
        """
        Perform in-silico group perturbations on dataset, compute cosine similarities,
        and write results to disk under prefix.
        """
        import torch
        from torch.utils.data import DataLoader
        from torch.nn.utils.rnn import pad_sequence
        from collections import defaultdict
        from tqdm import tqdm

        #  INNER FUNCTIONS 

        def batch_fn(example):
            ids = example["input_ids"]
            toks = self.tokens_to_perturb
            example["tokens_to_perturb"] = self.tokens_to_perturb
            #idxs = [ids.index(t) for t in self.tokens_to_perturb if t in ids]
            if self.mode == "spot":
                # original logic: pick first hit (or multi-hit, up to you)
                idxs = [ids.index(t) for t in toks if t in ids]
            else:  # neighbor mode: require at least one in spot _and_ neighbor
                spot_boundary = self.max_len
                idxs = []
                for t in toks:
                    # all positions of this token
                    pos = [i for i, x in enumerate(ids) if x == t]
                    # split into spot (< boundary) and neighbor (>= boundary)
                    spot_pos     = [i for i in pos if i < spot_boundary]
                    neighbor_pos = [i for i in pos if i >= spot_boundary]
                    # only keep if both regions contain the token
                    if spot_pos and neighbor_pos:
                        idxs.extend(pos)

            example["perturb_index"] = idxs or [-100]

            if self.perturb_type == "delete":
                example = pu.delete_indices(example)
            else:  # overexpress
                example = pu.overexpress_tokens(example, self.max_len)
                example["n_overflow"] = pu.calc_n_overflow(
                    self.max_len, example["length"], self.tokens_to_perturb, idxs
                )
            return example

        def padding_collate_fn_orig(batch):
            input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]
            lengths = torch.tensor([len(ids) for ids in input_ids])
            padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

            return {
                "input_ids": padded_input_ids,
                "lengths": lengths
            }

        def padding_collate_fn_pert(batch):
            input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]
            lengths = torch.tensor([len(ids) for ids in input_ids])
            perturb_index = [sample["perturb_index"] for sample in batch]
            padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

            return {
                "input_ids": padded_input_ids,
                "lengths": lengths,
                "perturb_index": perturb_index
            }

        #  PREPARE DATA 

        total = len(dataset)

        cos_sims = defaultdict(list)
        gene_embs = defaultdict(list)

        # Map perturbations
        perturbed = dataset.map(batch_fn, num_proc=self.nproc)

        if self.perturb_type == "overexpress":
            dataset = dataset.add_column("n_overflow", perturbed["n_overflow"])
            dataset = dataset.map(pu.truncate_by_n_overflow, num_proc=self.nproc)

        # Create DataLoaders
        orig_loader = DataLoader(
            dataset,
            batch_size=self.forward_batch_size,
            shuffle=False,
            collate_fn=padding_collate_fn_orig
        )

        pert_loader = DataLoader(
            perturbed,
            batch_size=self.forward_batch_size,
            shuffle=False,
            collate_fn=padding_collate_fn_pert
        )

        #  MAIN LOOP 

        for batch_num, (orig_batch, pert_batch) in enumerate(
            tqdm(zip(orig_loader, pert_loader), total=len(orig_loader))
        ):
            #  Align batch lengths 
            min_seq_len = min(orig_batch["input_ids"].shape[1], pert_batch["input_ids"].shape[1])
            orig_batch["input_ids"] = orig_batch["input_ids"][:, :min_seq_len]
            pert_batch["input_ids"] = pert_batch["input_ids"][:, :min_seq_len]

            #  Embedding extraction 
            full_orig = get_embs(
                model, orig_batch, "gene", layer_to_quant, self.pad_token_id,
                self.forward_batch_size, token_gene_dict=self.token_gene_dict,
                summary_stat=None, silent=True
            )
            full_pert = get_embs(
                model, pert_batch, "gene", layer_to_quant, self.pad_token_id,
                self.forward_batch_size, token_gene_dict=self.token_gene_dict,
                summary_stat=None, silent=True
            )
            #  Align embedding lengths 
            min_emb_len = min(full_pert.shape[1], full_orig.shape[1])
            full_pert = full_pert[:, :min_emb_len, :]
            full_orig = full_orig[:, :min_emb_len, :]

            #  Gene-level cosine similarities 
            if self.cell_states_to_model is None or self.emb_mode == "cell_and_gene":
                gene_cos = pu.quant_cos_sims(
                    full_pert, full_orig, self.cell_states_to_model,
                    self.state_embs_dict, emb_mode="gene"
                )

            #  Cell-level cosine similarities 
            if self.cell_states_to_model is not None:
                orig_cell = pu.mean_nonpadding_embs(
                    full_orig, torch.tensor(orig_batch["length"], device="cuda"), dim=1
                )
                pert_cell = pu.mean_nonpadding_embs(
                    full_pert, torch.tensor(pert_batch["length"], device="cuda"), dim=1
                )
                cell_cos = pu.quant_cos_sims(
                    pert_cell, orig_cell, self.cell_states_to_model,
                    self.state_embs_dict, emb_mode="cell"
                )

            #  Gene embedding dictionary update 
            if self.emb_mode == "cell_and_gene":
                n_genes = gene_cos.size(1)
                genes_list = [
                    [g for g in seq.tolist() if g not in self.tokens_to_perturb][:n_genes]
                    for seq in orig_batch["input_ids"]
                ]
                for i, gene_seq in enumerate(genes_list):
                    for j, g in enumerate(gene_seq):
                        key = (tuple(self.tokens_to_perturb)
                            if len(self.tokens_to_perturb) > 1
                            else self.tokens_to_perturb[0])
                        gene_embs[(key, g)].append(gene_cos[i, j].item())

            #  Cosine similarity dictionary update 
            if self.cell_states_to_model is None:
                if self.perturb_type == "overexpress":
                    nonpad = [l - len(self.tokens_to_perturb) for l in pert_batch["lengths"]]
                else:
                    nonpad = pert_batch["lengths"]
                cos_data = pu.mean_nonpadding_embs(
                    gene_cos, torch.tensor(nonpad, device="cuda")
                )
                cos_sims = self.update_perturbation_dictionary(
                    cos_sims, cos_data, None, None, None
                )
            else:
                for state, sim_list in cell_cos.items():
                    cos_sims[state] = self.update_perturbation_dictionary(
                        cos_sims[state], sim_list, None, None, None
                    )

            #  Memory cleanup 
            del orig_batch, pert_batch, full_orig, full_pert
            torch.cuda.empty_cache()

        #  SAVE RESULTS 

        pu.write_perturbation_dictionary(
            cos_sims, f"{prefix}_cell_embs_dict_{self.tokens_to_perturb}"
        )
        if self.emb_mode == "cell_and_gene":
            pu.write_perturbation_dictionary(
                gene_embs, f"{prefix}_gene_embs_dict_{self.tokens_to_perturb}"
            )

    def isp_perturb_all(self, model, dataset: Dataset, layer_to_quant: int, prefix: str):
        """
        Perturb each cell individually, compute cosine shifts, and write results in batches.
        """

        import torch
        from torch.utils.data import DataLoader
        from torch.nn.utils.rnn import pad_sequence
        from collections import defaultdict
        from tqdm import trange

        #  Collate function 
        def padding_collate_fn_orig(batch):
            input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]
            lengths = torch.tensor([len(ids) for ids in input_ids])
            padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            return {"input_ids": padded_input_ids, "lengths": lengths}

        #  Initialize storage 
        batch_idx = -1

        if self.cell_states_to_model is None:
            cos_sims = defaultdict(list)
        else:
            cos_sims = {
                st: defaultdict(list)
                for st in pu.get_possible_states(self.cell_states_to_model)
            }

        gene_embs = defaultdict(list) if self.emb_mode == "cell_and_gene" else None

        #  Iterate over cells 
        for i in trange(len(dataset), desc="Perturbing all cells"):

            #  Original embedding 
            cell = dataset.select([i])
            cell_batch = padding_collate_fn_orig(cell)

            full_orig = get_embs(
                model, cell_batch, "gene", layer_to_quant, self.pad_token_id,
                self.forward_batch_size, token_gene_dict=self.token_gene_dict,
                summary_stat=None, silent=True
            )

            genes = cell_batch["input_ids"][0][:].tolist()

            #  Perturbation batch 
            pert_ds, idxs = pu.make_perturbation_batch(
                cell, self.perturb_type, self.tokens_to_perturb,
                self.anchor_token, self.combos, self.nproc
            )

            pert_batch = padding_collate_fn_orig(pert_ds)

            full_pert = get_embs(
                model, pert_batch, "gene", layer_to_quant, self.pad_token_id,
                self.forward_batch_size, token_gene_dict=self.token_gene_dict,
                summary_stat=None, silent=True
            )

            #  Align embedding lengths 
            min_len = min(full_orig.shape[1], full_pert.shape[1])
            full_orig = full_orig[:, :min_len, :]
            full_pert = full_pert[:, :min_len, :]

            #  Slice out perturbed genes 
            n_pert = 1 + self.combos
            if self.perturb_type == "overexpress":
                pert_emb = full_pert[:, n_pert:, :]
                genes = genes[n_pert:]  # remove overexpressed tokens
            else:
                pert_emb = full_pert

            #  Comparison batch for cosine sims 
            orig_batch = pu.make_comparison_batch(
                full_orig, idxs, perturb_group=False
            )

            gene_cos = pu.quant_cos_sims(
                pert_emb, orig_batch, self.cell_states_to_model,
                self.state_embs_dict, emb_mode="gene"
            )

            #  Cell-level similarity 
            if self.cell_states_to_model is not None:
                orig_cell = pu.compute_nonpadded_cell_embedding(
                    full_orig, "mean_pool"
                )
                pert_cell = pu.compute_nonpadded_cell_embedding(
                    full_pert, "mean_pool"
                )
                cell_cos = pu.quant_cos_sims(
                    pert_cell, orig_cell, self.cell_states_to_model,
                    self.state_embs_dict, emb_mode="cell"
                )

            #  Gene-level dictionary update 
            if self.emb_mode == "cell_and_gene":
                for p_i, p_gene in enumerate(genes):
                    affected = genes[:p_i] + genes[p_i+1:]
                    for a_gene, sim in zip(affected, gene_cos[p_i].tolist()):
                        key = (tuple(self.tokens_to_perturb)
                            if isinstance(self.tokens_to_perturb, (list, tuple))
                            else self.tokens_to_perturb)
                        gene_embs[(p_gene, a_gene)].append(sim)

            #  Cosine similarity aggregation 
            if self.cell_states_to_model is None:
                avg_shifts = torch.mean(gene_cos, dim=1)
                cos_sims = self.update_perturbation_dictionary(
                    cos_sims, avg_shifts, None, None, genes
                )
            else:
                for state, sims in cell_cos.items():
                    cos_sims[state] = self.update_perturbation_dictionary(
                        cos_sims[state], sims, None, None, genes
                    )

            # Periodic save
            if i % 100 == 0:
                pu.write_perturbation_dictionary(
                    cos_sims, f"{prefix}_dict_cell_embs_1Kbatch{batch_idx}"
                )
                if gene_embs:
                    pu.write_perturbation_dictionary(
                        gene_embs, f"{prefix}_dict_gene_embs_1Kbatch{batch_idx}"
                    )

            #  Rotate batch dictionaries 
            if i % 1000 == 0 and i > 0:
                batch_idx += 1
                if self.cell_states_to_model is None:
                    cos_sims = defaultdict(list)
                else:
                    cos_sims = {
                        st: defaultdict(list)
                        for st in pu.get_possible_states(self.cell_states_to_model)
                    }
                if gene_embs is not None:
                    gene_embs = defaultdict(list)
                torch.cuda.empty_cache()

            del cell_batch, pert_batch, full_orig, full_pert
            torch.cuda.empty_cache()

        #  Final write 
        pu.write_perturbation_dictionary(
            cos_sims, f"{prefix}_dict_cell_embs_1Kbatch{batch_idx}"
        )
        if gene_embs:
            pu.write_perturbation_dictionary(
                gene_embs, f"{prefix}_dict_gene_embs_1Kbatch{batch_idx}"
            )

    def update_perturbation_dictionary(
        self,
        cos_sims: defaultdict,
        sims: torch.Tensor,
        _data,                      # unused here but kept for signature compatibility
        _indices,                  # ditto
        genes: Optional[List] = None,
    ) -> defaultdict:
        """
        Append cosine‐similarity values to cos_sims.
        - If perturb_group: squeeze sims to a 1D list and store under the perturbed‐genes key.
        - Else: pair each sim with the corresponding gene in `genes`.
        """
        # sanity check
        if genes is not None and sims.shape[0] != len(genes):
            raise ValueError(
                f"sims rows ({sims.shape[0]}) != len(genes) ({len(genes)})"
            )

        if self.perturb_group:
            # determine key: single token or tuple of tokens
            tok_key = (
                tuple(self.tokens_to_perturb)
                if len(self.tokens_to_perturb) > 1
                else self.tokens_to_perturb[0]
            )
            # flatten to Python list
            vals = torch.squeeze(sims).tolist()
            if not isinstance(vals, list):
                vals = [vals]
            cos_sims[(tok_key, "cell_emb")].extend(vals)

        else:
            for val, gene in zip(sims.tolist(), genes or []):
                cos_sims[(gene, "cell_emb")].append(val)

        return cos_sims