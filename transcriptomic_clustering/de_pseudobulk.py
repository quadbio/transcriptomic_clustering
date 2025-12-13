import anndata as ad
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import decoupler as dc

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

import multiprocessing as mp
from functools import partial

from .diff_expression import get_qdiff, filter_gene_stats, calc_de_score

import logging
logger = logging.getLogger(__name__)


class BulkDE:
    """
    Class to handle selection of conditions, identification of replicates/batches,
    assessment of batch inclusion, and (later) generation of pseudoreplicates for DE analysis.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        group_key: str,
        query: str,
        reference: Union[str, List[str]],
        group_names: Tuple[str, str] = ("query", "reference"),
        comparison_key_added: str = "_comparison_group",
        replicate_key: Optional[str] = None,
        pb_replicate_key: str = "_psbulk_replicate",
        replicate_min_cells: int = 50,
        replicate_min_fraction: float = 0.3,
        batch_key: Optional[str] = None,
        min_bridging_batches: int = 2,
        pb_batch_key: str = "_psbulk_batch",

        # Decoupler arguments
        layer: str = 'counts',
        mode: str = 'sum',

        # Pseudoreplicate arguments
        min_replicates: int = 5,
        resampling_fraction: float = 0.6,
        n_repetitions: int = 10,
        min_list_overlap: float = 0.8,
        
        # DE arguments
        alpha: float = 0.05,
        cooks: bool = True,
        independent_filter: bool = True,
        fit_type: str = "mean",
 
        # Other arguments
        n_cpus: int = 1,
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize the helper with AnnData and column keys.
        """

        self.adata = adata
        self.group_key = group_key
        self.query = query
        self.reference = reference
        self.group_names = group_names
        self.comparison_key_added = comparison_key_added
        self.replicate_key = replicate_key
        self.pb_replicate_key = pb_replicate_key
        self.replicate_min_cells = replicate_min_cells
        self.replicate_min_fraction = replicate_min_fraction
        self.batch_key = batch_key
        self.min_bridging_batches = min_bridging_batches
        self.pb_batch_key = pb_batch_key

        self.layer = layer
        self.mode = mode

        self.min_replicates = min_replicates
        self.resampling_fraction = resampling_fraction
        self.n_repetitions = n_repetitions
        self.min_list_overlap = min_list_overlap
        
        self.alpha = alpha
        self.cooks = cooks
        self.independent_filter = independent_filter
        self.fit_type = fit_type

        self.seed = seed
        self.n_cpus = n_cpus
        self.verbose = verbose

        # Instantiate random generator
        self.rng = np.random.default_rng(self.seed)

        # Select the cells corresponding to the two conditions
        self.adata_sub = self._select_conditions(
            adata=self.adata,
            group_key=self.group_key,
            query=self.query,
            reference=self.reference,
            group_names=self.group_names,
            comparison_key_added=self.comparison_key_added,
        )

        if self.adata_sub.n_obs == 0:
            logger.error("Condition selection failed. Exiting initialization.")
            return
        
        # ensure canonical psbulk columns exist and get their names
        self.adata_sub, self.pb_replicate_key, self.pb_batch_key = self._format_psbulk_columns(
            adata_sub=self.adata_sub,
            comparison_key=self.comparison_key_added,
            replicate_key=self.replicate_key,
            batch_key=self.batch_key,
            ps_replicate_key=self.pb_replicate_key,
            ps_batch_key=self.pb_batch_key
        )
    
        # Identify replicates and batches using the concrete psbulk_replicate_key column
        self.adata_sub, self.sample_hierarchy, self.replicate_info = self._identify_replicates_and_batches(
            adata_sub=self.adata_sub,
            comparison_key=self.comparison_key_added,
            group_names=self.group_names,
            replicate_key=self.pb_replicate_key,
            replicate_min_cells=self.replicate_min_cells,
            replicate_min_fraction=self.replicate_min_fraction,
            batch_key=self.pb_batch_key,
            verbose=self.verbose,
        )

        # Assess batch inclusion
        self.batch_info = self._include_batch(
            sample_hierarchy=self.sample_hierarchy,
            min_bridging_batches=self.min_bridging_batches,
            group_names=self.group_names)
        self.batch_as_covariate = self.batch_info["include_batch"]

        # Filter adata_sub to only include cells that belong to valid replicates
        valid_reps = (
            self.replicate_info["valid_replicates_by_condition"][self.group_names[0]]
            + self.replicate_info["valid_replicates_by_condition"][self.group_names[1]]
        )
        self.adata_sub = self.adata_sub[self.adata_sub.obs[self.pb_replicate_key].isin(valid_reps)].copy()

        # Generate pseudoreplicates if needed to ensure minimum replicates per condition
        self.de = self._de(
            adata_sub=self.adata_sub,
            comparison_key=self.comparison_key_added,
            group_names=self.group_names,
            sample_hierarchy=self.sample_hierarchy,
            psbulk_replicate_key=self.pb_replicate_key,
            batch_key=self.pb_batch_key if self.batch_as_covariate else None,
            min_replicates=self.min_replicates,
            resampling_fraction=self.resampling_fraction,
            rng=self.rng,
            layer=self.layer,
            mode=self.mode,
            alpha=self.alpha,
            cooks=self.cooks,
            fit_type=self.fit_type,
            independent_filter=self.independent_filter,
            n_repetitions=self.n_repetitions,
            min_list_overlap=self.min_list_overlap,
            n_cpus=self.n_cpus,
            verbose=self.verbose,
        )

    @staticmethod
    def _select_conditions(
        adata: ad.AnnData,
        group_key: str,
        query: str,
        reference: Union[str, List[str]],
        group_names: Tuple[str, str],
        comparison_key_added: str,
    ) -> Optional[ad.AnnData]:
        """Return AnnData subset containing query vs. reference groups."""

        mask_query = adata.obs[group_key] == query
        if mask_query.sum() == 0:
            logger.error(f"No cells found for query group '{query}' in '{group_key}'.")

        if isinstance(reference, str):
            mask_reference = (
                adata.obs[group_key] != query if reference == "rest"
                else adata.obs[group_key] == reference
            )
        else:
            mask_reference = adata.obs[group_key].isin(reference)

        if mask_reference.sum() == 0:
            logger.error(f"No cells found for reference group '{reference}' in '{group_key}'.")

        mask = mask_query | mask_reference
        adata_sub = adata[mask].copy()

        adata_sub.obs.loc[:, comparison_key_added] = np.where(
            mask_query[mask], group_names[0], group_names[1]
        )

        return adata_sub

    @staticmethod
    def _format_psbulk_columns(
        adata_sub: ad.AnnData,
        comparison_key: str,
        replicate_key: Optional[str],
        batch_key: Optional[str],
        ps_replicate_key: str,
        ps_batch_key: str
    ) -> Tuple[ad.AnnData, str, str]:
        """
        Ensure adata_sub.obs contains canonical psbulk columns:
        - ps_replicate_col (prefix condition + original replicate id, or condition_replicate_1 if None)
        - ps_batch_col (copy of batch_key if provided, otherwise 'batch_1')

        Returns:
        - adata_sub (obs modified in place)
        - name of replicate column written (ps_replicate_col)
        - name of batch column written (ps_batch_col)
        """

        obs = adata_sub.obs

        # Create _psbulk_replicate
        if replicate_key is not None and replicate_key in obs.columns:
            # Vectorized concat: "Condition_ReplicateID>"
            obs[ps_replicate_key] = (
                obs[comparison_key].astype(str).str.cat(obs[replicate_key].astype(str), sep="_")
            )
        else:
            # No replicate info provided, then collapse to single per-condition replicate id
            obs[ps_replicate_key] = obs[comparison_key].astype(str) + "_replicate_1"

        # Create _psbulk_batch
        if batch_key is not None and batch_key in obs.columns:
            # copy values as-is (no prefix)
            obs[ps_batch_key] = obs[batch_key].astype(str)
        else:
            # Single global batch (all cells same batch)
            obs[ps_batch_key] = "batch_1"

        # Convert to categorical
        obs[ps_replicate_key] = obs[ps_replicate_key].astype("category")
        obs[ps_batch_key] = obs[ps_batch_key].astype("category")

        return adata_sub, ps_replicate_key, ps_batch_key

    @staticmethod
    def _identify_replicates_and_batches(
        adata_sub: ad.AnnData,
        comparison_key: str,
        group_names: Tuple[str, str],
        replicate_key: Optional[str],
        replicate_min_cells: int,
        replicate_min_fraction: float,
        batch_key: Optional[str],
        verbose: bool = False,
    ) -> Tuple[Optional[Dict], Optional[Dict[str, Any]]]:
        """
        Identify "true" biological replicates per condition and return a sample_hierarchy structure:
            {condition -> replicate -> batch -> [obs_names]}.

        A replicate is considered "true" if it has at least `replicate_min_cells` cells
        *and* represents at least `replicate_min_fraction` of all cells in its condition.
        If no replicate in a condition meets these criteria, the entire condition is
        collapsed into one replicate.

        Batch evaluation is done only on replicates deemed "true". The function simply
        structures the data; no pseudoreplicates are generated at this stage.
        """

        # Make a working copy
        adata_sub = adata_sub.copy()

        # Ensure comparison_key present
        if comparison_key not in adata_sub.obs.columns:
            logger.error(f"comparison_key '{comparison_key}' not found in adata.obs")
            return None, None

        # Compute counts per condition and replicate
        # The observed=True makes it such that in each group of the comparison_key, only the 
        # replicates that belong to that group are retained
        counts = (
            adata_sub.obs.groupby([comparison_key, replicate_key], observed=True) 
              .size()
              .reset_index(name="n_cells")
        )

        # Condition totals
        counts["condition_total"] = counts.groupby(comparison_key)["n_cells"].transform("sum")
        counts["fraction"] = counts["n_cells"] / counts["condition_total"]

        # Determine which replicates can be considered as replicates (should not be too small)
        # Debatable if logical_and or logical_or is better here
        mask_true = (counts["n_cells"] >= replicate_min_cells) | (counts["fraction"] >= replicate_min_fraction)
        counts["considered_as_replicate"] = mask_true

        # Build mapping of true replicates per condition with structure
        # {condition: [replicate_id, ...], ...}
        valid_replicates_by_condition = (
            counts[counts["considered_as_replicate"]]
            .groupby(comparison_key)[replicate_key]
            .apply(list)
            .to_dict()
        )

        # Now, there might be cases where no replicate in a condition meets the criteria. This might be due to an 
        # unfortunate choice of thresholds e.g. they don't generalize well in iterative clustering algorithms, where
        # for very small clusters that are to be compared none of the replicates have enough cells. In these cases,
        # collapse the entire condition into a single replicate
        all_conditions = list(group_names)
        collapsed_conditions = []
        for cond in all_conditions:
            if cond not in valid_replicates_by_condition.keys():
                collapsed_conditions.append(cond)
                valid_replicates_by_condition[cond] = [f"{cond}_collapsed_rep"]

                adata_sub.obs[replicate_key] = np.where(
                    adata_sub.obs[comparison_key] == cond,
                    f"{cond}_collapsed_rep",
                    adata_sub.obs[replicate_key]
                )

        # Build sample_hierarchy structure for ONLY for the replicates that can be considered (including collapsed reps)
        sample_hierarchy: Dict[str, Dict[str, Dict[str, list]]] = {}
        for cond in all_conditions:
            sample_hierarchy[cond] = {}

            # If this condition was collapsed, take all obs in that condition as that single replicate
            if cond in collapsed_conditions:
                collapsed_id = valid_replicates_by_condition[cond][0]  # there is only one
                cond_cells = adata_sub.obs[adata_sub.obs[comparison_key] == cond]  # get all cells for this condition
                # Group by batch for that whole condition
                for batch_val, sub in cond_cells.groupby(batch_key):
                    # Because I'm iterating over the batch_vals, for the first iteration, the
                    # dict entry for the collapsed replicate is created using setdefault.
                    # Here it can never happen that there are no cells
                    sample_hierarchy[cond].setdefault(collapsed_id, {})[batch_val] = sub.index.tolist()

            else:
                # iterate through each true replicate and collect its per-batch cells
                for rep in valid_replicates_by_condition[cond]:
                    sample_hierarchy[cond].setdefault(rep, {})
                    # Get cells that belong to this replicate in this condition
                    rep_cells = adata_sub.obs[(adata_sub.obs[comparison_key] == cond) & (adata_sub.obs[replicate_key] == rep)]
                    # Group by batch. Here it can happen that in a batch there are no cells
                    # If this is the case, don't include it
                    for batch_val, sub in rep_cells.groupby(batch_key):
                        cell_ids = sub.index.tolist()
                        if len(cell_ids) > 0:
                            sample_hierarchy[cond][rep][batch_val] = sub.index.tolist()

        # Prepare info
        condition_totals = counts.groupby(comparison_key)["n_cells"].sum().to_dict()
        info = {
            "valid_replicates_by_condition": valid_replicates_by_condition,
            "collapsed_conditions": collapsed_conditions,
            "replicate_counts_table": counts, 
            "condition_totals": condition_totals,
            "used_replicate_key": replicate_key,
            "used_batch_key": batch_key,
            "replicate_min_cells": replicate_min_cells,
            "replicate_min_fraction": replicate_min_fraction,
        }

        #if verbose:
        # print(info)
        # print("Condition totals:", condition_totals)
        # print("Valid replicates by condition:", valid_replicates_by_condition)
        # if collapsed_conditions:
        #     print("Collapsed conditions (no valid replicate found):", collapsed_conditions)

        return adata_sub, sample_hierarchy, info

    @staticmethod
    def _include_batch(
        sample_hierarchy: dict,
        min_bridging_batches: int = 2,
        group_names: Tuple[str, str] = ("query", "reference"),
    ) -> Dict[str, Any]:
        """
        Determine whether 'batch' can be included as a covariate in DE testing.

        Criteria:
            - At least one batch must be present in both conditions (bridging batch exists).
            - Only "true" replicates (or collapsed replicates) are considered.

        Returns a dict with the batch inclusion decision and bridging batches.
        """

        condA, condB = group_names

        sample_hierarchy_a = sample_hierarchy[condA]
        sample_hierarchy_b = sample_hierarchy[condB]

        batches_a = {b for rep in sample_hierarchy_a.values() for b in rep.keys()}
        batches_b = {b for rep in sample_hierarchy_b.values() for b in rep.keys()}

        bridging = sorted(batches_a & batches_b)
        if len(bridging) >= min_bridging_batches:
            include_batch = True
        else:
            include_batch = False

        result = {
            f"batches_in_{condA}": sorted(batches_a),
            f"batches_in_{condB}": sorted(batches_b),
            "include_batch": include_batch,
            "bridging_batches": bridging
        }

        #print(result)

        return result
    
    def _generate_pseudoreplicate(
        self,
        adata_sub: ad.AnnData,
        obs_names: List[str],
        psbulk_replicate_key: str,
        batch_key: Optional[str],
        sampling_fraction: float,
        rng: np.random.default_rng,
        layer: str,
        mode: str
    ) -> ad.AnnData:
        """
        Generate a single pseudoreplicate by sampling cells from the provided obs_names.
        """

        n_cells = len(obs_names)
        n_sample = max(1, int(n_cells * sampling_fraction))

        sampled_cells = rng.choice(obs_names, size=n_sample, replace=False)
        adata_sampled = adata_sub[sampled_cells].copy()

        return dc.pp.pseudobulk(
            adata_sampled,
            sample_col=psbulk_replicate_key,
            groups_col=batch_key,
            layer=layer,
            mode=mode
        )

    def _ensure_min_replicates(
        self,
        adata_sub: ad.AnnData,
        sample_hierarchy: dict,
        psbulk_replicate_key: str,
        batch_key: Optional[str],
        required_reps: dict,
        resampling_fraction: float,
        rng: np.random.default_rng,
        layer: str,
        mode: str
    ) -> ad.AnnData:
        """
        Ensure that each condition has at least `min_replicates` pseudoreplicates by generating additional
        pseudoreplicates through resampling if necessary.
        """
        adata_list = []

        for condition, replicates in sample_hierarchy.items():

            reps_needed = required_reps[condition]

            # If batch should be considered
            if batch_key is not None:
                
                for i in range(reps_needed):
                    # Randomly select a replicate to resample from
                    rep_id = rng.choice(list(replicates.keys()))
                    # Randomly select a batch of the replicate (if it exists)
                    batches = replicates[rep_id]
                    batch_id = rng.choice(list(batches.keys()))
                    obs_names = batches[batch_id]
                    adata_rep = self._generate_pseudoreplicate(
                        adata_sub,
                        obs_names,
                        psbulk_replicate_key,
                        batch_key,
                        resampling_fraction,
                        rng,
                        layer,
                        mode
                    )
                    # Rename the replicate to indicate it's a pseudoreplicate
                    new_rep_id = f"{rep_id}_pr_{i+1}"
                    adata_rep.obs[psbulk_replicate_key] = new_rep_id
                    # Ensure that the obs name is unique. There is only one because
                    # one batch has been selected
                    adata_rep.obs_names = [f"{adata_rep.obs_names.tolist()[0]}_pr_{i+1}"]
                    adata_list.append(adata_rep)

            # Don't consider batch. Here the I sample from the entire replicate directly
            else:

                # Generate additional pseudoreplicates if needed
                for i in range(reps_needed):
                    # Randomly select a replicate to resample from
                    rep_id = rng.choice(list(replicates.keys()))
                    obs_names = [cell for batch_cells in replicates[rep_id].values() for cell in batch_cells]
                    adata_rep = self._generate_pseudoreplicate(
                        adata_sub,
                        obs_names,
                        psbulk_replicate_key,
                        batch_key,
                        resampling_fraction,
                        rng,
                        layer,
                        mode
                    )
                    # Rename the replicate to indicate it's a pseudoreplicate
                    new_rep_id = f"{rep_id}_pr_{i+1}"
                    adata_rep.obs[psbulk_replicate_key] = new_rep_id
                    # Ensure that the obs name is unique. There is only one because
                    # one replicate has been selected
                    adata_rep.obs_names = [f"{adata_rep.obs_names.tolist()[0]}_pr_{i+1}"]
                    adata_list.append(adata_rep)

        # Concatenate all pseudoreplicates into a single AnnData
        adata_pr = ad.concat(adata_list, axis=0)

        return adata_pr
    
    def _pydeseq2_wrapper(
            self,
            counts: pd.DataFrame, # pb x genes
            metadata: pd.DataFrame, # pb x covariates
            design: str,
            contrast: list,
            alpha: float,
            cooks: bool,
            fit_type: str,
            independent_filter: bool,
            n_cpus,
            verbose: bool,
            ) -> DeseqStats:
        """
        Wrapper around pydeseq2 DE testing.
        """

        inference = DefaultInference(
            n_cpus=n_cpus,
        )

        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design=design,
            inference=inference,
            refit_cooks=cooks,
            fit_type=fit_type,
            quiet=not verbose,
        )

        dds.fit_size_factors()
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_dispersion_prior()
        dds.fit_MAP_dispersions()
        dds.fit_LFC()
        dds.calculate_cooks()
        dds.refit()

        ds = DeseqStats(
            dds,
            contrast=contrast,
            alpha=alpha,
            cooks_filter=cooks,
            independent_filter=independent_filter,
            quiet=True, # otherwise it prints the summary table
            n_cpus=n_cpus
        )

        ds.run_wald_test()
        ds._cooks_filtering()
        ds._p_value_adjustment()
        ds.summary()

        df = ds.results_df.copy()

        return df
    
    @staticmethod
    def _aggregate_de_results(
        results: dict[str, pd.DataFrame],
        min_list_overlap: float,
    ) -> pd.DataFrame:
        """
        Aggregate DE results across repetitions.
        Keeps genes appearing in at least min_list_overlap fraction of runs
        and averages numeric columns.
        """
        n_runs = len(results)

        gene_counts = (
            pd.concat(
                [df.assign(_run=k) for k, df in results.items()],
                axis=0
            )
            .reset_index()
            .groupby("index")
            .size()
        )

        min_occurrences = int(np.ceil(min_list_overlap * n_runs))
        keep_genes = gene_counts[gene_counts >= min_occurrences].index

        all_results = pd.concat(results.values(), axis=0)
        all_results = all_results.loc[keep_genes]

        aggregated = (
            all_results
            .groupby(all_results.index)
            .mean(numeric_only=True)
        )

        #aggregated['is_de'] = aggregated['is_de'].astype(bool)

        return aggregated

    def _de(
            self,
            adata_sub: ad.AnnData,
            comparison_key: str, 
            group_names: Tuple[str, str],
            sample_hierarchy: dict,
            psbulk_replicate_key: str,
            batch_key: Optional[str],
            min_replicates: int,
            resampling_fraction: float,
            rng: np.random.default_rng,
            layer: str,
            mode: str,
            alpha: float,
            cooks: bool,
            fit_type: str,
            independent_filter: bool,
            n_repetitions: int,
            min_list_overlap: float,
            n_cpus: int,
            verbose: bool,
            ):
        # For the replicates that can be considered, generate a pseudobulk
        # If the batch key is not none, the replicates will be stratified by batch
        adata_pb = dc.pp.pseudobulk(
            adata_sub,
            sample_col=psbulk_replicate_key,
            groups_col=batch_key,
            layer=layer,
            mode=mode
        )
        adata_pb = adata_pb[
            (adata_pb.obs["psbulk_cells"] > 0)
            & (adata_pb.obs["psbulk_counts"] > 0)
        ].copy()

        # Determine how may replicates are needed to reach min_replicates per condition
        # If batch is considered, a replicate can be spread across multiple batches
        required_reps = {}
        for group in group_names:
            reps_dict = sample_hierarchy[group]
            if batch_key is not None:
                # count number of (replicate, batch) pairs for this condition
                current_reps = sum(
                    len(batches) for batches in reps_dict.values()
                )
            else:
                # count number of replicates (regardless of how many batches each has)
                current_reps = len(reps_dict)
            required_reps[group] = max(0, min_replicates - current_reps)

        # Separately determine the design formula:
        if batch_key is not None:
            design = f"~{comparison_key}+{batch_key}"
        else:
            design = f"~{comparison_key}"

        # If there are enough replicates in both groups, go to DE immediately:
        if all(v == 0 for v in required_reps.values()):
            adata_test = adata_pb.copy()
            counts = pd.DataFrame(
                adata_test.X,
                columns=adata_test.var_names,
                index=adata_test.obs_names
                )
            metadata = pd.DataFrame(
                adata_test.obs[[comparison_key] + ([batch_key] if batch_key is not None else [])],
                index=adata_test.obs_names
            )
            contrast = [comparison_key, group_names[0], group_names[1]]
            ds_results = self._pydeseq2_wrapper(
                counts=counts,
                metadata=metadata,
                design=design,
                contrast=contrast,
                alpha=alpha,
                cooks=cooks,
                fit_type=fit_type,
                independent_filter=independent_filter,
                n_cpus=n_cpus,
                verbose=verbose,
            )
            return ds_results
        
        # Else generate pseudoreplicates
        else:
            results = {}
            for i in range(n_repetitions):
                adata_pr = self._ensure_min_replicates(
                    adata_sub=adata_sub,
                    sample_hierarchy=sample_hierarchy,
                    psbulk_replicate_key=psbulk_replicate_key,
                    batch_key=batch_key,
                    required_reps=required_reps,
                    resampling_fraction=resampling_fraction,
                    rng=rng,
                    layer=layer,
                    mode=mode
                )
                adata_test = ad.concat([adata_pb, adata_pr], axis=0)
                counts = pd.DataFrame(
                    adata_test.X,
                    columns=adata_test.var_names,
                    index=adata_test.obs_names
                    )
                metadata = pd.DataFrame(
                    adata_test.obs[[comparison_key] + ([batch_key] if batch_key is not None else [])],
                    index=adata_test.obs_names
                )
                contrast = [comparison_key, group_names[0], group_names[1]]
                ds_results = self._pydeseq2_wrapper(
                    counts=counts,
                    metadata=metadata,
                    design=design,
                    contrast=contrast,
                    alpha=alpha,
                    cooks=cooks,
                    fit_type=fit_type,
                    independent_filter=independent_filter,
                    n_cpus=n_cpus,
                    verbose=verbose,
                )
                results[str(i)] = ds_results

            return self._aggregate_de_results(
                results=results,
                min_list_overlap=min_list_overlap
            )
        
def de_pairs_pseudobulk(
        adata_norm: ad.AnnData,
        cluster_assignments: Dict[Any, List],
        pairs: List[Tuple[int, int]],
        cl_means: pd.DataFrame,
        cl_present: pd.DataFrame,
        cl_size: Dict[Any, int],
        de_thresholds: Dict[str, Any],
        de_kwargs
    ):

    logger.info(f'Comparing {len(pairs)} pairs')
    de_pairs = {}
    for (cluster_a, cluster_b) in tqdm(pairs):

        idx_a = np.array(cluster_assignments[cluster_a], dtype=int)
        idx_b = np.array(cluster_assignments[cluster_b], dtype=int)
        idx = np.concatenate([idx_a, idx_b])

        adata = adata_norm[idx].copy()
        adata.obs['cluster_id'] = np.where(np.isin(idx, idx_a), 'cluster_a', 'cluster_b')

        de = BulkDE(
            adata=adata,
            group_key='cluster_id',
            query='cluster_a',
            reference='cluster_b',
            **de_kwargs
        )

        de_results = de.de
        de_results = de_results.reindex(cl_means.columns)

        # Get DE score
        de_pair_stats = pd.DataFrame(index=cl_means.columns)
        de_pair_stats['p_value'] = de_results['pvalue']
        de_pair_stats['p_adj'] = de_results['padj']
        de_pair_stats['lfc'] = de_results['log2FoldChange']
        de_pair_stats["meanA"] = cl_means.loc[cluster_a]
        de_pair_stats["meanB"] = cl_means.loc[cluster_b]
        de_pair_stats["q1"] = cl_present.loc[cluster_a]
        de_pair_stats["q2"] = cl_present.loc[cluster_b]
        de_pair_stats["qdiff"] = get_qdiff(cl_present.loc[cluster_a], cl_present.loc[cluster_b])

        de_pair_up = filter_gene_stats(
            de_stats=de_pair_stats,
            gene_type='up-regulated', 
            cl1_size=cl_size[cluster_a],
            cl2_size=cl_size[cluster_b],
            **de_thresholds
        )
        up_score = calc_de_score(de_pair_up['p_adj'].values)

        de_pair_down = filter_gene_stats(
            de_stats=de_pair_stats,
            gene_type='down-regulated',
            cl1_size=cl_size[cluster_a],
            cl2_size=cl_size[cluster_b],
            **de_thresholds
        )
        down_score = calc_de_score(de_pair_down['p_adj'].values)

        de_pairs[(cluster_a, cluster_b)] = {
            'score': up_score + down_score,
            'up_score': up_score,
            'down_score': down_score,
            'up_genes': de_pair_up.index.to_list(),
            'down_genes': de_pair_down.index.to_list(),
            'up_num': len(de_pair_up.index),
            'down_num': len(de_pair_down.index),
            'num': len(de_pair_up.index) + len(de_pair_down.index)
        }

    de_pairs = pd.DataFrame(de_pairs).T
    return de_pairs