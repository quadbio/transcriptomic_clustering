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
    assessment of batch inclusion, and generation of pseudoreplicates for DE analysis.
    """

    def __init__(
        self,
        adata:  ad.AnnData,
        group_key: str,
        query:  str,
        reference: Union[str, List[str]],
        group_names: Tuple[str, str] = ("query", "reference"),
        comparison_key_added: str = "_comparison_group",
        replicate_key:  Optional[str] = None,
        batch_key: Optional[str] = None,
        sample_min_cells: int = 50,
        sample_min_fraction: float = 0.3,
        min_coverage: float = 0.8,
        min_bridging_batches: int = 2,

        # Decoupler arguments
        layer: str = 'counts',
        mode: str = 'sum',

        # Pseudoreplicate arguments
        min_samples:  int = 5,
        resampling_fraction: float = 0.6,
        n_repetitions: int = 10,
        min_list_overlap: float = 0.8,
        
        # DE arguments
        alpha:  float = 0.05,
        cooks: bool = True,
        independent_filter: bool = True,
        fit_type: str = "mean",
 
        # Other arguments
        n_cpus:  int = 1,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the BulkDE analysis. 
        
        Parameters
        ----------
        adata :  AnnData
            Annotated data matrix
        group_key : str
            Column in adata.obs containing group labels
        query : str
            Value in group_key for the query group
        reference : str or list of str
            Value(s) in group_key for reference group, or "rest"
        group_names : tuple
            Names for (query, reference) in output
        comparison_key_added :  str
            Column name to add for comparison groups
        replicate_key : str, optional
            Column for biological replicates
        batch_key : str, optional
            Column for batch information
        sample_min_cells : int
            Minimum cells for a sample to be valid
        sample_min_fraction :  float
            Minimum fraction of condition cells for a sample to be valid
        min_coverage : float
            Minimum fraction of cells that must be covered by valid samples
        min_bridging_batches : int
            Minimum batches present in both conditions to include batch in design
        layer : str
            Layer to use for pseudobulk
        mode : str
            Aggregation mode for pseudobulk
        min_samples : int
            Minimum samples per condition for DE
        resampling_fraction :  float
            Fraction of cells to sample when generating pseudoreplicates
        n_repetitions : int
            Number of repetitions when using pseudoreplicates
        min_list_overlap : float
            Minimum fraction of repetitions a gene must appear in
        alpha : float
            Significance threshold
        cooks : bool
            Whether to apply Cook's distance filtering
        independent_filter : bool
            Whether to apply independent filtering
        fit_type :  str
            Dispersion fit type for DESeq2
        n_cpus : int
            Number of CPUs for parallel processing
        seed : int
            Random seed
        verbose : bool
            Whether to print progress information
        """
        self.adata = adata
        self.group_key = group_key
        self.query = query
        self.reference = reference
        self.group_names = group_names
        self.comparison_key_added = comparison_key_added
        self.replicate_key = replicate_key
        self.batch_key = batch_key
        self.sample_min_cells = sample_min_cells
        self.sample_min_fraction = sample_min_fraction
        self.min_coverage = min_coverage
        self.min_bridging_batches = min_bridging_batches

        self.layer = layer
        self. mode = mode

        self.min_samples = min_samples
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

        self.rng = np.random.default_rng(self.seed)

        # Step 1: Select cells for the two conditions
        self. adata_sub = self._select_conditions(
            adata=self. adata,
            group_key=self.group_key,
            query=self.query,
            reference=self.reference,
            group_names=self.group_names,
            comparison_key_added=self.comparison_key_added,
        )

        if self. adata_sub is None or self.adata_sub.n_obs == 0:
            raise ValueError("Condition selection failed.  No cells found.")

        # Step 2: Identify samples and determine design
        (
            self.adata_sub,
            self.sample_hierarchy,
            self.sample_info,
            self.design,
            self.include_batch,
        ) = self._identify_samples_and_design(
            adata_sub=self. adata_sub,
            comparison_key=self.comparison_key_added,
            group_names=self.group_names,
            replicate_key=self. replicate_key,
            batch_key=self.batch_key,
            min_cells=self.sample_min_cells,
            min_fraction=self.sample_min_fraction,
            min_coverage=self.min_coverage,
            min_bridging_batches=self.min_bridging_batches,
            verbose=self.verbose,
        )

        # Determine the sample key used for pseudobulking
        self.sample_key = self. sample_info["sample_key_used"]
        
        # Step 3: Run DE analysis
        self. de = self._run_de(
            adata_sub=self. adata_sub,
            comparison_key=self.comparison_key_added,
            group_names=self.group_names,
            sample_hierarchy=self.sample_hierarchy,
            sample_key=self.sample_key,
            batch_key=self.batch_key if self.include_batch else None,
            design=self.design,
            min_samples=self.min_samples,
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
        """Return AnnData subset containing query vs.  reference groups."""
        
        mask_query = adata.obs[group_key] == query
        if mask_query.sum() == 0:
            logger.error(f"No cells found for query group '{query}' in '{group_key}'.")
            return None

        if isinstance(reference, str):
            mask_reference = (
                adata.obs[group_key] != query if reference == "rest"
                else adata.obs[group_key] == reference
            )
        else:
            mask_reference = adata.obs[group_key]. isin(reference)

        if mask_reference.sum() == 0:
            logger.error(f"No cells found for reference group '{reference}' in '{group_key}'.")
            return None

        mask = mask_query | mask_reference
        adata_sub = adata[mask]. copy()

        adata_sub.obs[comparison_key_added] = np.where(
            mask_query[mask], group_names[0], group_names[1]
        )

        return adata_sub

    @staticmethod
    def _identify_samples_and_design(
        adata_sub: ad.AnnData,
        comparison_key: str,
        group_names:  Tuple[str, str],
        replicate_key: Optional[str],
        batch_key: Optional[str],
        min_cells: int,
        min_fraction: float,
        min_coverage: float,
        min_bridging_batches: int,
        verbose: bool,
    ) -> Tuple[ad.AnnData, Dict, Dict, str, bool]:
        """
        Identify samples (replicates or batches) and determine design formula. 
        
        Logic:
        - If replicate_key provided:  use as biological replicates, optionally stratify by batch
        - If only batch_key provided: use batches as technical replicates, check for bridging
        - If neither:  collapse each condition into single sample
        
        Returns:
            adata_sub: Modified AnnData with updated sample assignments
            sample_hierarchy: {condition -> sample_id -> batch_id -> [obs_names]}
            info: Dictionary with validation details
            design: Design formula string
            include_batch: Whether batch is included in design
        """
        adata_sub = adata_sub.copy()
        obs = adata_sub.obs

        # Determine sample key and whether to stratify by batch
        if replicate_key is not None: 
            sample_key = replicate_key
            stratify_by_batch = batch_key is not None
            batch_is_sample = False
        elif batch_key is not None: 
            sample_key = batch_key
            stratify_by_batch = False
            batch_is_sample = True
        else:
            sample_key = None
            stratify_by_batch = False
            batch_is_sample = False

        # Create internal sample column
        internal_sample_key = "_de_sample"
        if sample_key is not None:
            adata_sub.obs[internal_sample_key] = (
                obs[comparison_key]. astype(str) + "_" + obs[sample_key].astype(str)
            )

        sample_hierarchy = {}
        valid_samples_by_condition = {}
        collapsed_conditions = []

        for cond in group_names:
            cond_mask = adata_sub.obs[comparison_key] == cond
            cond_cells = adata_sub.obs[cond_mask]
            cond_total = len(cond_cells)

            if sample_key is None:
                # No sample info - collapse entire condition
                collapsed_id = f"{cond}_collapsed"
                sample_hierarchy[cond] = {collapsed_id: {"batch_1": cond_cells.index. tolist()}}
                valid_samples_by_condition[cond] = [collapsed_id]
                collapsed_conditions.append(cond)
                adata_sub. obs. loc[cond_mask, internal_sample_key] = collapsed_id
                continue

            # Count cells per sample
            sample_counts = cond_cells.groupby(internal_sample_key, observed=True).size()
            sample_fractions = sample_counts / cond_total

            # Identify valid samples
            valid_mask = (sample_counts >= min_cells) | (sample_fractions >= min_fraction)
            valid_samples = sample_counts[valid_mask]. index.tolist()

            # Check coverage
            if valid_samples: 
                coverage = sample_counts[valid_samples].sum() / cond_total
                if coverage < min_coverage:
                    valid_samples = []

            # Collapse if no valid samples
            if not valid_samples:
                collapsed_id = f"{cond}_collapsed"
                valid_samples = [collapsed_id]
                collapsed_conditions.append(cond)
                adata_sub.obs.loc[cond_mask, internal_sample_key] = collapsed_id
                cond_cells = adata_sub.obs[cond_mask]

            valid_samples_by_condition[cond] = valid_samples

            # Build hierarchy for this condition
            sample_hierarchy[cond] = {}
            for sample_id in valid_samples:
                sample_cells = cond_cells[cond_cells[internal_sample_key] == sample_id]

                if stratify_by_batch and cond not in collapsed_conditions:
                    sample_hierarchy[cond][sample_id] = {}
                    for batch_val, batch_group in sample_cells.groupby(batch_key, observed=True):
                        if len(batch_group) > 0:
                            sample_hierarchy[cond][sample_id][batch_val] = batch_group. index.tolist()
                else:
                    # When batch_is_sample, store the original batch name for bridging check
                    if batch_is_sample and cond not in collapsed_conditions: 
                        # Extract original batch name from internal sample key (remove condition prefix)
                        original_batch = sample_id.replace(f"{cond}_", "", 1)
                        sample_hierarchy[cond][sample_id] = {original_batch: sample_cells.index.tolist()}
                    else:
                        sample_hierarchy[cond][sample_id] = {"batch_1": sample_cells.index.tolist()}

        # Filter adata_sub to only include cells in valid samples
        all_valid_samples = [s for samples in valid_samples_by_condition.values() for s in samples]
        adata_sub = adata_sub[adata_sub.obs[internal_sample_key].isin(all_valid_samples)].copy()

        # Determine if batch should be included in design
        include_batch = False
        bridging_batches = []

        # Case 1: replicate_key provided with batch_key (stratify by batch)
        if stratify_by_batch and batch_key is not None:
            batches_per_cond = {}
            for cond in group_names:
                if cond not in collapsed_conditions:
                    batches_per_cond[cond] = set(
                        b for samples in sample_hierarchy[cond].values()
                        for b in samples. keys()
                    )
                else:
                    batches_per_cond[cond] = set()

            bridging_batches = list(
                batches_per_cond[group_names[0]] & batches_per_cond[group_names[1]]
            )
            include_batch = len(bridging_batches) >= min_bridging_batches

        # Case 2: only batch_key provided (batch IS the sample)
        elif batch_is_sample and batch_key is not None:
            # Check bridging:  which original batch names appear in both conditions
            batches_per_cond = {}
            for cond in group_names:
                if cond not in collapsed_conditions:
                    # Extract batch names from hierarchy (stored as the single key in each sample's dict)
                    batches_per_cond[cond] = set(
                        batch_name
                        for sample_dict in sample_hierarchy[cond]. values()
                        for batch_name in sample_dict.keys()
                    )
                else:
                    batches_per_cond[cond] = set()

            bridging_batches = list(
                batches_per_cond[group_names[0]] & batches_per_cond[group_names[1]]
            )
            include_batch = len(bridging_batches) >= min_bridging_batches

        # Determine design formula
        if include_batch:
            design = f"~{comparison_key}+{batch_key}"
        else: 
            design = f"~{comparison_key}"

        info = {
            "valid_samples_by_condition": valid_samples_by_condition,
            "collapsed_conditions": collapsed_conditions,
            "sample_key_used": internal_sample_key,
            "original_sample_key": sample_key,
            "batch_key_used": batch_key if include_batch else None,
            "bridging_batches": bridging_batches,
            "is_technical_replicates": batch_is_sample,
        }

        if verbose:
            print(f"Design formula: {design}")
            print(f"Valid samples: {valid_samples_by_condition}")
            if collapsed_conditions:
                print(f"Collapsed conditions: {collapsed_conditions}")
            if info["is_technical_replicates"]: 
                print("Note: Using batches as technical replicates")
            if bridging_batches:
                print(f"Bridging batches: {bridging_batches}")

        return adata_sub, sample_hierarchy, info, design, include_batch

    def _generate_pseudoreplicate(
        self,
        adata_sub: ad. AnnData,
        obs_names: List[str],
        sample_key: str,
        batch_key: Optional[str],
        sampling_fraction: float,
        rng: np.random.Generator,
        layer: str,
        mode: str,
    ) -> ad.AnnData:
        """Generate a single pseudoreplicate by sampling cells."""
        n_sample = max(1, int(len(obs_names) * sampling_fraction))
        sampled_cells = rng.choice(obs_names, size=n_sample, replace=False)
        adata_sampled = adata_sub[sampled_cells].copy()

        return dc.pp.pseudobulk(
            adata_sampled,
            sample_col=sample_key,
            groups_col=batch_key,
            layer=layer,
            mode=mode,
        )

    def _generate_pseudoreplicates(
        self,
        adata_sub: ad.AnnData,
        sample_hierarchy: Dict,
        sample_key: str,
        batch_key: Optional[str],
        required_samples: Dict[str, int],
        resampling_fraction: float,
        rng: np.random. Generator,
        layer: str,
        mode: str,
    ) -> ad.AnnData:
        """Generate additional pseudoreplicates to meet minimum sample requirements."""
        adata_list = []

        for condition, samples in sample_hierarchy.items():
            n_needed = required_samples[condition]
            if n_needed <= 0:
                continue

            sample_ids = list(samples.keys())

            for i in range(n_needed):
                # Randomly select a sample to resample from
                source_sample = rng.choice(sample_ids)
                batches = samples[source_sample]

                if batch_key is not None and len(batches) > 1:
                    # Sample from a random batch within the sample
                    source_batch = rng.choice(list(batches.keys()))
                    obs_names = batches[source_batch]
                else:
                    # Sample from all cells in the sample
                    obs_names = [cell for cells in batches.values() for cell in cells]

                adata_pr = self._generate_pseudoreplicate(
                    adata_sub=adata_sub,
                    obs_names=obs_names,
                    sample_key=sample_key,
                    batch_key=batch_key,
                    sampling_fraction=resampling_fraction,
                    rng=rng,
                    layer=layer,
                    mode=mode,
                )

                # Rename to indicate pseudoreplicate
                new_sample_id = f"{source_sample}_pr_{i+1}"
                adata_pr.obs[sample_key] = new_sample_id
                adata_pr. obs_names = [f"{idx}_pr_{i+1}" for idx in adata_pr.obs_names]
                adata_list.append(adata_pr)

        return ad.concat(adata_list, axis=0) if adata_list else None

    def _pydeseq2_wrapper(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        design: str,
        contrast: List[str],
        alpha: float,
        cooks:  bool,
        fit_type:  str,
        independent_filter:  bool,
        n_cpus: int,
        verbose:  bool,
    ) -> pd.DataFrame:
        """Run PyDESeq2 differential expression analysis."""
        inference = DefaultInference(n_cpus=n_cpus)

        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design=design,
            inference=inference,
            refit_cooks=cooks,
            fit_type=fit_type,
            quiet=not verbose,
        )

        dds. fit_size_factors()
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
            quiet=True,
            n_cpus=n_cpus,
        )

        ds.run_wald_test()
        ds._cooks_filtering()
        ds._p_value_adjustment()
        ds. summary()

        return ds.results_df. copy()

    @staticmethod
    def _aggregate_de_results(
        results: Dict[str, pd.DataFrame],
        min_list_overlap: float,
    ) -> pd.DataFrame:
        """Aggregate DE results across repetitions."""
        n_runs = len(results)
        min_occurrences = int(np.ceil(min_list_overlap * n_runs))

        # Count gene occurrences across runs
        gene_counts = (
            pd.concat([df.reset_index() for df in results.values()])
            .groupby("index")
            .size()
        )
        keep_genes = gene_counts[gene_counts >= min_occurrences].index

        # Average results for kept genes
        all_results = pd.concat(results.values())
        all_results = all_results. loc[all_results.index.isin(keep_genes)]

        return all_results.groupby(all_results.index).mean(numeric_only=True)

    def _run_de(
        self,
        adata_sub: ad.AnnData,
        comparison_key: str,
        group_names: Tuple[str, str],
        sample_hierarchy: Dict,
        sample_key: str,
        batch_key: Optional[str],
        design: str,
        min_samples: int,
        resampling_fraction: float,
        rng: np.random.Generator,
        layer: str,
        mode: str,
        alpha: float,
        cooks:  bool,
        fit_type:  str,
        independent_filter:  bool,
        n_repetitions: int,
        min_list_overlap: float,
        n_cpus: int,
        verbose: bool,
    ) -> pd.DataFrame:
        """Run differential expression analysis with optional pseudoreplicate generation."""
        
        # Generate pseudobulk
        adata_pb = dc.pp.pseudobulk(
            adata_sub,
            sample_col=sample_key,
            groups_col=batch_key,
            layer=layer,
            mode=mode,
        )
        adata_pb = adata_pb[
            (adata_pb.obs["psbulk_cells"] > 0) & (adata_pb.obs["psbulk_counts"] > 0)
        ].copy()

        if verbose:
            print(f"Pseudobulk samples: {adata_pb.n_obs}")

        # Determine how many additional samples are needed per condition
        required_samples = {}
        for group in group_names:
            if batch_key is not None: 
                # Count (sample, batch) pairs
                current = sum(len(batches) for batches in sample_hierarchy[group].values())
            else:
                current = len(sample_hierarchy[group])
            required_samples[group] = max(0, min_samples - current)

        if verbose:
            print(f"Additional samples needed: {required_samples}")

        # Prepare contrast
        contrast = [comparison_key, group_names[0], group_names[1]]
        metadata_cols = [comparison_key] + ([batch_key] if batch_key else [])

        # If enough samples, run DE directly
        if all(v == 0 for v in required_samples.values()):
            counts = pd.DataFrame(
                adata_pb.X,
                columns=adata_pb.var_names,
                index=adata_pb.obs_names,
            )
            metadata = adata_pb.obs[metadata_cols]. copy()

            return self._pydeseq2_wrapper(
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

        # Otherwise, run with pseudoreplicates
        results = {}
        for i in range(n_repetitions):
            adata_pr = self._generate_pseudoreplicates(
                adata_sub=adata_sub,
                sample_hierarchy=sample_hierarchy,
                sample_key=sample_key,
                batch_key=batch_key,
                required_samples=required_samples,
                resampling_fraction=resampling_fraction,
                rng=rng,
                layer=layer,
                mode=mode,
            )

            if adata_pr is not None: 
                adata_test = ad.concat([adata_pb, adata_pr], axis=0)
            else:
                adata_test = adata_pb

            counts = pd.DataFrame(
                adata_test.X,
                columns=adata_test.var_names,
                index=adata_test.obs_names,
            )
            metadata = adata_test.obs[metadata_cols].copy()

            results[str(i)] = self._pydeseq2_wrapper(
                counts=counts,
                metadata=metadata,
                design=design,
                contrast=contrast,
                alpha=alpha,
                cooks=cooks,
                fit_type=fit_type,
                independent_filter=independent_filter,
                n_cpus=n_cpus,
                verbose=False,
            )

        return self._aggregate_de_results(results=results, min_list_overlap=min_list_overlap)
        
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

        print(de_kwargs)

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