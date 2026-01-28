import anndata as ad
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from .diff_expression import get_qdiff, filter_gene_stats, calc_de_score

import logging
logger = logging.getLogger(__name__)

import scbulkde as scb
from scbulkde.ut._logging import set_log_level
set_log_level("CRITICAL")

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

        de = scb.tl.de(
            data=adata,
            group_key='cluster_id',
            query='cluster_a',
            reference='cluster_b',
            **de_kwargs
        )

        de_results = de.results
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