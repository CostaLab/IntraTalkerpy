import numpy as np
import pandas as pd
import decoupler as dc
import os
from re import sub
from .utils import (eval_pval, eval_meanchange_tag, create_unfiltered_tf_scores, save_variable_tf_score, map_t_value)
from .plot import plot_tf_activity


def get_significant_tfs(tf_activities_sub, condition, out_path, tf_condition_significant, celltype, pval, meanchange, plot = True, condition_comparison = False):
    
    '''
    Description:
    -------------
    Analysis transcription factor activities for significant transcription factors.
   
    Parameters:
    -------------
    tf_activities_sub : AnnData
        Input Anndata Object with TF activities as X matrix.
        
    condition : str 
        Experiment condition (e.g. disease, knockout ...).
        
    out_path : str 
        Output path to save results.

    tf_condition_significant : pandas.DataFrame
        Condition comparison results.

    celltype : str
        Variable that accesses celltype meta data in anndata object.

    pval : float
        P-value to filter results.
        
    meanchange : float
        Meanchange value to filter results.
        
    plot : bool 
        Boolean variable to decide if code plots heatmaps or not.

    condition_comparison : bool
        Comparision between multiple conditions or single condition.

    Returns:
    ---------
    res : dict[str, pandas.DataFrame]
        A data frame with transcription factor activity scores per cell type. 
    '''
    
    #create directory
    single_result_path = os.path.join(out_path, condition)
    os.makedirs(single_result_path, exist_ok=True)

    number_of_clusters = (len(tf_activities_sub.obs[celltype].unique()))

    #get marker TFs for single condition 
    anndataobject_markers_new_format = dc.tl.rankby_group(tf_activities_sub, groupby= celltype, reference="rest", method="t-test")
    anndataobject_markers = (
        anndataobject_markers_new_format.rename(
            columns={
                "name": "names",
                "stat": "statistic",
                "pval": "pvals",
                "padj": "pvals_adj",
            }
        )
        .astype(
            {
                "group": str,
                "reference": str,
                "names": str,
            }
        )
    )
    anndataobject_markers.rename(columns={"names" : "gene", "group": "cluster", "statistic" : "scores"}, inplace=True)
    
    anndataobject_markers["tag"] = None
    anndataobject_markers["meanchange_tag"] = None
    
    #add significance tags to p-value and meanchange
    anndataobject_markers["tag"] = anndataobject_markers["pvals_adj"].apply(eval_pval)
    anndataobject_markers["meanchange_tag"] = anndataobject_markers["meanchange"].apply(eval_meanchange_tag)
      
    anndataobject_markers.to_csv(os.path.join(single_result_path, f"{condition}_specific_markers_t_test.csv"), index=False)

    #using the custom pval and meanchange filters to create tag mapping for significant tfs
    clusters = sorted(anndataobject_markers["cluster"].unique())
    tag_mapping = anndataobject_markers[["gene", "tag", "meanchange_tag", "cluster", "pvals_adj", "meanchange"]]
    tag_mapping = tag_mapping[(tag_mapping["pvals_adj"] < float(pval))] 
    tag_mapping = tag_mapping[(tag_mapping["meanchange"] > float(meanchange)) | 
                              (tag_mapping["meanchange"] < -float(meanchange))]

    tag_mapping = tag_mapping.pivot(index="gene", columns="cluster", values="tag")

    for cluster in clusters:
        if cluster not in tag_mapping.columns:
            tag_mapping[cluster] = np.nan

    tag_mapping = tag_mapping[clusters]
    tag_mapping = tag_mapping.astype("object")
    tag_mapping.fillna("ns", inplace=True)

    #creating df with genes as index and tf activity scores summarized by celltype/cluster (columns)
    tf_activities_sub.obs_names = tf_activities_sub.obs[celltype].astype(str)
    tf_scores_df = tf_activities_sub.to_df()

    unfiltered_tf_scores = create_unfiltered_tf_scores(tf_scores_df, condition, celltype, single_result_path)
   
    #Filter to only include tfs that match the tag_mapping/are markers
    col_num = tf_scores_df.columns.isin(tag_mapping.index)  
    filtered_tf_scores_df = tf_scores_df.loc[:, col_num]
    filtered_summarized_tf_scores_df = filtered_tf_scores_df.groupby(celltype, observed = False).mean().T
    filtered_summarized_tf_scores_df.sort_index(axis=1, inplace=True)
    filtered_summarized_tf_scores_df.index.name = "gene"
    filtered_summarized_tf_scores_df.to_csv(os.path.join(single_result_path, f"tf_scores_{condition}.csv"))

    #includes variability value of a gene's tf activity over all celltypes/clusters
    tf_scores_variable_table = save_variable_tf_score(filtered_summarized_tf_scores_df, condition, single_result_path, plot)

    #plots heatmap
    if plot:
        plot_tf_activity(filtered_summarized_tf_scores_df, tag_mapping, condition, single_result_path)
    
    filtered_summarized_tf_scores_df.index = filtered_summarized_tf_scores_df.index.map(lambda x: sub(".,", "_", x))

    #creates df of significant tfs per cluster
    #merges the tf score df with the marker tf df so that only the tf scores of tfs classified as significant by decoupler rank sources groups function are included
    anndataobject_markers_merged = anndataobject_markers.merge(map_t_value(filtered_summarized_tf_scores_df, anndataobject_markers),  on=['gene', 'cluster'], how='inner')
    anndataobject_markers_merged = anndataobject_markers_merged[anndataobject_markers_merged.tag != "ns"]
    anndataobject_markers_merged = anndataobject_markers_merged.where(anndataobject_markers_merged["t_value"] > 0, np.nan) 
    anndataobject_markers_merged.dropna(inplace=True)
    
    res_tmp = anndataobject_markers_merged[["gene","tag", "cluster", "t_value"]]
    res_tmp.to_csv(single_result_path + "/significant_cluster_tf_results_" + condition + ".csv", index=0) # type: ignore
    
    res = {}
    res["cluster"] = res_tmp
    
    #creates df of significant tfs per condition
    if condition_comparison:
        unfiltered_tf_scores = unfiltered_tf_scores.where(unfiltered_tf_scores > 0, np.nan) 
        tf_condition_significant["gene"] = tf_condition_significant["gene"].apply(lambda x: sub(".,", "_", x))
        tf_condition_significant = tf_condition_significant.merge(map_t_value(unfiltered_tf_scores, tf_condition_significant), left_on=None, right_on=None, left_index=False, right_index=False)
        tf_condition_significant.dropna(inplace=True)
    
        res["condition"] = tf_condition_significant
        res["condition"].to_csv(f"{single_result_path}/significant_condition_tf_results_{condition}.csv", index=0)

    return res