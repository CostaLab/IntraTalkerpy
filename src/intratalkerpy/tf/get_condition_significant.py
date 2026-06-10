import pandas as pd
import decoupler as dc
import numpy as np
from re import sub
from statsmodels.stats.multitest import multipletests


def condition_comparison_significant(tf_activities, out_path, celltype, condition, comparison_list, num_cell_filter = 0):
    '''
    
    Description:
    --------------
    Creates a dataframe with all significant TFs for a pairwise comparison of two conditions given in the condition list.

    Parameters:
    --------------
    
    tf_activities : AnnData
        TF activities anndata object.

    out_path : str
        Output path to save results.

    comparison_list : list
        List of wished comparisons.

    num_cell_filter : int
        Minimum number of cells in each cell type.

    Returns:
    ---------
    vs_df_dic : dict[str, pandas.DataFrame] 
        Dictionary with one dataframe per comparison containing all marker TFs found in condition comparison analysis with respective condition, reference, tf, scores, meanchange, pvals, pvals_adj, CellType, FDR, r and significance tag.
    '''

    vs_df_dic = {}

    if isinstance(comparison_list[0], str):  
        comparison_list = [comparison_list]
    
    for vs1, vs2 in comparison_list:

        # #replaces special symbols in condition names with _
        renamed_vs1 =  sub("([,;.:-])", "_", vs1)
        renamed_vs2 =  sub("([,;.:-])", "_", vs2)
        print(f"vs1: {renamed_vs1}, vs2: {renamed_vs2}") 

        res = pd.DataFrame()

        #subsets to one cell type for condition comparison
        for i in tf_activities.obs[celltype].unique(): 
            comparison_sub = tf_activities[(tf_activities.obs[celltype] == i) & (tf_activities.obs[condition].isin([vs1, vs2]))]
            if len(pd.unique(comparison_sub.obs[condition])) == 2:
                condition_table = comparison_sub.obs[[condition]].copy()
                condition_table.columns = ["condition"]
                metadata_counts = condition_table.groupby("condition", observed = False).size()
                
                #checks if number of cells is above minimum given in arguments list
                if (metadata_counts.iloc[0] + metadata_counts.iloc[1]) > num_cell_filter:
                    g = comparison_sub.obs[condition].astype("category")
                    g = g.cat.set_categories([vs1, vs2])
        
                    #finding markers for the condition comparison
                    res_tmp_new_format = dc.tl.rankby_group(comparison_sub, groupby= condition, reference="rest", method="t-test")
                    res_tmp = (
                        res_tmp_new_format.rename(
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
                    res_tmp.rename(columns={"group": "condition", "statistic" : "scores"}, inplace=True)
            
                    #calculating wilcoxon scores to use for r value calculation (used in heatmaps)
                    res_heatmap_new_format = dc.tl.rankby_group(comparison_sub, groupby= condition, reference="rest", method="wilcoxon")
                    res_heatmap = (
                        res_heatmap_new_format.rename(
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
                    res_heatmap.rename(columns={"group": "condition", "statistic" : "scores"}, inplace=True)

                    #subsetting into both conditions to calculate r value
                    group1 = comparison_sub.X[g == vs1]
                    group2 = comparison_sub.X[g == vs2]
                        
                    res_heatmap["r"] = (res_heatmap["scores"] / np.sqrt(len(group1) + len(group2)))
                    res_heatmap["CellType"] = i
                    res_tmp["CellType"] = i
                    _, res_tmp["FDR"], _, _ = multipletests(res_tmp["pvals"], alpha=0.05, method='fdr_bh')
                    
                    #subsetting for unique gene indices due to results being symmetrical 
                    res_tmp = res_tmp[res_tmp["condition"] == vs1]

                    res_heatmap = res_heatmap[["names", "r", "CellType", "condition"]]
                    res_tmp = pd.merge(res_tmp, res_heatmap, on = ["names","CellType", "condition"])

                    res = pd.concat([res, res_tmp], ignore_index=True)

        res_df = res.dropna()

        def assign_significance_tag(fdr):
            if fdr < 0.001:
                return "***"
            elif fdr < 0.01:
                return "**"
            elif fdr < 0.05:
                return "*"
            else:
                return "ns"

        res_df = res_df.assign(tag=res_df["FDR"].apply(assign_significance_tag))
        res_df.rename(columns={"names":"tf", "group": "condition"}, inplace=True)

        res_df.to_csv(f"{out_path}/all_tfs_{renamed_vs1}_vs_{renamed_vs2}.csv", index=False)

        result_name = f"{renamed_vs1}_{renamed_vs2}"
        vs_df_dic[result_name] = res_df

    return vs_df_dic