from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def h_clust(data):
            
            '''
            Description:
            ------------
            Hierarchichal clustering of TF scores. 
            
            Parameters:
            ------------
            data : pandas.DataFrame 
                Dataframe with TF scores.

            Returns:
            ---------
            linkage_matrix : 
               Hierarchically clustered TF scores.
            '''

            dist_matrix = pdist(data.T)
            linkage_matrix = linkage(dist_matrix, method = "average")   
            return linkage_matrix

def plot_condition_tf_activities(tf_activity_tables, out_path, which = "both"):
   
   '''
   Description:
   -------------
   Generates cluster and condition heatmap with r effect size only for significant genes.
   Plots heatmap of significant TFs and their activity scores from the compared condition analysis.
   
   Parameters:
   -----------
   tf_activity_tables: pandas.DataFrame 
        Dataframe with TF scores.

   out_path: str
        Output path for results.

   which : str
        Which heatmaps to plot ("annotated", "compressed" or "both").

   Returns:
   ---------    
   None
   '''
   if "both" or "annotated":
     #annotation is the tag mapping
     for result_name, df in tf_activity_tables.items():
          #initialize the df for current result
          name_df = pd.DataFrame(df)

          #filter for significant results
          significant_res = name_df[name_df["tag"] == "***"]
          significant_genes = np.unique(significant_res["tf"])
          name_df = name_df[name_df['tf'].isin(significant_genes)]


          tag_mapping = name_df[["tf", "tag", "CellType"]]

          tag_mapping = tag_mapping.pivot(index="tf", columns="CellType", values="tag")
          tag_mapping.fillna("ns", inplace=True)

          #drop columns that only contain ns
          cols_all_ns = tag_mapping.columns[(tag_mapping == "ns").all()]
          tag_mapping = tag_mapping.drop(columns=cols_all_ns)

          #subset df and cluster hierarchically
          name_df_r = name_df[["r", "tf", "CellType"]]
          name_df_cluster = name_df_r.pivot_table(index = "tf", columns = "CellType", values = "r", aggfunc = "mean")
          name_df_cluster.fillna(0, inplace=True) 
          name_df_cluster = name_df_cluster.drop(columns=cols_all_ns)
          h_clust_matrix = h_clust(name_df_cluster)

          #calculate size of clustermap
          calc_size = ((len(name_df["CellType"].unique()) * 1.5), (len(name_df_cluster) * 0.2))
          
          #plot clustermap
          cluster_map = sns.clustermap(name_df_cluster, cbar_kws={"label": "r"}, figsize=calc_size, cmap="vlag", center=0, annot= tag_mapping,
                         col_linkage= h_clust_matrix, fmt="", yticklabels=True, cbar_pos=(1, 0.5, 0.03, 0.05), dendrogram_ratio=(0.2, 0.05))
          plt.setp(cluster_map.ax_heatmap.get_xticklabels(), rotation=45)  
          plt.savefig(out_path + "/" + result_name + "_cluster_condition_activity_difference.pdf", bbox_inches='tight')
          plt.close()

   if "both" or "compressed":
        #no annotation
        for result_name, df in tf_activity_tables.items():
            #initialize the df for current result
            name_df = pd.DataFrame(df)
            
            #filter for significant results
            significant_res = name_df[name_df["tag"] == "***"]
            significant_genes = np.unique(significant_res["tf"])
            name_df = name_df[name_df['tf'].isin(significant_genes)]
            
            #subset df
            name_df_r = name_df[["r", "tf", "CellType"]]
            name_df_cluster = name_df_r.pivot_table(index = "tf", columns = "CellType", values = "r", aggfunc = "mean")
            name_df_cluster.fillna(0, inplace=True) 

            #hierarchical clustering
            h_clust_matrix = h_clust(name_df_cluster)
            name_df_cluster.reset_index()
          
            #calculate size of clustermap
            calc_size = ((len(name_df["CellType"].unique()) * 1.5
            ), (len(name_df_cluster) * 0.06))

            #plot clustermap
            cluster_map = sns.clustermap(name_df_cluster, cbar_kws={"label": "r"}, figsize=calc_size, cmap="vlag", center=0, annot= None,
                        yticklabels=False, col_linkage= h_clust_matrix, fmt="", cbar_pos=(1, 0.5, 0.02, 0.1), dendrogram_ratio=(0.2, 0.05))
            plt.setp(cluster_map.ax_heatmap.get_xticklabels(), rotation=45) 
            
            plt.savefig(out_path + "/" + result_name +  "_cluster_condition_activity_difference_compressed.pdf", bbox_inches='tight')
            plt.close()



def plot_tf_activity(filtered_summarized_tf_scores_df, tag_mapping, condition, out_path, which = "both"):

    '''
    Description:
    -------------
    Plots the tf activity t-values with respective genes for cluster analysis as annotated and compressed heatmaps and saves them as separate PDFs.

    Parameters:
    ------------
    filtered_summarized_tf_scores_df : pandas.DataFrame
        DataFrame with TF scores.
    
    tag_mapping : pandas.DataFrame
        Significance tag annotation DataFrame of the same size as the tf score DataFrame.

    condition : str
        Sample condition for file naming(e.g. control, disease ...)

    out_path : str
        Output path to save results.

    which : str
        Which heatmaps to plot ("annotated", "compressed" or "both").

    Returns:
    --------
    None
    '''

    if "both" or "annotated":
          #annotation is the tag mapping
          cols_all_ns = tag_mapping.columns[(tag_mapping == "ns").all()]
          tag_mapping = tag_mapping.drop(columns=cols_all_ns)
          filtered_summarized_tf_scores_df = filtered_summarized_tf_scores_df.drop(columns=cols_all_ns)

          #calculate size of clustermap
          calc_size = ((len(filtered_summarized_tf_scores_df.columns.unique()) * 1.5), (len(filtered_summarized_tf_scores_df) * 0.2))
          
          #plot clustermap
          cluster_map = sns.clustermap(filtered_summarized_tf_scores_df, cbar_kws={"label": "t-score"}, figsize=calc_size, cmap="vlag", center=0, annot= tag_mapping, fmt="", 
                                        yticklabels=True, cbar_pos=(1, 0.5, 0.03, 0.05), dendrogram_ratio=(0.2, 0.05))
          cluster_map.ax_heatmap.set_xlabel("Cell Type")
          cluster_map.ax_heatmap.set_ylabel("Transcription Factor")
          plt.setp(cluster_map.ax_heatmap.get_xticklabels(), rotation=45) 

          plt.savefig(out_path + "/tf_activity_" + condition + ".pdf", bbox_inches='tight')
          plt.close()

    if "both" or "compressed":
          #no annotation
          calc_size = ((len(filtered_summarized_tf_scores_df.columns.unique()) * 1.5), (len(filtered_summarized_tf_scores_df) * 0.06))
          cluster_map = sns.clustermap(filtered_summarized_tf_scores_df, cbar_kws={"label": "t-value"}, figsize=calc_size, cmap="vlag", center=0, 
                                        yticklabels=False, cbar_pos=(1, 0.5, 0.02, 0.1), dendrogram_ratio=(0.2, 0.05))
          cluster_map.ax_heatmap.set_xlabel("Cell Type")
          cluster_map.ax_heatmap.set_ylabel("Transcription Factor")
          plt.setp(cluster_map.ax_heatmap.get_xticklabels(), rotation=45) 
          
          plt.savefig(out_path + "/tf_activity_compressed_" + condition + ".pdf", bbox_inches='tight')
          plt.close()

def plot_top_variable_tfs(filtered_summarized_tf_scores_df_var, condition, out_path):
     '''
     Description:
     -------------
     Plots the top 20 variable TFs as a heatmap and saves it as a PDF.

     Parameters:
     ------------
     filtered_summarized_tf_scores_df : pandas.DataFrame
          DataFrame with TF scores.
     
     condition : str
          Sample condition for file naming(e.g. control, disease ...)

     out_path : str
          Output path to save results.

     Returns:
     --------
     None
     '''
         
     top_variable_tfs = filtered_summarized_tf_scores_df_var.sort_values("var", ascending=False).head(n=20).drop(columns="var")
     cluster_map = sns.clustermap(top_variable_tfs, cmap="vlag", center=0, vmin=top_variable_tfs.min(axis=None), cbar_kws={"label": "t-score"},
                                   cbar_pos=(1, 0.5, 0.02, 0.1), dendrogram_ratio=(0.2, 0.05))
     cluster_map.ax_heatmap.set_xlabel("Cell Type")
     cluster_map.ax_heatmap.set_ylabel("Transcription Factor")
     plt.setp(cluster_map.ax_heatmap.get_xticklabels(), rotation=45) 
     plt.savefig(out_path + "/tf_activity_top20_variable_" + condition + ".pdf", bbox_inches='tight')
     plt.close()
    