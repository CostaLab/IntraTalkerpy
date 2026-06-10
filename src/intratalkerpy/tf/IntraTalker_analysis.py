import anndata as ad
import pandas as pd
import scanpy as sc
import os
from re import sub
import decoupler as dc
import numpy as np
from statsmodels.stats.multitest import multipletests

from .utils import validate_input_arguments, AverageExpression
from .utils import load_tf_activities
from .get_significant_TFs import get_significant_tfs
from .generate_intracellular_network import generate_intracellular_network, generate_CrossTalkeR_input
from .plot import plot_condition_tf_activities
from .get_condition_significant import condition_comparison_significant
from .tfobject import make_TFOBj

def IntraTalker_analysis(anndataobject, tf_activities = None, arguments_list = None):

    '''
    Description:
    -------------
    Main function that runs the IntraTalker Analysis
    Performs intracellular network anaysis based on conditions and clusters using decoupleR TF activity results and scRNA-seq data.

    Parameters:
    -------------
    anndataobject : AnnData 
        Input Anndata Object as h5ad file.
    
    tf_activities : csv | pandas.DataFrame
        Matrix with TF activities for each cell in the scRNA-seq data; Input as either csv or DataFrame.
    
    arguments_list : list
        Named list with custom options for the analysis. See validate_input_arguments in utils for further clarification.

    Returns:
    ---------
    tf : TFObj
        Object containing tf_activities_condition,
                    tf_activities_cluster,
                    average_gene_expression,
                    regulon,
                    CTR_input_condition,
                    CTR_input_cluster,
                    intracellular_network_condition,
                    intracellular_network_cluster.
    '''
    
    #load Anndata object
    if (isinstance(anndataobject, str)):
        anndataobject = ad.read_h5ad(anndataobject)

    #get arguments list
    arguments_list = validate_input_arguments(arguments_list)

    #create directory
    tf_path = os.path.join(arguments_list["out_path"], "TF_results")
    os.makedirs(tf_path, exist_ok=True)

    condition = anndataobject.obs[arguments_list["condition"]]

    #load tf activities(csv input)
    tf_activities = load_tf_activities(anndataobject, tf_activities, arguments_list)

    sc.pp.scale(tf_activities)

    #sets the stage for decision if single condition or comparison analysis is done
    if arguments_list["comparison_list"] is not None:
        if (len(arguments_list["comparison_list"]) > 0) & (len(pd.unique(anndataobject.obs[arguments_list["condition"]])) < 2):
            arguments_list["comparison_list"] = None
            print("Only one condition was found in the data, although a list of comparisons was provided. The analyses are performed only for the present condition!")

    #code for only single condition analysis/cluster analysis where marker TFs are grouped by celltype/cluster
    if arguments_list["comparison_list"] is None:

        result_list = {}
        gene_expression_list = {}
        CTR_cluster_list = {}
        intranet_cluster_list = {}

        #loops through each condition and creates single condition results
        for condition_iterable in anndataobject.obs[arguments_list["condition"]].unique():

            #subset anndata object and tf activities to single condition
            sub_object = anndataobject[anndataobject.obs[arguments_list["condition"]] == condition_iterable].copy()
            tf_activities_sub = tf_activities[tf_activities.obs[arguments_list["condition"]] == condition_iterable].copy() # type: ignore

            #replaces special symbols in condition names with _
            renamed_condition = sub("([,;.:-])", "_", condition_iterable)

            #get average gene expression for single condition from scRNA data
            sub_object_avg = AverageExpression(sub_object, renamed_condition, celltype = arguments_list["celltype"], outpath= arguments_list["out_path"])
        
            #get marker TFs for single condition
            tf_activity_scores = get_significant_tfs(tf_activities_sub,
                                                        renamed_condition,
                                                        tf_path,
                                                        tf_condition_significant = None,
                                                        celltype = arguments_list["celltype"],
                                                        pval = arguments_list["pval"],
                                                        meanchange = arguments_list["meanchange"],
                                                        plot = arguments_list["plot"],
                                                        condition_comparison= False)
            
            print("tf activities done")

            #add tf activity scores and average gene expression to result lists 
            result_list[renamed_condition] = tf_activity_scores
            gene_expression_list[renamed_condition] = sub_object_avg
            

            CTR_cluster_list[renamed_condition] = generate_CrossTalkeR_input(tf_activity_scores["cluster"],
                                                                            gene_expression_list[renamed_condition],                                           
                                                                            arguments_list["reg"],
                                                                            arguments_list["organism"])

            print("CTR input cluster done")
              
            intranet_cluster_list[renamed_condition] = generate_intracellular_network(tf_activity_scores["cluster"],
                                                                                  gene_expression_list[renamed_condition],
                                                                                  arguments_list["reg"],
                                                                                  arguments_list["organism"])
            
            print("intracellular network cluster done")
            
        #save as result object
        tf = make_TFOBj(
                tf_activities_condition = list(),
                tf_activities_cluster = result_list,
                average_gene_expression = gene_expression_list,
                regulon = arguments_list["reg"],
                CTR_input_condition = list(),
                CTR_input_cluster = CTR_cluster_list,
                intracellular_network_condition = list(),
                intracellular_network_cluster = intranet_cluster_list)

        #with open((arguments_list["out_path"] + "tf.pickle"), "wb") as file:
        #    pickle.dump(tf, file)

        return tf

    else:
        #code for compared condition analysis with additional single condition analysis
        
        #compared condition analysis:
        out_path_compared = (tf_path + "/compared")
        if not os.path.isdir(out_path_compared):
            os.mkdir(out_path_compared)

        #finds marker TFs for compared conditions
        compared_significant_tfs = condition_comparison_significant(tf_activities, out_path_compared, arguments_list["celltype"], 
                                                                    arguments_list["condition"], arguments_list["comparison_list"], 
                                                                    arguments_list["num_cell_filter"])
        
        print("compared tfs done")
        
        #creates heatmaps for compared condition (TF activity in first condition when compared to second condition)
        if arguments_list["plot"] == True:
            plot_condition_tf_activities(compared_significant_tfs, out_path_compared, which = "both")

    
        result_condition_list = {}
        result_cluster_list = {}
        gene_expression_list = {}
        CTR_condition_list = {}
        CTR_cluster_list = {}
        intranet_condition_list = {}
        intranet_cluster_list = {}

        #single condition/cluster analysis:
        for condition_iterable in anndataobject.obs[arguments_list["condition"]].unique():

            #subset anndata object and tf activities to single condition
            sub_object = anndataobject[anndataobject.obs[arguments_list["condition"]] == condition_iterable]
            tf_activities_sub = tf_activities[tf_activities.obs[arguments_list["condition"]] == condition_iterable]
            
            #initialize df
            compared_tfs = pd.DataFrame({"gene" : pd.Series(dtype="str"), "tag" : pd.Series(dtype="str"), "cluster" : pd.Series(dtype="str")})
        
            #replaces special symbols in condition names with _
            renamed_condition = sub("([,;.:-])", "_", condition_iterable)
            
            #filters compared tfs
            for result_name, df in compared_significant_tfs.items(): 
                if renamed_condition in result_name:
                    tf_condition_significant = compared_significant_tfs[result_name].copy()
                    tf_condition_significant = tf_condition_significant[tf_condition_significant["FDR"] < float(arguments_list["pval"])]
                    tf_condition_significant = tf_condition_significant[(tf_condition_significant["meanchange"] > float(arguments_list["meanchange"])) | (tf_condition_significant["meanchange"] < (0 - float(arguments_list["meanchange"])))]
                    tf_condition_significant = tf_condition_significant[["tf", "tag", "CellType"]]
                    tf_condition_significant.rename(columns={"tf":"gene", "CellType": "cluster"}, inplace=True)
                    compared_tfs = pd.concat([compared_tfs, tf_condition_significant])


            #get average gene expression for single condition from scRNA data
            sub_object_avg = AverageExpression(sub_object, renamed_condition, celltype = arguments_list["celltype"], 
                                               outpath= arguments_list["out_path"])
            
            #get marker TFs for single condition
            tf_activity_scores = get_significant_tfs(tf_activities_sub,
                                               renamed_condition,
                                               tf_path,
                                               compared_tfs,
                                               celltype = arguments_list["celltype"],
                                               pval = arguments_list["pval"],
                                               meanchange = arguments_list["meanchange"],
                                               plot = arguments_list["plot"],
                                               condition_comparison= True)
            
            print("tf activities done")
            
            #add tf activity scores and average gene expression to result lists 
            result_condition_list[renamed_condition] = tf_activity_scores["condition"]
            result_cluster_list[renamed_condition] = tf_activity_scores["cluster"]
            gene_expression_list[renamed_condition] = sub_object_avg
            
        
            CTR_condition_list[renamed_condition] = generate_CrossTalkeR_input(tf_activity_scores["condition"],
                                                                            gene_expression_list[renamed_condition],                                         
                                                                           arguments_list["reg"],
                                                                           arguments_list["organism"])
            
            print("CTR input condition done")
    
            CTR_cluster_list[renamed_condition] = generate_CrossTalkeR_input(tf_activity_scores["cluster"],
                                                                            gene_expression_list[renamed_condition],                                    
                                                                            arguments_list["reg"],
                                                                            arguments_list["organism"])
    
            print("CTR input cluster done")

            intranet_condition_list[renamed_condition] = generate_intracellular_network(tf_activity_scores["condition"],
                                                                                  gene_expression_list[renamed_condition],
                                                                                  arguments_list["reg"],
                                                                                  arguments_list["organism"])
    
            print("intracellular network condition done")

            intranet_cluster_list[renamed_condition] = generate_intracellular_network(tf_activity_scores["cluster"],
                                                                                  gene_expression_list[renamed_condition],
                                                                                  arguments_list["reg"],
                                                                                  arguments_list["organism"])
            print("intracellular network cluster done")
        
        #save as result object
        tf = make_TFOBj(
                tf_activities_condition = result_condition_list,
                tf_activities_cluster = result_cluster_list,
                average_gene_expression = gene_expression_list,
                regulon = arguments_list["reg"],
                CTR_input_condition = CTR_condition_list,
                CTR_input_cluster = CTR_cluster_list,
                intracellular_network_condition = intranet_condition_list,
                intracellular_network_cluster = intranet_cluster_list)

        #with open((arguments_list["out_path"] + "tf.pickle"), "wb") as file:
        #    pickle.dump(tf, file)
        
        return tf
    

    