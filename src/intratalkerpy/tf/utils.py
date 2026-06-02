import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import variance
import importlib.resources as pkg_resources
from . import data 
from .plot import plot_top_variable_tfs


def load_csv(filename):
    """
    Description:
    -------------
    Loads a CSV file from the package's data folder.

    Parameters:
    ------------
    filename : str
      Path to the file.

    Returns:
    ---------
    f : csv
      Loaded CSV.
    """
    with pkg_resources.files(data).joinpath(filename).open("r") as f:
        return pd.read_csv(f)
    

def add_entry(source, target, gene_A, gene_B, type_gene_A, type_gene_B, MeanLR):
  '''
  Description:
  -------------
  Adds entry to a dataframe in the CrossTalkeR input format.

  Parameters:
  ------------
  source : pandas.Series
    source cell type

  target : pandas.Series
    target cell type

  gene_A : pandas.Series

  gene_B : pandas.Series

  type_gene_A : pandas.Series

  type_gene_B : pandas.Series

  MeanLR : pandas.Series
    Is the same as tf score/t value. 

  Returns:
  --------
  df : dict[str, pandas.Series]
    Entry in CrossTalkeR input dataframe.
  '''
  df = {"source" : source,
      "target" : target,
      "gene_A" : gene_A,
      "gene_B" : gene_B,
      "type_gene_A" : type_gene_A,
      "type_gene_B" : type_gene_B,
      "MeanLR" : MeanLR}

  return df

def map_t_value(tf_scores_df, anndataobject_markers):
    
    '''
    Description:
    -------------
    Merges the significant gene output from the dc.rank_sources_groups function with the t value/tf score from the filtered and summarized decoupler tf activity dataframe.

    Parameters:
    -----------
    tf_scores_df : pandas.DataFrame
    
    anndataobject_markers : pandas.DataFrame

    Returns:
    ---------
    t_value_df : pandas.DataFrame
      Dataframe with gene, cell type, significance tag and t-value.
    '''
    anndataobject_markers = anndataobject_markers.set_index("gene")
    merged_df = tf_scores_df.merge(anndataobject_markers[["cluster"]], left_index=True, right_index=True, how="inner")
    melted_df = merged_df.reset_index().melt(id_vars=["gene", "cluster"], var_name="cell_type", value_name="t_value")
    t_value_df = melted_df[melted_df["cell_type"] == melted_df["cluster"]].drop(columns="cell_type")
    return t_value_df


def create_unfiltered_tf_scores(tf_scores_df, condition, celltype, out_path):  
    '''
    Description:
    -------------
    Creates a csv containing the summarized unfiltered tf scores from the decoupler TF activity dataframe.

    Parameters:
    ------------
    tf_scores_df : pandas.DataFrame
      Dataframe with TF activity scores.

    condition : str
      Condition annotation.

    celltype : str
      Celltype anntotation.

    out_path : str
      Output path to save results.

    Returns:
    ---------
    summarized_tf_scores_df : pandas.DataFrame
      Dataframe with TF score per celltype for all TFs.
    ''' 
    summarized_tf_scores_df = tf_scores_df.groupby(celltype, observed = True).mean().T
    summarized_tf_scores_df.index.names = ["gene"]
    summarized_tf_scores_df.to_csv(out_path + "/unfiltered_tf_scores_" + condition + ".csv")
    return summarized_tf_scores_df


def save_variable_tf_score(filtered_summarized_tf_scores_df, condition, out_path, plot):
    
    '''
    Description:
    ------------
    This function saves the transcription factor activity scores per cell type into a csv table.

    Parameters:
    ------------
    filtered_summarized_tf_scores_df : pandas.DataFrame
      Data frame with transcription factor activity scores per cell type.
    
    condition : str
      Sample condition for file naming(e.g. control, disease ...)
    
    out_path : str
      Output path to save results.

    Returns:
    ---------
    filtered_summarized_tf_scores_df_var : pandas.DataFrame
      Dataframe with TF scores and their variance over cell types for filtered genes.
    '''
    filtered_summarized_tf_scores_df_var = filtered_summarized_tf_scores_df.copy()
    filtered_summarized_tf_scores_df_var["var"] = filtered_summarized_tf_scores_df_var.var(axis=1)
    filtered_summarized_tf_scores_df_var.to_csv(out_path + "/variable_tf_scores_" + condition + ".csv")

    if plot:
        plot_top_variable_tfs(filtered_summarized_tf_scores_df_var, condition, out_path)

    return filtered_summarized_tf_scores_df_var



def eval_pval(p_val):
    '''
    Description:
    ------------
    Creates significance tags based on the p-value.

    Parameters:
    ------------
    p_val : float
       P-value to be evaluated.

    Returns:
    ---------
    txt : str
      Significance tags for TF scores based on their p-value.

    '''
    p_val = float(p_val)
    if p_val < 0.001: 
      txt = "***"
    elif p_val < 0.01: 
      txt = "**"
    elif p_val < 0.05: 
      txt = "*"
    else:
      txt = "ns"
    return(txt)


def eval_meanchange_tag(meanchange):
    '''
    Description:
    -------------
    Creates significance tags based on the meanchange.

    Parameters:
    ------------
    meanchange: float
      Meanchange to be evaluated.

    Returns:
    ---------
    txt : str
      Significance tags for TF scores based on their meanchange.
    '''
    if meanchange >= 1.0: 
      txt = "***"
    elif meanchange > 0.5: 
      txt = "**"
    elif meanchange > 0.0: 
      txt = "*"
    else:
      txt = "ns"
    return(txt)



def AverageExpression(sub_object, name_iterable = None, celltype = None, outpath = None):
    '''
    Description:
    -------------
    Calculates the average gene expression of each gene per cell type for each condition.

    Parameters:
    -----------
    sub_object : AnnData
      AnnData object subset to one condition.
    
    name_iterable : str
      formatted name of the condition.

    celltype : str
      Celltype annotation.

    outpath : str
      Output path to save results.

    Returns:
    ---------
    avg_df : pandas.DataFrame
      Average gene expression of each gene per cell type.
    '''
    gene_ids = sub_object.var.index.values
    obs = sub_object[:,gene_ids].X.toarray()
    obs = np.expm1(obs)
    avg_df = pd.DataFrame(obs,columns=gene_ids,index= sub_object.obs[celltype])
    avg_df = avg_df.groupby(level=0, observed=False).mean()
    avg_df.T.to_csv(f"{outpath}{name_iterable}_average_gene_expression_by_cluster_exp.csv")

    return avg_df.T


def validate_input_arguments (arguments_list):
    
    '''
    Description:
    -------------()
    Checks arguments passed by User for validity.

    Parameters: 
    ------------
    arguments_list : list(str, float, bool) 

     List of user defined arguments. They include: 

    out_path : str
      Output path to save results.

    celltype : str
      Metadata field containing cell type annotations.

    condition : str
      Metadata field containing condition annotations.

    organism : str
      "human" or "mouse".
    
    meanchange : float
      Cutoff value for meanchange.

    pval : float 
      Cutoff value for p-value.

    num_cell_filter : int
      Minimum number of cells in each cell type.

    reg : str | pandas.DataFrame
      Path to regulon csv or direct input as pandas.DataFrame. Must include source, target and weight columns.

    plot : bool
      Whether to generate plots or not.

    decoupler_matrix_format : str
      "R" or "Python", since csvs generated by the R version of decoupler need to be transposed.

    Returns:
    ---------
    arguments_list : list
      Validated list of arguments.
    '''

    if arguments_list["out_path"] is None:
        print("Please provide an output path")
    elif arguments_list["out_path"][-1] != "/":
        arguments_list["out_path"] = arguments_list["out_path"] + "/"

    if arguments_list["celltype"] is None:
        raise ValueError("Please provide the name of the metadata field containing cell type annotations.")

    if arguments_list["condition"] is None:
        raise ValueError("Please provide the name of the metadata field containing condition annotations.")

    if arguments_list["organism"] is None:
        arguments_list["organism"] = "human"

    if arguments_list["meanchange"] is None:
        arguments_list["meanchange"] = 0.0

    if arguments_list ["pval"] is None:
        arguments_list["pval"] = 0.05

    if arguments_list ["num_cell_filter"] is None:
        arguments_list["num_cell_filter"] = 0

    if arguments_list["reg"] is None:
        raise ValueError("Please provide a regulon csv.")

    elif isinstance(arguments_list["reg"], str):
        arguments_list["reg"] = pd.read_csv(arguments_list["reg"], index_col=0)
        arguments_list["reg"] = pd.DataFrame.rename(arguments_list["reg"], columns={"source" : "tf"})

    if not "tf" in arguments_list["reg"] and "target" in arguments_list["reg"] and "weight" in arguments_list["reg"]:
        raise NameError("Not all necessary columns found in regulon table! Please make sure that the regulon has the columns 'source', 'target' and 'weight'!")
    
    if arguments_list["plot"] is None:
        arguments_list["plot"] = True
    elif not isinstance(arguments_list["plot"], (bool)):
        raise ValueError("Plot argument must be a boolean value.")

    return(arguments_list)


def add_node_type(df):
      
    '''
    Description:
    -------------
    Adds gene type to the name of the gene.

    Parameters:
    -------------
    df : pandas.DataFrame
      Dataframe with all interactions.

    Returns:
    ----------
    df : pandas.DataFrame
      Dataframe with node type added to gene type.
    '''
      
    df['gene_A'] = np.where(df['type_gene_A'] == 'Ligand', df['gene_A'] + '|L', df['gene_A'])
    df['gene_A'] = np.where(df['type_gene_A'] == 'Receptor', df['gene_A'] + '|R', df['gene_A'])
    df['gene_A'] = np.where(df['type_gene_A'] == 'Transcription Factor', df['gene_A'] + '|TF', df['gene_A'])
    df['gene_B'] = np.where(df['type_gene_B'] == 'Ligand', df['gene_B'] + '|L', df['gene_B'])
    df['gene_B'] = np.where(df['type_gene_B'] == 'Receptor', df['gene_B'] + '|R', df['gene_B'])
    df['gene_B'] = np.where(df['type_gene_B'] == 'Transcription Factor', df['gene_B'] + '|TF', df['gene_B'])
    return df



def combine_LR_and_TF(tf_table, LR_prediction, out_path, condition, add_nodetype = False):
  '''
  Description:
  ------------
  Combining Ligand-Receptor interaction prediction with Transcription Factor interaction predictions

  Parameters:
  ------------
  tf_table : pandas.DataFrame
    Table with TF interactions.
  
  LR_prediction : pandas.DataFrame
    Path to or dataframe with ligand-receptor interaction prediction.
  
  out_path: str
    Output path to save results.
  
  condition: str
    Sample condition of data. Used for the filename, so preferably replace special symbols with an underscore.

  Returns:
  ---------
  complete_interactions : pandas.DataFrame
    Dataframe of with source, target, gene_A, gene_B, type_gene_A, type_gene_B, MeanLR columns for CrossTalkeR input.
  '''

  if isinstance(LR_prediction, pd.DataFrame):
    lr_table = LR_prediction
  else: 
    lr_table = pd.read_csv(LR_prediction, index_col=0)
  

  intra_connections = pd.DataFrame()
  for celltype in np.unique([lr_table["source"], lr_table["target"]]):
    lr_filtered_ligands = lr_table[lr_table["source"] == celltype]
    lr_filtered_receptors = lr_table[lr_table["target"] == celltype]
    lr_ligands = np.unique(lr_filtered_ligands["gene_A"])
    lr_receptors = np.unique(lr_filtered_receptors["gene_B"])
   
    tf_table_receptors = tf_table[(tf_table["target"] == celltype) & (tf_table["type_gene_A"] == "Receptor")]
    tf_table_ligands = tf_table[(tf_table["source"] == celltype) & (tf_table["type_gene_B"] == "Ligand")]

    tf_receptor_interactions =  tf_table_receptors[tf_table_receptors["gene_A"].isin(lr_receptors)]
    tf_ligand_interactions = tf_table_ligands[tf_table_ligands["gene_B"].isin(lr_ligands)]


    intra_connections = pd.concat([intra_connections, tf_receptor_interactions, tf_ligand_interactions])
  intra_connections["all_pair"] = (intra_connections["source"] + "/" 
                                    + intra_connections["gene_A"] + "/"
                                    + intra_connections["target"] + "/"
                                    + intra_connections["gene_B"])
    
  intra_connections = intra_connections.drop_duplicates(subset=["all_pair"])
  intra_connections.drop(columns=["all_pair"], inplace=True)

  complete_interactions = pd.concat([intra_connections, lr_table])

  if add_nodetype:
    complete_interactions = add_node_type(complete_interactions)
      
  complete_interactions.to_csv((out_path + "CrossTalkeR_input_" + condition + ".csv"), index=False)
  return(complete_interactions)



def combine_LR_and_TF_complexes(tf_table, LR_prediction, out_path, condition, add_nodetype = False):
  '''
  Description:
  ------------
  Combining Ligand-Receptor interaction prediction with Transcription Factor interaction predictions

  Parameters:
  ------------
  tf_table : pandas.DataFrame
    Table with TF interactions.
  
  LR_prediction: pandas.DataFrame
    Path to or dataframe with ligand-receptor interaction prediction.
  
  out_path: str
    Output path to save results.
  
  condition: str
    Sample condition of data. Used for the filename, so preferably replace special symbols with an underscore.

  Returns:
  ---------
  complete_interactions : pandas.DataFrame
    Dataframe of  with source, target, gene_A, gene_B, type_gene_A, type_gene_B, MeanLR columns for CrossTalkeR input.
  '''

  if isinstance(LR_prediction, pd.DataFrame):
    lr_table = LR_prediction
  else: 
    lr_table = pd.read_csv(LR_prediction, index_col=False)
  
  intra_connections = pd.DataFrame()
  for celltype in np.unique([lr_table["source"], lr_table["target"]]):
    lr_filtered_ligands = lr_table[lr_table["source"] == celltype]
    lr_filtered_receptors = lr_table[lr_table["target"] == celltype]
    lr_ligands = np.unique(lr_filtered_ligands["gene_A"])
    lr_receptors = np.unique(lr_filtered_receptors["gene_B"])

    lr_receptors = pd.Series(lr_receptors)
    contains_complex = lr_receptors.str.contains("_", na=False)
    
    R_with_complex = lr_receptors[contains_complex]
    R_without_complex = lr_receptors[(~contains_complex)]
  
    tf_table_receptors = tf_table[(tf_table["target"] == celltype) & (tf_table["type_gene_A"] == "Receptor")]
    tf_receptor_interactions =  tf_table_receptors[tf_table_receptors["gene_A"].isin(R_without_complex)]

    c_receptors = tf_table_receptors[tf_table_receptors["gene_A"].apply(lambda x: any(gene in x.split("+") for gene in lr_receptors))]
    
    complex_df = pd.DataFrame()
    if len(R_with_complex) > 0:
      for complex in R_with_complex:
        receptors = complex.split("_")
        R_TF_with_complex = tf_table_receptors[tf_table_receptors["gene_A"].isin(receptors)]
 
        if len(R_TF_with_complex) == 0:
          continue
        
        R_TF_with_complex.drop_duplicates()
        R_TF_with_complex.loc[:,"gene_A"] = complex
        complex_df = pd.concat([complex_df, R_TF_with_complex])


    tf_receptor_interactions = pd.concat([tf_receptor_interactions, complex_df])
  
    tf_table_ligands = tf_table[(tf_table["source"] == celltype) & (tf_table["type_gene_B"] == "Ligand")]

    tf_ligand_interactions = tf_table_ligands[tf_table_ligands["gene_B"].isin(lr_ligands)]

    intra_connections = pd.concat([intra_connections, tf_receptor_interactions, c_receptors, tf_ligand_interactions])
  
  intra_connections["all_pair"] = (intra_connections["source"] + "/" 
                                    + intra_connections["gene_A"] + "/"
                                    + intra_connections["target"] + "/"
                                    + intra_connections["gene_B"])

  intra_connections = intra_connections.drop_duplicates(subset=["all_pair"])
  intra_connections.drop(columns=["all_pair"], inplace=True)

  complete_interactions = pd.concat([intra_connections, lr_table])
  
  if add_nodetype:
    complete_interactions = add_node_type(complete_interactions)
      
  complete_interactions.to_csv((out_path + "CrossTalkeR_input_" + condition + ".csv"), index=False) #quoting= csv.QUOTE_NONNUMERIC)
  return(complete_interactions)
