import pandas as pd
import numpy as np
from .utils import add_entry, load_csv


def generate_CrossTalkeR_input(tf_activities, gene_expression, regulon, organism = "human"):
  '''
  Description:
  -------------
  Generates CrossTalkeR input from significant tf table.

  This function loads the transcription factor activity table/data frame
  for multiple cell types, and generates the CrossTalkeR input table.
  The returned input table contains receptor-transcription factor and
  transcription factor-ligand interactions based on the OmniPath database and the DoRothEA regulon.

  Parameters:
  -----------
  tf_activities : pandas.DataFrame
        Data Frame with transcription factor activities by cell type.

  gene_expression :pandas. DataFrame
        Table with average gene expression levels.

  regulon : pandas.DataFrame
        Regulon with source and target genes.
   
  organism : str
        Organism for which the data is being processed.

  Returns:
  --------
  output_df : pandas.DataFrame
    A data frame with CrossTalkeR input
  '''
  receptors : object

  if organism == "human":
    ligands = load_csv("ligands_human.csv")
    R2TF = load_csv("rtf_db_human.csv")
  elif organism == "mouse": 
    ligands = load_csv("ligands_mouse.csv")
    R2TF = load_csv("rtf_db_mouse.csv")
  else:
    raise NameError("Invalid organism to generate CrossTalkeR input!")

  #ligands
  ligands = ligands.drop_duplicates()
  #receptors
  R2TF = R2TF.set_index("tf")

  #mapping source and target with confidences
  sorted_regulon = regulon[["tf", "target"]]
  sorted_regulon = sorted_regulon.set_index("tf")

  #filtering for active TFs
  tf_activities = tf_activities[tf_activities["t_value"] > 0]

  #preextract values
  genes = tf_activities["gene"].values
  clusters = tf_activities["cluster"].values
  t_values = tf_activities["t_value"].values

  output_list = []
 
  
  for row in range(len(tf_activities)):
      tf = str(genes[row])
      cluster = clusters[row]
      t_value = t_values[row]

      #checking which data set's tfs are in the regulon
      if tf in sorted_regulon.index:
          targets = sorted_regulon.loc[tf, "target"]
      if isinstance(targets, str):
          targets = [targets]

      #checking which data set's tfs are in the receptor to tf file
      if tf in R2TF.index:
          receptors = R2TF.loc[tf, "receptor"]
          if isinstance(receptors, str):
              receptors = [receptors]
      else:
            receptors = []

      #checking which targets and ligands are the same
      if len(targets) > 0:
          tf_ligands = np.intersect1d(targets, ligands)

      #add interacting tfs and ligands to output 
      if len(tf_ligands) > 0:
          existing_entries = set()
          for ligand in tf_ligands:
              if ligand in gene_expression.index:
                      ex_value = gene_expression.at[ligand, cluster]
                      if ex_value != 0:
                          if (ligand, tf) not in existing_entries:
                              existing_entries.add((ligand, tf))
                              output_list.append(add_entry(source=cluster,
                                                            target=cluster,
                                                            gene_A=tf,
                                                            gene_B=ligand,
                                                            type_gene_A="Transcription Factor",
                                                            type_gene_B="Ligand",
                                                            MeanLR=t_value))
      #add interacting tfs and receptors to output 
      if len(receptors) > 0:
          for receptor in receptors:
              output_list.append(add_entry(source=cluster,
                                            target=cluster,
                                            gene_A=receptor,
                                            gene_B=tf,
                                            type_gene_A="Receptor",
                                            type_gene_B="Transcription Factor",
                                            MeanLR=t_value))

  output_df = pd.DataFrame(output_list)
  output_df[["gene_A", "gene_B"]] = output_df[["gene_A", "gene_B"]].replace("_", "+", regex=True)
  output_df.drop_duplicates(inplace=True)

  return output_df


def generate_intracellular_network(tf_activities, gene_expression, regulon, organism="human"):
    
    '''
    Description:
    ------------
    Generate connections in intracellular network.

    This function loads the transcription factor activity table/data frame
    for multiple cell types, and generates a table containing all detected intracellular connections.
    
    Parameters:
    ------------
    tf_activities : pandas.DataFrame
        Data Frame with transcription factor activities by cell type.

    gene_expression : pandas.DataFrame
        Table with average gene expression levels.

    regulon : pandas.DataFrame
       Regulon with source and target genes.
  
    organism : str
        Organism for which the data is being processed.

    Returns:
    ---------
    recept_regulon : pandas.DataFrame
        A data frame with intracellular network.
    '''

    if organism == "human":
        R2TF = load_csv("rtf_db_human.csv").set_index("tf")
    elif organism == "mouse":
        R2TF = load_csv("rtf_db_mouse.csv").set_index("tf")
    else:
        raise NameError("Invalid organism to generate intracellular network.")

    #mapping source and target with confidences
    sorted_regulon = regulon[["tf", "target"]]
    sorted_regulon = sorted_regulon.set_index("tf")

    #filtering for active TFs
    tf_activities = tf_activities[tf_activities["t_value"] > 0]
    
    #preextract values
    tf_genes = tf_activities["gene"].values
    tf_celltypes = tf_activities["cluster"].values
    tf_scores = tf_activities["t_value"].values

    TFTG_list = []
    RTF_list = []

    for row in range(len(tf_activities)):
        tf = str(tf_genes[row])
        celltype = tf_celltypes[row]
        tf_score = tf_scores[row]
        
        #checking which data set's tfs are in the regulon
        if tf in sorted_regulon.index:
          targets = sorted_regulon.loc[tf, "target"] 
        else:
          targets = []
        
        #checking which data set's tfs are in the receptor to tf file
        if tf in R2TF.index:
           receptors = R2TF.loc[tf, "receptor"]
           if isinstance(receptors, str):
               receptors = [receptors]
        else:
           receptors  = []

        if len(targets) > 0 and len(receptors) > 0:
            for target in targets:
                if target in gene_expression.index:
                    ex_value = gene_expression.at[target, celltype]
         
                    if ex_value != 0:
                        TFTG_list.append({
                            "celltype": celltype,
                            "TF": tf,
                            "Target_Gene": target,
                            "TF_Score": tf_score
                        })

            for receptor in receptors:
                RTF_list.append({
                    "TF": tf,
                    "Receptor": receptor
                })

    TFTG_df = pd.DataFrame(TFTG_list)
    RTF_df = pd.DataFrame(RTF_list)

    recept_regulon = pd.merge(RTF_df, TFTG_df, on="TF")
    recept_regulon = recept_regulon.sort_values("TF")
    return recept_regulon