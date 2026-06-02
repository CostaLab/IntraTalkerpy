import pandas as pd

class TFObj:
    def __init__(self,
    tf_activities_condition : list,
    tf_activities_cluster : list,
    average_gene_expression : list,
    regulon : pd.DataFrame,
    CTR_input_condition : list,
    CTR_input_cluster : list,
    intracellular_network_condition : list,
    intracellular_network_cluster : list):

        self.tf_activities_condition = tf_activities_condition
        self.tf_activities_cluster = tf_activities_cluster
        self.average_gene_expression = average_gene_expression
        self.regulon = regulon
        self.CTR_input_condition = CTR_input_condition
        self.CTR_input_cluster = CTR_input_cluster
        self.intracellular_network_condition = intracellular_network_condition
        self.intracellular_network_cluster = intracellular_network_cluster


def make_TFOBj(tf_activities_condition : list,
    tf_activities_cluster : list,
    average_gene_expression : list,
    regulon : pd.DataFrame,
    CTR_input_condition : list,
    CTR_input_cluster : list,
    intracellular_network_condition : list,
    intracellular_network_cluster : list):
        
        '''
        Description:
        ------------
        Saves the IntraTalker Analysis results in an object.
        
        Parameters:
        -----------
        tf_activities_condition : list
            List of tf activities dataframes from compared condition analysis per condition.

        tf_activities_cluster : list
            List of tf activities dataframes from cluster analysis per condition.

        average_gene_expression : list
            List of average gene expression dataframes per condition.

        regulon : DataFrame 
            Regulon with source and target genes.

        CTR_input_condition : list
            List of CrossTalkeR input dataframes from compared condition analysis per condition.

        CTR_input_cluster : list
            List of CrossTalkeR input dataframes from cluster analysis per condition.

        intracellular_network_condition : list
            List of intracellular network dataframes from compared condition analysis per condition.

        intracellular_network_cluster : list
            List of intracellular network dataframes from cluster analysis per condition.

        Returns:
        ---------
        tf : TFObj
            Object with IntraTalker Analysis results.
        '''

        tf = TFObj(tf_activities_condition,
            tf_activities_cluster,
            average_gene_expression,
            regulon,
            CTR_input_condition,
            CTR_input_cluster,
            intracellular_network_condition,
            intracellular_network_cluster)

        return tf