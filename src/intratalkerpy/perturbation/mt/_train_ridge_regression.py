import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from typing import Optional

def compute_receptor_coeff_stats(
    coeff_matrix: pd.DataFrame,
    regulon: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute coefficient statistics from a regression coefficient matrix stored
    in an AnnData object and save the results back into adata.uns.

    Intended to be called directly after training the regression model.

    Parameters
    ----------
    coeff_matrix : pd.DataFrame
        DataFrame containing the coefficient matrix with receptors as rows and
        samples/iterations as columns.
    regulon : pd.DataFrame
        DataFrame with a 'Receptor' column used to count how many target genes
        each receptor regulates.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the computed statistics for each receptor:
            - Gene         : receptor/gene name
            - Mean_Coeff   : mean coefficient across columns
            - Median_Coeff : median coefficient across columns
            - Std_Coeff    : standard deviation of coefficients
            - Sum_Coeff    : sum of coefficients
            - AbsSum_Coeff : sum of absolute values of coefficients
            - Max_Coeff    : maximum coefficient
            - Min_Coeff    : minimum coefficient
            - Num_Positive : number of positive coefficients
            - Num_Negative : number of negative coefficients
            - Num_Nonzero  : number of non-zero coefficients
            - count        : number of target genes regulated (from regulon)
    """

    coeff_stats = pd.DataFrame(index=coeff_matrix.index)
    coeff_stats["Mean_Coeff"]   = coeff_matrix.mean(axis=1)
    coeff_stats["Median_Coeff"] = coeff_matrix.median(axis=1)
    coeff_stats["Std_Coeff"]    = coeff_matrix.std(axis=1)
    coeff_stats["Sum_Coeff"]    = coeff_matrix.sum(axis=1)
    coeff_stats["AbsSum_Coeff"] = coeff_matrix.abs().sum(axis=1)
    coeff_stats["Max_Coeff"]    = coeff_matrix.max(axis=1)
    coeff_stats["Min_Coeff"]    = coeff_matrix.min(axis=1)
    coeff_stats["Num_Positive"] = (coeff_matrix > 0).sum(axis=1)
    coeff_stats["Num_Negative"] = (coeff_matrix < 0).sum(axis=1)
    coeff_stats["Num_Nonzero"]  = (coeff_matrix != 0).sum(axis=1)

    coeff_stats = coeff_stats.reset_index()
    coeff_stats.columns = ["Gene"] + list(coeff_stats.columns[1:])

    coeff_stats = coeff_stats[coeff_stats["Mean_Coeff"] != 0]

    r_tg_counts = regulon.groupby("Receptor").size().to_frame(name="count")
    coeff_stats = pd.merge(
        coeff_stats, r_tg_counts,
        left_on="Gene", right_index=True,
        how="left",
    )
    coeff_stats["count"] = coeff_stats["count"].fillna(0).astype(int)

    return coeff_stats


def calculate_coef_matrix_ridge(
    data,
    regulon,
    alpha=10.0,
    source_col="source",
    target_col="target"
):
    """
    Fit Ridge regression models to estimate receptor influence
    on target gene expression and store the resulting coefficient matrix and
    regression statistics in the AnnData object.

    For each gene in the expression matrix, a Ridge regression model is trained
    using the expression of its upstream regulators (receptors) as predictors. Genes
    that are not present in the regulon or have no valid receptor regulators are
    assigned zero coefficients. Results are stored in ``data.uns``.

    Parameters
    ----------
    data : anndata.AnnData
        Annotated data matrix where rows are observations (cells/samples) and
        columns are genes. Expression values are retrieved via ``data.to_df()``.
    regulon : pandas.DataFrame
        A DataFrame describing receptor-target relationships. Must contain at least
        two columns: one for source receptors and one for target genes (see
        ``source_col`` and ``target_col``).
    alpha : float, optional
        Regularisation strength for Ridge regression. Larger values impose
        stronger regularisation. Default is ``10.0``.
    source_col : str, optional
        Name of the column in ``regulon`` that contains receptor (source) gene
        names. Default is ``"source"``.
    target_col : str, optional
        Name of the column in ``regulon`` that contains target gene names.
        Default is ``"target"``.

    Returns
    -------
    anndata.AnnData
        The input ``data`` object with the following entries added to
        ``data.uns``:

        ``"regression_coef_matrix"`` : pandas.DataFrame
            A (genes × genes) coefficient matrix where entry ``[i, j]``
            represents the Ridge regression coefficient of receptor *i* predicting
            target gene *j*. Columns are target genes; rows are all genes.
            Entries are zero when a gene is not a regulator of that target.

        ``"regression_statistics"`` : pandas.DataFrame
            Per-target-gene regression diagnostics, indexed by target gene
            name, with the following columns:

            - ``num_receptors`` – number of receptors used as predictors.
            - ``num_selected_features`` – number of non-zero receptor coefficients.
            - ``alpha`` – regularisation strength used by the fitted model.
            - ``mse`` – mean squared error on the training data.
            - ``rmse`` – root mean squared error on the training data.
            - ``nrmse_mean`` – RMSE normalised by the mean of the target.
            - ``nrmse_maxmin`` – RMSE normalised by the (max − min) range of the target.
            - ``r2`` – R² coefficient of determination on the training data.

        ``"regression_coef_statistics"`` : object
            Summary statistics over the coefficient matrix, as returned by
            ``compute_receptor_coeff_stats()``.

    Notes
    -----
    - Self-regulation is excluded: if a gene appears as both source and target
      for itself, it is removed from the predictor set before fitting.
    - Only receptors present in both the regulon and the expression matrix are used
      as predictors.
    - Genes absent from the regulon receive an all-zero coefficient vector.
    - Regression metrics are computed on training data (in-sample), so they
      reflect model fit rather than generalisation performance.

    Examples
    --------
    >>> import anndata as ad
    >>> data = calculate_coef_matrix_ridge(
    ...     data=adata,
    ...     regulon=regulon_df,
    ...     outpath="results/",
    ...     alpha=5.0,
    ...     source_col="source",
    ...     target_col="target",
    ... )
    >>> coef_matrix = data.uns["regression_coef_matrix"]
    >>> stats = data.uns["regression_statistics"]
    """
    
    df_exp = data.to_df()
    genes = df_exp.columns
    all_target_genes = set(df_exp.columns, ).intersection(set(regulon[target_col].values))
    empty_coefs = pd.Series(np.zeros(len(genes)), index=genes)

    statisics_tf = {}
    coef_by_target = []
    for target in genes:
        print(target)
        if not target in all_target_genes:
            tmp_coefs = empty_coefs.copy()
            tmp_coefs[target] = 0
        else:
            tmp_coefs = empty_coefs.copy()
            tfs_of_target = list(set(regulon[regulon[target_col].isin([target])][source_col]))

            if target in tfs_of_target:
                tfs_of_target.remove(target)
            if len(tfs_of_target) == 0:
                tmp_coefs[target] = 0

            else:
                target_exp = df_exp[target].to_frame()
                tfs_of_target = list(set(tfs_of_target).intersection(set(df_exp.columns)))
                tfs_exp = df_exp[tfs_of_target]

                if len(target_exp.columns) == 1 and len(tfs_exp.columns) > 0:
                    ridge_reg = Ridge(alpha=alpha)
                    ridge_reg_fit = ridge_reg.fit(tfs_exp, np.ravel(target_exp))
                    ridge_pred = ridge_reg_fit.predict(tfs_exp)
                    mse_test = mean_squared_error(np.ravel(target_exp), ridge_pred)
                    rmse_test = root_mean_squared_error(np.ravel(target_exp), ridge_pred)
                    r2_test = r2_score(np.ravel(target_exp), ridge_pred)
                    tmp_coefs[tfs_of_target] = ridge_reg_fit.coef_
                    statisics_tf[target] = {"num_receptors": len(tfs_of_target),
                                            "num_selected_features": np.count_nonzero(ridge_reg_fit.coef_),
                                            "alpha": ridge_reg_fit.alpha, 
                                            "mse": mse_test, 
                                            "rmse": rmse_test,
                                            "nrmse_mean": rmse_test / np.mean(np.ravel(target_exp)),
                                            "nrmse_maxmin": rmse_test / ( max(np.ravel(target_exp)) - min(np.ravel(target_exp))),
                                            "r2": r2_test}
        coef_by_target.append(tmp_coefs)
    coef_matrix = pd.concat(coef_by_target, axis=1)
    coef_matrix.columns = genes

    df = pd.DataFrame.from_dict(statisics_tf, orient="index")
    data.uns["regression_statistics"] = df
    
    data.uns["regression_coef_matrix"] = coef_matrix
    
    coef_stats = compute_receptor_coeff_stats(coef_matrix)
    data.uns["regression_coef_statistics"] = coef_stats

    return data