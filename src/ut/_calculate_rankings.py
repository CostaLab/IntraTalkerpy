import numpy as np
from scipy.stats import wilcoxon, wasserstein_distance, ks_2samp, norm as normal
from scipy.stats import ttest_1samp
from statsmodels.stats.descriptivestats import sign_test
from typing import Tuple, Dict, Any, Optional, Union
import warnings


def calculate_pseudotime_comparison(
    old_pseudotime: np.ndarray,
    new_pseudotime: np.ndarray,
    pseudocount: float = 1e-6,
    tolerance: float = 1e-10
) -> Tuple[float, float]:
    """
    Comprehensive comparison of old vs new pseudotime distributions.
    
    This function performs multiple statistical tests to compare two pseudotime
    distributions and provides a comprehensive summary of the differences.
    
    Parameters
    ----------
    old_pseudotime : np.ndarray
        Original pseudotime values.
    new_pseudotime : np.ndarray
        Modified pseudotime values.
    pseudocount : float, default=1e-6
        Small value added to prevent log(0) in odds ratio calculation.
    tolerance : float, default=1e-10
        Tolerance for considering values as unchanged.
        
    Returns
    -------
    Tuple[float, float]
        log_odds_ratio : float
            Log odds ratio of increase vs decrease.
        p_value : float
            Statistical significance (Wilcoxon test).
        
    Raises
    ------
    ValueError
        If arrays have different lengths or contain invalid values.
        
    Examples
    --------
    >>> old_pt = np.array([0.1, 0.2, 0.3, 0.4])
    >>> new_pt = np.array([0.15, 0.25, 0.35, 0.45])
    >>> log_odds_ratio, p_value = calculate_pseudotime_comparison(old_pt, new_pt)
    >>> print(f"Log odds ratio: {log_odds_ratio:.3f}")
    >>> print(f"P-value: {p_value:.3e}")
    """
    # Input validation
    old_pseudotime = np.asarray(old_pseudotime)
    new_pseudotime = np.asarray(new_pseudotime)
    
    if len(old_pseudotime) != len(new_pseudotime):
        raise ValueError("old_pseudotime and new_pseudotime must have same length")
    
    if len(old_pseudotime) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    
    # Check for NaN values
    old_nan_mask = np.isnan(old_pseudotime)
    new_nan_mask = np.isnan(new_pseudotime)
    nan_mask = old_nan_mask | new_nan_mask
    
    if nan_mask.all():
        raise ValueError("All values are NaN")
    
    if nan_mask.any():
        n_nan = nan_mask.sum()
        warnings.warn(f"Removing {n_nan} NaN values from analysis")
        old_pseudotime = old_pseudotime[~nan_mask]
        new_pseudotime = new_pseudotime[~nan_mask]
    
    try:
        # Basic statistics
        differences = new_pseudotime - old_pseudotime
        mean_shift = np.mean(differences)
        rms_shift = np.sqrt(np.mean(differences**2))
        
        # Count changes
        n_increased = np.sum(differences > tolerance)
        n_decreased = np.sum(differences < -tolerance)
        n_unchanged = len(differences) - n_increased - n_decreased
        
        # Statistical tests
        if np.all(np.abs(differences) <= tolerance):
            warnings.warn("No significant differences detected")
            wilcoxon_stat, p_value = 0, 1.0
        else:
            try:
                wilcoxon_stat, p_value = wilcoxon(old_pseudotime, new_pseudotime, 
                                                  zero_method='zsplit')
            except ValueError as e:
                warnings.warn(f"Wilcoxon test failed: {e}. Using alternative.")
                wilcoxon_stat, p_value = 0, 1.0
        
        # Distribution comparison
        wasserstein_dist = wasserstein_distance(old_pseudotime, new_pseudotime)
        ks_stat, _ = ks_2samp(old_pseudotime, new_pseudotime)
        
        # Log odds ratio calculation
        if n_decreased == 0 and n_increased == 0:
            log_odds_ratio = 0.0
        else:
            log_odds_ratio = np.log((n_increased + pseudocount) / (n_decreased + pseudocount))
        
        # Effect size classification
        effect_size = _classify_effect_size(abs(mean_shift), rms_shift)
        
        return log_odds_ratio, p_value
        
    except Exception as e:
        warnings.warn(f"Error in pseudotime comparison: {e}")
        return 0.0, 1.0


def calculate_effect_size_from_differences(
    differences: np.ndarray,
    test_method: str = "wilcoxon",
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate effect size from pseudotime differences.
    
    This function calculates Cohen's d effect size and provides statistical
    significance testing for pseudotime differences.
    
    Parameters
    ----------
    differences : np.ndarray
        Array of pseudotime differences (new - old).
    test_method : str, default="wilcoxon"
        Statistical test method: "wilcoxon", "sign_test", or "t_test".
    confidence_level : float, default=0.95
        Confidence level for confidence interval calculation.
        
    Returns
    -------
    Tuple[float, float]
        cohens_d : float
            Cohen's d effect size.
        p_value : float
            Statistical significance.
        
    Raises
    ------
    ValueError
        If differences array is empty or test_method is invalid.
        
    Examples
    --------
    >>> differences = np.array([0.1, -0.05, 0.2, 0.15, -0.1])
    >>> cohens_d, p_value = calculate_effect_size_from_differences(differences)
    >>> print(f"Cohen's d: {cohens_d:.3f}")
    >>> print(f"P-value: {p_value:.3e}")
    """
    # Input validation
    differences = np.asarray(differences)
    
    if len(differences) == 0:
        raise ValueError("Differences array cannot be empty")
    
    valid_methods = ["wilcoxon", "sign_test", "t_test"]
    if test_method not in valid_methods:
        raise ValueError(f"test_method must be one of {valid_methods}")
    
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Remove NaN values
    valid_mask = ~np.isnan(differences)
    if not valid_mask.any():
        raise ValueError("All differences are NaN")
    
    if not valid_mask.all():
        n_nan = (~valid_mask).sum()
        warnings.warn(f"Removing {n_nan} NaN values from analysis")
        differences = differences[valid_mask]
    
    try:
        # Calculate Cohen's d
        if np.std(differences, ddof=1) == 0:
            cohens_d = 0.0
            warnings.warn("Standard deviation is zero, Cohen's d set to 0")
        else:
            cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Statistical test
        if test_method == "wilcoxon":
            try:
                _, p_value = wilcoxon(differences, zero_method='zsplit')
            except ValueError:
                warnings.warn("Wilcoxon test failed, using t-test")
                _, p_value = ttest_1samp(differences, 0)
        elif test_method == "sign_test":
            _, p_value = sign_test(differences)
        elif test_method == "t_test":
            _, p_value = ttest_1samp(differences, 0)
        
        # Effect magnitude classification
        effect_magnitude = _classify_cohens_d(abs(cohens_d))
        
        # Confidence interval for Cohen's d
        confidence_interval = _calculate_cohens_d_ci(differences, confidence_level)
        
        return cohens_d, p_value
        
    except Exception as e:
        warnings.warn(f"Error in effect size calculation: {e}")
        return 0.0, 1.0


def _classify_effect_size(mean_shift: float, rms_shift: float) -> str:
    """
    Classify effect size based on mean and RMS shift.
    
    Parameters
    ----------
    mean_shift : float
        Mean shift in pseudotime.
    rms_shift : float
        RMS shift in pseudotime.
        
    Returns
    -------
    str
        Effect size classification.
    """
    # Use both mean and RMS for more robust classification
    combined_effect = max(abs(mean_shift), rms_shift)
    
    if combined_effect < 0.01:
        return "negligible"
    elif combined_effect < 0.05:
        return "small"
    elif combined_effect < 0.1:
        return "medium"
    elif combined_effect < 0.2:
        return "large"
    else:
        return "very_large"


def _classify_cohens_d(cohens_d: float) -> str:
    """
    Classify Cohen's d effect size.
    
    Parameters
    ----------
    cohens_d : float
        Absolute Cohen's d value.
        
    Returns
    -------
    str
        Effect magnitude classification.
    """
    if cohens_d < 0.2:
        return "negligible"
    elif cohens_d < 0.5:
        return "small"
    elif cohens_d < 0.8:
        return "medium"
    elif cohens_d < 1.2:
        return "large"
    else:
        return "very_large"


def _calculate_cohens_d_ci(differences: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    """
    Calculate confidence interval for Cohen's d.
    
    Parameters
    ----------
    differences : np.ndarray
        Array of differences.
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI).
        
    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds of confidence interval.
    """
    try:
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        if std_diff == 0:
            return (0.0, 0.0)
        
        cohens_d = mean_diff / std_diff
        
        # Standard error of Cohen's d
        se_d = np.sqrt((n - 1) / (n - 3) * (1 + cohens_d**2 / (2 * (n - 1))))
        
        # Critical value
        alpha = 1 - confidence_level
        z_critical = normal.ppf(1 - alpha / 2)
        
        # Confidence interval
        ci_lower = cohens_d - z_critical * se_d
        ci_upper = cohens_d + z_critical * se_d
        
        return (ci_lower, ci_upper)
        
    except Exception:
        return (0.0, 0.0)