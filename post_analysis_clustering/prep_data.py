import pandas as pd
import numpy as np
import re
from post_analysis_clustering.utils import timer

@timer
def driver_get_raw_and_scaled(df: pd.DataFrame, num_experiment: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate raw and scaled features from the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing both raw and scaled features.
        num_experiment (int, optional): Number of experimental clustering columns at the end to include. Defaults to 9.

    Returns:
        tuple:
            raw_df (pd.DataFrame): DataFrame with only raw features and experimental clustering columns.
            scale_df (pd.DataFrame): DataFrame with scaled features and experimental clustering columns.
    """    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(num_experiment, int) or num_experiment <= 0:
        raise ValueError("num_experiment must be a positive integer.")

    try:
        raw_df = df.loc[:, ~df.columns.str.startswith("SCALED_")]
        scale_df = df.iloc[:, [0]].join(df.loc[:, df.columns.str.startswith("SCALED_")]).join(df.iloc[:, -num_experiment:])
        return raw_df, scale_df
    except Exception as e:
        raise RuntimeError(f"Error in driver_get_raw_and_scaled: {e}")

@timer
def get_feature_list(
    df: pd.DataFrame,
    exclude_k_columns: bool = False
) -> list:
    """
    Extract feature list by removing the first (primary key) column and optionally:
    - The last column (assumed to be cluster column).
    - Any columns that start with 'K' followed by numbers (e.g., K2, K10).

    Args:
        df (pd.DataFrame): Input DataFrame.
        exclude_k_columns (bool): If True, exclude columns matching the pattern 'K\\d+'.

    Returns:
        list: List of feature column names based on selection rules.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.shape[1] < 3:
        raise ValueError("DataFrame must have at least 3 columns (primary_key, features, target_cluster).")

    try:
        feature_list = df.columns[1:-1].tolist()  # Exclude first and last columns
        if exclude_k_columns:
            feature_list = [col for col in feature_list if not re.match(r"^K\d+$", col)]
        return feature_list
    except Exception as e:
        raise RuntimeError(f"Error in get_feature_list: {e}")

@timer
def get_numeric_feature_lists(
    df: pd.DataFrame,
    primary_key: str,
    target_cluster: str,
    nunique_threshold: int = 20
) -> tuple[list, list]:
    """
    Returns two lists of numerical features:
    1. Continuous features: numerical columns with nunique >= threshold.
    2. Discrete features: numerical columns with nunique < threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        primary_key (str): Column name for the primary key (will be excluded).
        target_cluster (str): Column name for the target/cluster (will be excluded).
        nunique_threshold (int): Threshold to distinguish discrete vs. continuous. Default is 20.

    Returns:
        tuple: (continuous_features: list[str], discrete_features: list[str])
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    excluded = set()

    for col in [primary_key, target_cluster]:
        if col in df.columns:
            excluded.add(col)
            print(f"Ignoring column '{col}' — marked as primary key or target cluster.")

    numeric_cols = df.select_dtypes(include=['number']).columns
    usable_cols = [col for col in numeric_cols if col not in excluded]

    continuous_features = []
    discrete_features = []
    
    print("\nAnalyzing numerical features:")
    for col in usable_cols:
        unique_vals = df[col].nunique()
        print(f"  • '{col}' has {unique_vals} unique values")
        if unique_vals >= nunique_threshold:
            continuous_features.append(col)
            print(f"    → Classified as CONTINUOUS ({unique_vals} ≥ {nunique_threshold})")
        else:
            discrete_features.append(col)
            print(f"    → Classified as DISCRETE ({unique_vals} < {nunique_threshold})")

    ignored_non_numeric = [col for col in df.columns if col not in numeric_cols and col not in excluded]
    for col in ignored_non_numeric:
        print(f"Ignoring column '{col}' — not a numeric type.")

    return continuous_features, discrete_features