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
def get_feature_list(df: pd.DataFrame) -> list:
    """
    Extract feature list by removing the first (primary key) and last (cluster) columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of feature column names (excluding first and last columns).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.shape[1] < 3:
        raise ValueError("DataFrame must have at least 3 columns which are primary_key, ...features..., target_cluster.")

    try:
        return df.columns[1:-1].tolist()
    except Exception as e:
        raise RuntimeError(f"Error in get_feature_list: {e}")

@timer    
def get_feature_list_exc_k(df: pd.DataFrame) -> list:
    """
    Extract feature list by removing:
    - The first (primary key) column.
    - Any columns that start with 'K' followed by numbers (e.g., K2, K10).

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of feature column names excluding 'K{number}' columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.shape[1] < 3:
        raise ValueError("DataFrame must have at least 3 columns which are primary_key, ...features..., target_cluster.")
    try:
        feature_list = df.columns[1:]
        return [col for col in feature_list if not re.match(r"^K\d+$", col)]
    except Exception as e:
        raise RuntimeError(f"Error in get_feature_list_exc_k: {e}")

#####################################################################################

@timer
def get_model_centroids(model) -> list | None:
    """
    Extract centroids from a clustering model.

    Args:
        model (Any): Trained clustering model object.

    Returns:
        list or None: List of centroids if available, otherwise None.
    """
    if model is None:
        raise ValueError("Input model cannot be None.")

    try:
        return model.cluster_centers_.tolist() if hasattr(model, "cluster_centers_") else None
    except Exception as e:
        raise RuntimeError(f"Error extracting centroids: {e}")
