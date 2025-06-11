import pandas as pd
import numpy as np

class BaseVis:
    """
    Base visualization class for post-clustering analysis.

    Provides input validation and stores key analysis parameters including features,
    cluster labels, and primary identifiers for downstream visualizations.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing all data.
        features (list[str]): List of feature column names to analyze or visualize.
        target_cluster (str): Column name indicating the cluster assignments.
        primary_key (str): Unique identifier for each row (e.g., customer ID).
    """        
    def __init__(self, 
                 df: pd.DataFrame,
                 features: list[str],
                 target_cluster: str,
                 primary_key: str,
                ):
        # inputs
        self.df = df
        self.features = features
        self.target_cluster = target_cluster
        self.primary_key = primary_key
        
        # validate immediately
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validates that the input types and column names are correct.

        Ensures that:
            - `df` is a pandas DataFrame
            - `features` is a list of strings
            - `target_cluster` and `primary_key` are string column names
            - All feature, cluster, and key columns exist in the DataFrame

        Raises:
            TypeError: If any input is of an incorrect type.
            ValueError: If required columns are missing from the DataFrame.
        """
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"`df` must be a pandas DataFrame, got {type(self.df)}")
        if not isinstance(self.features, list) or not all(isinstance(f, str) for f in self.features):
            raise TypeError("`features` must be a list of strings")
        if not isinstance(self.target_cluster, str):
            raise TypeError("`target_cluster` must be a string")
        if not isinstance(self.primary_key, str):
            raise TypeError("`primary_key` must be a string")

        for f in self.features:
            if f not in self.df.columns:
                raise ValueError(f"Feature '{f}' not found in DataFrame columns")
        if self.target_cluster not in self.df.columns:
            raise ValueError(f"`target_cluster` '{self.target_cluster}' not found in DataFrame columns")
        if self.primary_key not in self.df.columns:
            raise ValueError(f"`primary_key` '{self.primary_key}' not found in DataFrame columns")

