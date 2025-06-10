import pandas as pd
import numpy as np

class BaseLean:
    def __init__(self, 
                 df: pd.DataFrame,
                 features: list[str],
                 target_cluster: str,
                 models: dict = None,
                 n_rank: int = None,
                 pct_thres: float = None,
                 vote_score: int = None):
        # inputs
        self.df = df
        self.features = features
        self.target_cluster = target_cluster
        self.models = models
        
        # attributes
        self.n_rank = n_rank
        self.pct_thres = pct_thres
        self.vote_score = vote_score
        
        # validate immediately
        self._validate_inputs()
        self._validate_attributes()

    def _validate_inputs(self):
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"`df` must be a pandas DataFrame, got {type(self.df)}")
        if not isinstance(self.features, list) or not all(isinstance(f, str) for f in self.features):
            raise TypeError("`features` must be a list of strings")
        if not isinstance(self.target_cluster, str):
            raise TypeError("`target_cluster` must be a string")
        for f in self.features:
            if f not in self.df.columns:
                raise ValueError(f"Feature '{f}' not found in DataFrame columns")
        if self.target_cluster not in self.df.columns:
            raise ValueError(f"`target_cluster` '{self.target_cluster}' not found in DataFrame columns")
            
    def _validate_attributes(self):
        if self.n_rank is not None:
            if not isinstance(self.n_rank, int) or self.n_rank < 1:
                raise ValueError("`n_rank` must be a positive integer.")

        if self.pct_thres is not None:        
            if not isinstance(self.pct_thres, (float, int)) or not (0 < self.pct_thres <= 1):
                raise ValueError("`pct_thres` must be a float between 0 and 1.")

        if self.models is not None:
            if not isinstance(self.models, dict):
                raise TypeError("`models` must be a dictionary if provided.")
            if self.vote_score is not None:
                if not isinstance(self.vote_score, int) or not (1 <= self.vote_score <= len(self.models)):
                    raise ValueError(f"`vote_score` must be an integer between 1 and number of models ({len(self.models)})")
        elif self.vote_score is not None:
            if not isinstance(self.vote_score, int) or self.vote_score < 1:
                raise ValueError("`vote_score` must be a positive integer if models are not provided.")

            
