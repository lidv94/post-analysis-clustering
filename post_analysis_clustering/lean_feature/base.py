class BaseLean:
    def __init__(self, df):
        self.df = df
    
    def _validate_inputs(self):
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(self.df)}")
        if not isinstance(self.features, list) or not all(isinstance(f, str) for f in self.features):
            raise TypeError("features must be a list of strings")
        if not isinstance(self.target_cluster, str):
            raise TypeError("target_cluster must be a string")
        for f in self.features:
            if f not in self.df.columns:
                raise ValueError(f"Feature '{f}' not found in DataFrame columns")
        if self.target_cluster not in self.df.columns:
            raise ValueError(f"target_cluster '{self.target_cluster}' not found in DataFrame columns")
        if not isinstance(self.n_rank, int) or self.n_rank < 1:
            raise ValueError("`n_rank` must be a positive integer.")
        if not isinstance(self.pct_thres, int) or self.pct_thres < 1:
            raise ValueError("`pct_thres` must be a positive integer.")   
        if not isinstance(self.vote_score, int) or not (1 <= self.vote_score <= len(self.models)):
            raise ValueError(f"`vote_score` must be a positive integer ≤ number of models ({len(self.models)})")