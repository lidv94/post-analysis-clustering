import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap,Normalize
from post_analysis_clustering.utils import timer
from post_analysis_clustering.lean_feature.model_creation import ModelCreation
from post_analysis_clustering.lean_feature.base import BaseLean

class LeanImportanceThreshold(BaseLean):
    """
    Performs feature selection based on cumulative importance voting across 
    binned permutation results from multiple models and thresholds.

    This class is designed for post-clustering analysis to identify lean 
    (important) features by evaluating the cumulative voting across thresholds 
    (e.g., individual importance, below/above percentile thresholds).

    Attributes:
        df (pd.DataFrame): Input dataset.
        features (List[str]): List of feature column names.
        target_cluster (str): Name of the column indicating cluster membership.
        model_creation (ModelCreation): Model creation and importance computation handler.
        vote_score (int): Threshold number of votes a feature must receive to be considered important.
    """
    def __init__(self, 
                 df, 
                 features, 
                 target_cluster, 
                 model_creation: ModelCreation,
                 vote_score: int=3):
        
        # Pull values from model_creation
        models = model_creation.models
        n_rank = model_creation.n_rank
        pct_thres = model_creation.pct_thres

        # Call BaseLean init
        super().__init__(df, features, target_cluster, models, n_rank, pct_thres, vote_score)

        self.model_creation = model_creation
    
    @timer
    def RunModelCreation(self):
        """
        Executes the model training and feature importance computation.

        Delegates execution to the `ModelCreation` object and stores the following:
          - `final_cumsum_bin`: Raw binned importance scores per feature/cluster.
          - `final_cumsum_score`: Aggregated vote counts across thresholds.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - final_cumsum_bin: Detailed importance values.
                - final_cumsum_score: Vote count summary by feature and cluster.
        """
        print("Running model creation and importance ranking...")

        result = self.model_creation.run(
            df=self.df,
            features=self.features,
            target_cluster=self.target_cluster,
        )

        # Extract from dict instead of tuple
        self.final_cumsum_bin = result["final_cumsum_bin"]
        self.final_cumsum_score = result["final_cumsum_score"]

        print("Importance analysis completed successfully.")
        print(f"final_cumsum_bin shape: {self.final_cumsum_bin.shape}")
        print(f"final_cumsum_score shape: {self.final_cumsum_score.shape}")

        return self.final_cumsum_bin, self.final_cumsum_score
    
    @timer
    def GetLeanFeature(self, 
                        final_cumsum_score: pd.DataFrame = None
                           ) :
        """
        Filters features for each cluster based on the cumulative vote score across binned importance levels.

        If `final_cumsum_score` is not provided, uses `self.final_cumsum_score` 
        (computed via `_bin_cumsum_percentiles()` if necessary).

        Features are retained if their total vote score is greater than or equal to `self.vote_score`.

        Args:
            final_cumsum_score (pd.DataFrame, optional): DataFrame with vote scores for each feature and segment.
                Must include columns: ['Segment', 'Feature', model_1, model_2, ..., model_n].

        Returns:
            Tuple[Dict[int, List[str]], List[str]]:
                - Dictionary mapping each cluster ID to a list of selected features.
                - Union list of all selected features across all clusters.

        Raises:
            Exception: If computation fails due to missing data or other errors.
        """
        try:
            if final_cumsum_score is None:
                if not hasattr(self, "final_cumsum_score") or self.final_cumsum_score is None:
                    print("No final_cumsum_score provided. Computing binned importance...")
                    self._bin_cumsum_percentiles()
                final_cumsum_score = self.final_cumsum_score

            df = final_cumsum_score.copy()
            unique_segments = sorted(df['Segment'].unique())
            cluster_lean_features_dict = {}
            union_lean_feature_set = set()

            print(f'Threshold of vote score >= {self.vote_score}')

            for cluster in unique_segments:
                df_cluster = df[df['Segment'] == cluster].copy()

                # Sum across all model vote columns
                df_cluster['sum_vote_score'] = df_cluster.drop(columns=['Segment', 'Feature']).sum(axis=1)

                # Filter by threshold
                df_cluster_filtered = df_cluster[df_cluster['sum_vote_score'] >= self.vote_score]
                lean_feature_list = sorted(df_cluster_filtered['Feature'].tolist())

                # Store results
                cluster_lean_features_dict[cluster] = lean_feature_list
                union_lean_feature_set.update(lean_feature_list)
                union_lean_feature_list = sorted(list(union_lean_feature_set))
                print(f"Cluster {cluster}:")
                print(f"  Total features from raw: {len(set(final_cumsum_score['Feature'].to_list()))}")
                print(f"  Total features remaining after threshold filter: {len(lean_feature_list)}")

            union_lean_feature_list = sorted(list(union_lean_feature_set))
            print(f"\nUnion across all clusters:")
            print(f"  Total union features: {len(union_lean_feature_list)}")

            return cluster_lean_features_dict, union_lean_feature_list

        except Exception as e:
            print(f"Error in ImportanceThreshold: {e}")
            raise
