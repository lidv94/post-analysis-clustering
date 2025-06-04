import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from matplotlib.colors import ListedColormap,Normalize
from post_analysis_clustering.utils import timer
from post_analysis_clustering.lean_feature.model_creation import ModelCreation
from post_analysis_clustering.lean_feature.base import BaseLean

class LeanImportanceRank(BaseLean):
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

    def RunModelCreation(self):
        """
        Runs the ModelCreation process and stores final importance results.
        """
        print("Running model creation and importance ranking...")

        result = self.model_creation.run(
            df=self.df,
            features=self.features,
            target_cluster=self.target_cluster,
        )

        # Extract from dict instead of tuple
        self.final_imp = result["final_imp"]
        self.final_imp_score = result["final_imp_score"]

        print("Importance analysis completed successfully.")
        print(f"final_imp shape: {self.final_imp.shape}")
        print(f"final_imp_score shape: {self.final_imp_score.shape}")

        return self.final_imp, self.final_imp_score
           
    def GetLeanFeature(self, 
                       final_imp_score: pd.DataFrame = None, 
                       ):
        """
        Filters features by cluster and a threshold score for importance, and returns the remaining features for each cluster.
        """
        try:
            if final_imp_score is None:
                if not hasattr(self, "final_imp_score")  or self.final_imp_score is None:
                    print("No final_imp_score provided. Computing feature importance for all segments...")
                    self._cal_imp_all_binary_class()
                final_imp_score = self.final_imp_score

            df = final_imp_score.copy()
            unique_segments = sorted(df['Segment'].unique())
            cluster_lean_features_dict = {}
            union_lean_feature_set = set()
            
            print(f'Threshold of vote score >= {self.vote_score}')

            for cluster in unique_segments:
                df_cluster = df[df['Segment'] == cluster].copy()
                df_cluster['sum_top_model'] = df_cluster.drop(columns=['Segment', 'Feature']).sum(axis=1)

                df_cluster = df_cluster[df_cluster['sum_top_model'] >= self.vote_score]
                lean_feature_list = sorted(df_cluster['Feature'].tolist())
                union_lean_feature_set.update(lean_feature_list)

                cluster_lean_features_dict[cluster] = lean_feature_list

                print(f"Cluster {cluster}:")
                print(f"  Total features from raw: {len(set(final_imp_score['Feature'].to_list()))}")
                print(f"  Total features remaining after threshold filter: {len(lean_feature_list)}")

            union_lean_feature_list = sorted(list(union_lean_feature_set))
            print(f"\nUnion across all clusters:")
            print(f"  Total union features: {len(union_lean_feature_list)}")

            return cluster_lean_features_dict, union_lean_feature_list

        except Exception as e:
            print(f"Error in filter_thres_features_by_cluster: {e}")
            raise
        

class LeanChiSquare(BaseLean):
    def __init__(self):
        ...
        
    def run(self):
        ...
        
    def _plot(self):
        ...
        
    def plot(self):
        ...
        
