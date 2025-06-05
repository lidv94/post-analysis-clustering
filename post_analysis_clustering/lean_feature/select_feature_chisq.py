import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from matplotlib.colors import ListedColormap,Normalize
from post_analysis_clustering.utils import timer
from post_analysis_clustering.lean_feature.model_creation import ModelCreation
from post_analysis_clustering.lean_feature.base import BaseLean

class LeanChiSquare(BaseLean):
    def __init__(self, 
                 df, 
                 features, 
                 target_cluster,
                 vote_score: int = 3
                ):
        
        self.df = df
        self.features = features
        self.target_cluster = target_cluster
        self.vote_score = vote_score
        
        # Call BaseLean init
        super().__init__(df, 
                         features, 
                         target_cluster,
                         vote_score=vote_score)
        
    def PrepBin(self,
                     method: str = "equal_range",  # or "neg_zero_pos"
                     n_bins: int = 5,
                     neg_n_bins: int = 5,
                     pos_n_bins: int = 5,
                     drop_original: bool = True
                     ) -> pd.DataFrame:
        """
        Bin numerical features using the selected method:
        - "equal_range": Equal-width binning for all values.
        - "neg_zero_pos": Separate binning for negative, zero, and positive values.

        Parameters:
            method (str): Binning method to use ("equal_range" or "neg_zero_pos").
            n_bins (int): Number of bins for equal_range method.
            neg_n_bins (int): Number of bins for negative values (neg_zero_pos).
            pos_n_bins (int): Number of bins for positive values (neg_zero_pos).
            drop_original (bool): Whether to drop original columns.

        Returns:
            pd.DataFrame: DataFrame with binned features as new columns.
        """
        binned_df = self.df.copy()

        for col in self.features:
            # Skip constant columns
            if binned_df[col].nunique(dropna=False) <= 1:
                unique_val = binned_df[col].dropna().unique()
                label = f"SingleValue: {unique_val[0]}" if len(unique_val) > 0 else "NaN"
                binned_df[f"{col}_bin"] = label
                continue

            if method == "equal_range":
                # Warn for negative values
                if (binned_df[col] < 0).any():
                    print(f"Warning: Negative values detected in '{col}'. Consider using 'neg_zero_pos' method.")

                bin_edges = np.round(np.linspace(
                    binned_df[col].min(), binned_df[col].max(), n_bins + 1), 2)

                cut_result = pd.cut(
                    binned_df[col].fillna(binned_df[col].median()),
                    bins=bin_edges,
                    duplicates='drop',
                    include_lowest=True
                )

                binned_df[f"{col}_bin"] = cut_result.astype(str).str.replace(
                    r'([\d\.-]+)', lambda m: f"{float(m.group()):.2f}", regex=True)

            elif method == "neg_zero_pos":
                zero_mask = binned_df[col] == 0
                negative_mask = binned_df[col] < 0
                positive_mask = binned_df[col] > 0

                binned_col = pd.Series(index=binned_df.index, dtype="object")

                # Negative binning
                if negative_mask.any():
                    neg_values = binned_df.loc[negative_mask, col].round(2)
                    neg_edges = np.round(np.linspace(neg_values.min(), neg_values.max(), neg_n_bins + 1), 2)

                    if len(np.unique(neg_edges)) == 1:
                        binned_col.loc[negative_mask] = f"SingleNegativeBin: {neg_values.min()}"
                    else:
                        cut_neg = pd.cut(neg_values, bins=neg_edges, include_lowest=True)
                        binned_col.loc[negative_mask] = cut_neg.astype(str).str.replace(
                            r'([\d\.-]+)', lambda m: f"{float(m.group()):.2f}", regex=True)

                # Zero values
                if zero_mask.any():
                    binned_col.loc[zero_mask] = "= 0"

                # Positive binning
                if positive_mask.any():
                    pos_values = binned_df.loc[positive_mask, col].round(2)
                    pos_edges = np.round(np.linspace(pos_values.min(), pos_values.max(), pos_n_bins + 1), 2)

                    if len(np.unique(pos_edges)) == 1:
                        binned_col.loc[positive_mask] = f"SinglePositiveBin: {pos_values.min()}"
                    else:
                        cut_pos = pd.cut(pos_values, bins=pos_edges, include_lowest=True)
                        binned_col.loc[positive_mask] = cut_pos.astype(str).str.replace(
                            r'([\d\.-]+)', lambda m: f"{float(m.group()):.2f}", regex=True)

                binned_df[f"{col}_bin"] = binned_col

            else:
                raise ValueError(f"Unknown method: {method}. Choose 'equal_range' or 'neg_zero_pos'.")

        if drop_original:
            binned_df.drop(columns=self.features, inplace=True)

        return binned_df


    
    
    
    
#     def RunModelCreation(self):
#         """
#         Runs the ModelCreation process and stores final importance results.
#         """
#         print("Running model creation and importance ranking...")

#         result = self.model_creation.run(
#             df=self.df,
#             features=self.features,
#             target_cluster=self.target_cluster,
#         )

#         # Extract from dict instead of tuple
#         self.final_imp = result["final_imp"]
#         self.final_imp_score = result["final_imp_score"]

#         print("Importance analysis completed successfully.")
#         print(f"final_imp shape: {self.final_imp.shape}")
#         print(f"final_imp_score shape: {self.final_imp_score.shape}")

#         return self.final_imp, self.final_imp_score
           
#     def GetLeanFeature(self, 
#                        final_imp_score: pd.DataFrame = None, 
#                        ):
#         """
#         Filters features by cluster and a threshold score for importance, and returns the remaining features for each cluster.
#         """
#         try:
#             if final_imp_score is None:
#                 if not hasattr(self, "final_imp_score")  or self.final_imp_score is None:
#                     print("No final_imp_score provided. Computing feature importance for all segments...")
#                     self._cal_imp_all_binary_class()
#                 final_imp_score = self.final_imp_score

#             df = final_imp_score.copy()
#             unique_segments = sorted(df['Segment'].unique())
#             cluster_lean_features_dict = {}
#             union_lean_feature_set = set()
            
#             print(f'Threshold of vote score >= {self.vote_score}')

#             for cluster in unique_segments:
#                 df_cluster = df[df['Segment'] == cluster].copy()
#                 df_cluster['sum_top_model'] = df_cluster.drop(columns=['Segment', 'Feature']).sum(axis=1)

#                 df_cluster = df_cluster[df_cluster['sum_top_model'] >= self.vote_score]
#                 lean_feature_list = sorted(df_cluster['Feature'].tolist())
#                 union_lean_feature_set.update(lean_feature_list)

#                 cluster_lean_features_dict[cluster] = lean_feature_list

#                 print(f"Cluster {cluster}:")
#                 print(f"  Total features from raw: {len(set(final_imp_score['Feature'].to_list()))}")
#                 print(f"  Total features remaining after threshold filter: {len(lean_feature_list)}")

#             union_lean_feature_list = sorted(list(union_lean_feature_set))
#             print(f"\nUnion across all clusters:")
#             print(f"  Total union features: {len(union_lean_feature_list)}")

#             return cluster_lean_features_dict, union_lean_feature_list

#         except Exception as e:
#             print(f"Error in filter_thres_features_by_cluster: {e}")
#             raise
        

# class LeanChiSquare(BaseLean):
#     def __init__(self):
#         ...
        
#     def run(self):
#         ...
        
#     def _plot(self):
#         ...
        
#     def plot(self):
#         ...
        

        
#     ################################################################################
    