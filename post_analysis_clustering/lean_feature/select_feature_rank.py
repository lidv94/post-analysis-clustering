import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap,Normalize
from post_analysis_clustering.utils import timer
from post_analysis_clustering.lean_feature.model_creation import ModelCreation
from post_analysis_clustering.lean_feature.base import BaseLean

class LeanImportanceRank(BaseLean):
    """
    A class for performing permutation-based feature importance analysis
    using multiple classification models across cluster segments.

    This class integrates with `ModelCreation` to compute feature importance,
    rank features based on voting logic, and visualize results with heatmaps.

    Attributes:
        df (pd.DataFrame): Input DataFrame with features and cluster column.
        features (List[str]): List of feature names to evaluate.
        target_cluster (str): Name of the cluster assignment column.
        model_creation (ModelCreation): ModelCreation object that handles training and permutation.
        vote_score (int): Minimum number of models voting a feature as important.
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
        Runs the model training and permutation importance scoring process 
        using the ModelCreation object.

        Stores:
            - final_imp (pd.DataFrame): Raw permutation importance values for each feature.
            - final_imp_score (pd.DataFrame): Vote-based scoring of feature importance across models.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                final_imp and final_imp_score DataFrames.
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
    
    @timer       
    def GetLeanFeature(self, 
                       final_imp_score: pd.DataFrame = None, 
                       ):
        """
        Filters features for each cluster based on the vote threshold.

        If `final_imp_score` is not provided, it uses `self.final_imp_score`.

        Args:
            final_imp_score (pd.DataFrame, optional): Vote score DataFrame with columns:
                ['Segment', 'Feature', model_1, model_2, ..., model_n].

        Returns:
            Tuple[Dict[int, List[str]], List[str]]:
                - A dictionary mapping each cluster to its lean feature list.
                - A list of union features that passed the threshold in at least one cluster.

        Raises:
            Exception: If an error occurs during filtering.
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
        
    @timer
    def PlotHeatmapScore(self, 
                           final_imp: pd.DataFrame = None, 
                           compare_type: str = 'Normalized',
                           annot_type:str = 'Importance'
                          ):
        """
        Plots heatmaps of feature importances for each cluster and model.

        Args:
            final_imp (pd.DataFrame, optional): DataFrame with raw importance values.
                If None, uses self.final_imp.
            compare_type (str): One of {'global', 'percentage', 'normalized'}.
                - 'global': raw values.
                - 'percentage': column-wise percentage.
                - 'normalized': column-wise min-max scaling.
            annot_type (str): One of {'importance', 'rank'}.
                - 'importance': numeric values.
                - 'rank': ordinal ranks.

        Raises:
            ValueError: If `compare_type` or `annot_type` is invalid.
            Exception: If plotting fails.
        """
        try:            
            if final_imp is None:
                if not hasattr(self, 'final_imp') or self.final_imp is None:
                    print("No final_imp provided. Computing feature importance for all segments...")
                    self._cal_imp_all_binary_class()
                final_imp = self.final_imp

            df = final_imp.copy()
            unique_segments = sorted(df["Segment"].unique(), reverse=False)
            
            compare_type = compare_type.strip().lower()
            compare_valid_types = ['global', 'percentage', 'normalized']
            if compare_type not in compare_valid_types:
                raise ValueError(f"`compare_type` must be one of {compare_valid_types}.")
                
            annot_type = annot_type.strip().lower()
            annot_valid_types = ['importance', 'rank']
            if annot_type not in annot_valid_types:
                raise ValueError(f"`annot_type` must be one of {annot_valid_types}.")

            for segment in unique_segments:
                print(f"Plotting heatmap for segment {segment}")
                segment_data = df[df['Segment'] == segment]
                segment_data = segment_data.drop(columns='Segment').set_index("Feature")
                
                if annot_type== 'rank':
                    annot_data = segment_data.rank(ascending=False, axis=0).astype(int)
                    fmt_data = 'd'
                else: # importance data
                    annot_data = segment_data
                    fmt_data = '.2f'
                    
                
                if compare_type == 'global':
                    show_data = segment_data
                elif compare_type == 'percentage':
                    show_data = segment_data.div(segment_data.sum(axis=0), axis=1) * 100
                else:  # 'normalized'
                    show_data = (segment_data - segment_data.min()) / (segment_data.max() - segment_data.min())
                    

                # Plot heatmap
                plt.figure(figsize=(6, 5))
                sns.heatmap(show_data,
                            annot=annot_data,
                            fmt=fmt_data,
                            cmap=sns.light_palette("seagreen", as_cmap=True),
                            cbar=True,
                            linewidths=0.5)

                plt.title(f"{compare_type} Heatmap of Permutation Importance from Models\nCluster {segment} with {annot_type} Annotation",
                          fontsize=10, fontweight='bold')
                plt.xlabel("Classification Models", fontsize=12, fontweight='bold')
                plt.ylabel("Features", fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error in plot_heatmap_imp_all_binary_class : {e}")
            raise

    @timer
    def PlotVoteRank(self, 
                      final_imp_score: pd.DataFrame = None
                      ): 
        """
        Plots heatmaps of model voting results for each cluster

        Each cell represents the number of models that ranked a feature in the top-N with colors representing ranked importance..

        Args:
            final_imp_score (pd.DataFrame, optional): Voting score DataFrame.
                If None, uses self.final_imp_score.

        Raises:
            Exception: If plotting fails.
        """
        try: 
            if final_imp_score is None:
                if not hasattr(self, "final_imp_score")  or self.final_imp_score is None:
                    print("No final_imp_score provided. Computing feature importance for all segments...")
                    self._cal_imp_all_binary_class()
                final_imp_score = self.final_imp_score
                
            df = final_imp_score.copy()
            unique_segments = sorted(df["Segment"].unique())

            custom_order_palette = {
                0: "#ffffff",  # white
                1: "#e7f0f9", 2: "#d0e4f7", 3: "#a6c8ec", 4: "#7badde", 5: "#5192ce",
                6: "#2a77be", 7: "#1c5f9f", 8: "#144b85", 9: "#103c6c", 10: "#0e2f52",
                11: "#0b2340", 12: "#081f30", 13: "#041624", 14: "#02101a", 15: "#01080f"
            }

            custom_colors = [custom_order_palette[i] for i in sorted(custom_order_palette.keys())]
            custom_cmap = ListedColormap(custom_colors)

            for segment in unique_segments:
                print(f"Plotting heatmap for segment {segment}")

                segment_data = (
                    df[df["Segment"] == segment]
                    .drop(columns=["Segment"])
                    .set_index("Feature")
                )

                plt.figure(figsize=(6, 5))
                sns.heatmap(segment_data,
                            annot=True,
                            cmap=custom_cmap,
                            cbar=False,
                            linewidths=0.5,
                            vmin=0,
                            vmax=15)

                plt.title(f"Voting Results of Permutation Importance from Models for Cluster {segment}",
                          fontsize=10, fontweight='bold')
                plt.xlabel("Rank", fontsize=12, fontweight='bold')
                plt.ylabel("Feature", fontsize=12, fontweight='bold')
                plt.show()

        except Exception as e:
            print(f"Error in plot_vote_result_all_binary_class: {e}")
            raise        