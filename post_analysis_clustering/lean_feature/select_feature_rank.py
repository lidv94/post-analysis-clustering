import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        
    @timer
    def PlotHeatmapScore(self, 
                           final_imp: pd.DataFrame = None, 
                           compare_type: str = 'Normalized',
                           annot_type:str = 'Importance'
                          ):
        """
        Plots a heatmap of feature importance scores for multiple models, segmented by cluster.

        Parameters:
            final_imp (pd.DataFrame, optional): Importance DataFrame from _cal_imp_all_binary_class. 
                                                If None, uses self.final_imp or computes it.
            compare_type (str): One of ['global', 'percentage', 'normalized'] to scale heatmap values.
            annot_type (str): One of ['importance', 'rank'] to show annotation on heatmap.
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
        '''
          Plots heatmaps for each segment using precomputed feature importance scores, 
          with colors representing ranked importance.
        '''  
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