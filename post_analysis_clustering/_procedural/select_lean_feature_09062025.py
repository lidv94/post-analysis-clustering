import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import chi2_contingency
from matplotlib.colors import ListedColormap,Normalize
from post_analysis_clustering.utils import timer

class LeanFeature:
    def __init__(self, 
                 df: pd.DataFrame, 
                 features: list[str], 
                 target_cluster: str,
                 n_rank:int=5,
                 pct_thres:float=80,
                 vote_score: int = 3,
                ):
        self.df = df
        self.features = features
        self.target_cluster = target_cluster
        self.models = self._initialize_models()
        self.n_rank = n_rank
        self.pct_thres = pct_thres
        self.vote_score = vote_score
        self.final_imp = None
        self.final_pvt_imp_score = None
        self.final_cumsum = None
        self.bin_cumsum_df = None
        self.final_pvt_cumsum_score = None
        self._validate_inputs() 

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

    def _initialize_models(self):
        return {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, early_stopping=True),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=42),
            "Naive Bayes": GaussianNB()
        }

    @timer
    def _calculate_permutation_importance(self, model,X_test,y_test):
        # Calculate Permutation Importance for a given model.
        try:
            perm_importance = permutation_importance(model, 
                                                     X_test, 
                                                     y_test, 
                                                     n_repeats=10, 
                                                     random_state=42)
            return pd.DataFrame({
                "Feature": self.features,
                "Importance": perm_importance.importances_mean
            }).sort_values(by="Importance", ascending=False)
        except Exception as e:
            raise RuntimeError(f"Failed to compute permutation importance: {e}")

    @timer
    def _train_and_evaluate_model(self, model, X_train, X_test, y_train, y_test):
        # Train a given model, evaluate accuracy, and compute permutation importance.
        try:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            class_report = classification_report(y_test, test_pred, output_dict=True)
            importance_df = self._calculate_permutation_importance(model,X_test,y_test)

            return importance_df, train_acc, test_acc, class_report
        except Exception as e:
            raise RuntimeError(f"Error training model: {e}")

    @timer
    def _run_all_models(self, X_train, X_test, y_train, y_test):
        # Train multiple models, evaluate performance, and calculate permutation importance.
        importance_results = {}
        performance = {}
        classification_reports = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            importance_df, train_acc, test_acc, report =  self._train_and_evaluate_model(model, X_train, X_test, y_train, y_test) 
            importance_results[name] = importance_df
            performance[name] = {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
            classification_reports[name] = report

        return importance_results, performance, classification_reports
    
    @timer
    def _prep_binary_class(self):
        # Prepares binary classification labels for each cluster segment by converting the target cluster into binary columns.
        try:
            binary_df = self.df.copy() 
            for cluster_label in sorted(self.df[self.target_cluster].unique()):
                binary_df[f'is_cluster_{cluster_label}'] = (self.df[self.target_cluster] == cluster_label).astype(int)
            return binary_df
        except Exception as e:
            print(f"Error in _prep_binary_class : {e}")
            raise
    
    @timer
    def _cal_imp_one_binary_class(self,focus_segment):
        # Performs binary classification to evaluate feature importance for a specific cluster segment.
        try:
            binary_df = self._prep_binary_class()
            y = binary_df[f'is_cluster_{focus_segment}']
            X = binary_df[self.features]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            importance_results, performance, classification_reports = self._run_all_models(X_train, X_test, y_train, y_test)

            print(f"\n### Feature Importance : One-vs-All Classification for Cluster {focus_segment} ###")

            all_features = set()
            for df in importance_results.values():
                all_features.update(df['Feature'])

            importance_df = pd.DataFrame(index=sorted(all_features))

            for model, df in importance_results.items():
                model_importance = df.set_index('Feature')['Importance']
                importance_df[model] = importance_df.index.map(model_importance)

            importance_df.reset_index(inplace=True)
            importance_df.rename(columns={'index': 'Feature'}, inplace=True)

            return importance_df # only one segment
        except Exception as e:
            print(f"Runtime error in cal_imp_one_binary_class : {e}")
            raise


    @timer
    def _prep_rank_importance(self,focus_segment) -> pd.DataFrame:
        # Ranks features based on their importance scores across different models.
        try:
            importance_df = self._cal_imp_one_binary_class(focus_segment)
            melt_df = pd.melt(importance_df, id_vars="Feature", value_vars=importance_df.drop("Feature", axis=1).columns.tolist()) 
            
            melt_df = melt_df.sort_values(by=["variable", "value"], ascending=[True, False])
            
            rank_df = melt_df.groupby(['variable'])['value'].rank(method="dense", ascending=False)
            melt_df['rank'] = rank_df.astype(int)  
            
            # Only positive values contribute to cumsum
            melt_df['value_for_cumsum'] = melt_df['value'].where(melt_df['value'] > 0, 0)
            melt_df['cumsum'] = melt_df.groupby("variable")["value_for_cumsum"].cumsum()

            # Total also from positive values only
            melt_df["total"] = melt_df.groupby("variable")["value_for_cumsum"].transform("sum")
            melt_df["cumsum_pct"] = melt_df["cumsum"] / melt_df["total"]

            melt_df.drop(columns=["total", "value_for_cumsum"], inplace=True)
            return melt_df

        except Exception as e:
            print(f"Runtime error in prep_rank_importance : {e}")
            raise
            
    @timer
    def _pivot_rank_importance(self,focus_segment):
        # Creates a pivot table summarizing the ranks of features across models.
        try:
            melt_df = self._prep_rank_importance(focus_segment=focus_segment)
            pvt_imp = pd.pivot_table(melt_df,
                                     index='Feature',
                                     columns='rank', # rank_abs
                                     values='variable',
                                     fill_value=0,
                                     aggfunc='count')
            top_n_pvt_imp = pvt_imp.loc[:, :self.n_rank]
            return top_n_pvt_imp

        except Exception as e:
            print(f"Runtime error in pivot_rank_importance : {e}")
            raise

    @timer
    def _cal_imp_all_binary_class(self):
        """
        Computes feature importance for all unique segments in the target cluster column using a binary classification approach.
        """    
        try:
            unique_segments = sorted(self.df[self.target_cluster].unique(), reverse=False)
            all_imps = []
            all_pvt_imps_score = []
            all_cumsum = []

            for segment in unique_segments:
                print(f"Processing segment {segment}")

                # Compute feature importance for the given segment
                importance_df = self._cal_imp_one_binary_class(focus_segment=segment)
                
                # get melt for cumsum method
                melt_df = self._prep_rank_importance(focus_segment=segment)
                melt_df['Segment'] = segment
                all_cumsum.append(melt_df)

                # Get top N important features for imp rank method
                pvt_imp = self._pivot_rank_importance(focus_segment=segment)

                # Add segment identifier
                importance_df["Segment"] = segment
                all_imps.append(importance_df)

                # Add segment identifier
                pvt_imp["Segment"] = segment
                all_pvt_imps_score.append(pvt_imp)

            # Combine results
            final_imp = pd.concat(all_imps, axis=0).reset_index(drop=True)
            final_pvt_imp_score = pd.concat(all_pvt_imps_score, axis=0).reset_index()
            final_pvt_imp_score.columns.name = None
            final_cumsum = pd.concat(all_cumsum).reset_index(drop=True)

            # Sort columns: Feature, 1, 2, ..., Segment
            fixed_cols = ['Feature']
            numeric_cols = sorted([col for col in final_pvt_imp_score.columns if isinstance(col, int)])
            fixed_cols += numeric_cols + ['Segment']
            final_pvt_imp_score = final_pvt_imp_score[fixed_cols]

            # Fill NaNs with 0 and convert numeric columns to integers
            final_pvt_imp_score = final_pvt_imp_score.fillna(0)
            cols_to_convert = [col for col in final_pvt_imp_score.columns if col not in ['Feature', 'Segment']]
            final_pvt_imp_score[cols_to_convert] = final_pvt_imp_score[cols_to_convert].astype(int)
            
            self.final_imp = final_imp
            self.final_pvt_imp_score = final_pvt_imp_score
            self.final_cumsum = final_cumsum
            
            return self.final_imp, self.final_pvt_imp_score,self.final_cumsum

        except Exception as e:
            print(f"Error in cal_imp_all_binary_class : {e}")
            raise
            
    def _bin_cumsum_percentiles(self,
                                final_cumsum: pd.DataFrame=None, 
                                bin_size: int = 10
                               ) -> pd.DataFrame:
        """
        Bins `cumsum_pct` into percentile intervals and counts model occurrences per bin per feature & segment.
        """
        if final_cumsum is None:
            if not hasattr(self, "final_cumsum")  or self.final_cumsum is None:
                print("No final_cumsum provided. Computing feature importance for all segments...")
                self._cal_imp_all_binary_class()
            final_cumsum = self.final_cumsum
            
        df = final_cumsum.copy()

        # Ensure cumsum_pct is within [0, 1]
        df['cumsum_pct'] = df['cumsum_pct'].clip(0, 1)
        
        bin_edges = list(range(0, 100, bin_size)) + [99.99, 100]  # [0,20,40,60,80,99.99,100]

        bin_labels = [f"pct_{i+bin_size}" for i in range(0, 100 - bin_size, bin_size)] + ['pct_99.99', 'pct_100']

        df['bin'] = pd.cut(
            df['cumsum_pct'] * 100,
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=True
        )

        # Count models per bin per Feature and Segment
        count_df = (
            df.groupby(['Feature', 'Segment', 'bin'], observed=False)
              .size()
              .reset_index(name='count')
                )

        bin_cumsum_df =  count_df.pivot_table(index=['Feature', 'Segment'], 
                                     columns='bin', 
                                     values='count', 
                                     fill_value=0, 
                                     observed=False).reset_index()

        bin_cumsum_df['single_imp'] = np.where(bin_cumsum_df['pct_100']==len(self.models),
                                               bin_cumsum_df['pct_100'],
                                               0).astype(int)        
        # Reorder columns: Feature, Segment, cnt_models_all_one, bin columns
        ordered_cols = ['Feature']+['single_imp'] + bin_labels +['Segment']
        bin_cumsum_df = bin_cumsum_df.reindex(columns=ordered_cols)
        bin_cumsum_df[bin_labels] = bin_cumsum_df[bin_labels].astype(int)
        bin_cumsum_df = bin_cumsum_df.sort_values(by=["Segment", "Feature"], ascending=[True, False]).reset_index(drop=True)
        
        #########################################################
        
        print(f"Using threshold of {self.pct_thres}% to determine feature importance across models.")
        
        # Create threshold bin labels directly from the df
        df['bin_thres'] = np.where(
            df['cumsum_pct'] * 100 < self.pct_thres,
            f"pct=<thres",
            f"pct>thres"
        )

        # Count by Feature, Segment, bin_thres
        filtered_count_df = (
            df.groupby(['Feature', 'Segment', 'bin_thres'], observed=False)
              .size()
              .reset_index(name='count')
        )

        # Pivot to wide format: <thres / >=thres as columns
        final_pvt_cumsum_score = filtered_count_df.pivot_table(
            index=['Feature', 'Segment'],
            columns='bin_thres',
            values='count',
            fill_value=0,
            observed=False
        ).reset_index()

        # Add 'single_imp' column from bin_cumsum_df
        final_pvt_cumsum_score = final_pvt_cumsum_score.merge(
            bin_cumsum_df[['Feature', 'Segment', 'single_imp']],
            on=['Feature', 'Segment'],
            how='left'
        )

        # Reorder columns
        cols = ['Feature']+['single_imp']+ ['pct=<thres']+['pct>thres']+['Segment']
        final_pvt_cumsum_score = final_pvt_cumsum_score.reindex(columns=cols, fill_value=0)
        final_pvt_cumsum_score = final_pvt_cumsum_score.sort_values(by=["Segment", "Feature"], ascending=[True, False]).reset_index(drop=True)
        final_pvt_cumsum_score['pct=<thres'] = final_pvt_cumsum_score['pct=<thres'].astype(int)
        final_pvt_cumsum_score['pct>thres'] = final_pvt_cumsum_score['pct>thres'].astype(int)
        
        final_pvt_cumsum_score = final_pvt_cumsum_score.loc[:,['Feature','single_imp','pct=<thres','Segment']]
        
        self.bin_cumsum_df = bin_cumsum_df
        self.final_pvt_cumsum_score = final_pvt_cumsum_score
        
        
        return self.bin_cumsum_df , self.final_pvt_cumsum_score


    @timer
    def _plot_heatmap_imp_all_binary_class(self, 
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
    def _plot_vote_result_all_binary_class(self, 
                                          final_pvt_imp_score: pd.DataFrame = None
                                          ):
        '''
          Plots heatmaps for each segment using precomputed feature importance scores, 
          with colors representing ranked importance.
        '''  
        try: 
            if final_pvt_imp_score is None:
                if not hasattr(self, "final_pvt_imp_score")  or self.final_pvt_imp_score is None:
                    print("No final_pvt_imp_score provided. Computing feature importance for all segments...")
                    self._cal_imp_all_binary_class()
                final_pvt_imp_score = self.final_pvt_imp_score
                
            df = final_pvt_imp_score.copy()
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
            
    @timer
    def ImportanceRank_imp_features(self, 
                                         final_pvt_imp_score: pd.DataFrame = None, 
                                        ):
        """
        Filters features by cluster and a threshold score for importance, and returns the remaining features for each cluster.
        """
        try:
            if final_pvt_imp_score is None:
                if not hasattr(self, "final_pvt_imp_score")  or self.final_pvt_imp_score is None:
                    print("No final_pvt_imp_score provided. Computing feature importance for all segments...")
                    self._cal_imp_all_binary_class()
                final_pvt_imp_score = self.final_pvt_imp_score

            df = final_pvt_imp_score.copy()
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
                print(f"  Total features from raw: {len(set(final_pvt_imp_score['Feature'].to_list()))}")
                print(f"  Total features remaining after threshold filter: {len(lean_feature_list)}")

            union_lean_feature_list = sorted(list(union_lean_feature_set))
            print(f"\nUnion across all clusters:")
            print(f"  Total union features: {len(union_lean_feature_list)}")

            return cluster_lean_features_dict, union_lean_feature_list

        except Exception as e:
            print(f"Error in filter_thres_features_by_cluster: {e}")
            raise
            
    @timer
    def ImportanceThreshold_imp_features(self, 
                                        final_pvt_cumsum_score: pd.DataFrame = None
                                           ) :
        """
        Filters features by cluster based on the sum of importance votes across all bins (e.g., `single_imp`, `pct=<thres>`, `pct=>thres`),
        and returns features where the vote sum is >= self.vote_score.
        """
        try:
            if final_pvt_cumsum_score is None:
                if not hasattr(self, "final_pvt_cumsum_score") or self.final_pvt_cumsum_score is None:
                    print("No final_pvt_cumsum_score provided. Computing binned importance...")
                    self._bin_cumsum_percentiles()
                final_pvt_cumsum_score = self.final_pvt_cumsum_score

            df = final_pvt_cumsum_score.copy()
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
                print(f"  Total features from raw: {len(set(final_pvt_cumsum_score['Feature'].to_list()))}")
                print(f"  Total features remaining after threshold filter: {len(lean_feature_list)}")

            union_lean_feature_list = sorted(list(union_lean_feature_set))
            print(f"\nUnion across all clusters:")
            print(f"  Total union features: {len(union_lean_feature_list)}")

            return cluster_lean_features_dict, union_lean_feature_list

        except Exception as e:
            print(f"Error in ImportanceThreshold: {e}")
            raise
    
    ################################################################################
    
    # 1. Binning Function (equal range)
    @timer
    def _bin_features(self,
                    n_bins: int = 5,
                    drop_original: bool = True
                    ) -> pd.DataFrame:
        """
        Bin numerical features into equal-width intervals.
        If negative values are found, suggest using bin_features_neg_zero_pos instead.
        """
        binned_df = self.df.copy()

        for col in self.features:
            try:
                # Handle edge cases: NaNs and identical values
                if binned_df[col].nunique() <= 1:
                    binned_df[f"{col}_bin"] = "SingleValue"
                    continue

                if (binned_df[col] < 0).any():
                    print(f"Warning: Negative values detected in column '{col}'. Suggest using 'bin_features_neg_zero_pos' instead.")

                bin_edges = np.round(np.linspace(binned_df[col].min(), binned_df[col].max(), n_bins + 1), 2)
                binned_series = pd.cut(
                    binned_df[col].fillna(binned_df[col].median()),
                    bins=bin_edges,
                    duplicates='drop',
                    include_lowest=True
                )

                # Format bin labels to 2 decimal places
                binned_df[f"{col}_bin"] = binned_series.astype(str).str.replace(
                    r'([\d\.-]+)', lambda m: f"{float(m.group()):.2f}", regex=True)
            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                binned_df[f"{col}_bin"] = "Error"

        if drop_original:
            binned_df.drop(columns=self.features, inplace=True)

        return binned_df

    @timer
    def _bin_features_neg_zero_pos(self,
                                    pos_n_bins: int = 5,
                                    neg_n_bins: int = 5,
                                    drop_original: bool = True
                                ) -> pd.DataFrame:
        """
        Bin numerical features into separate intervals for negative, zero, and positive values.

        - Negative values are binned into negative bins.
        - Zero values are labeled as "= 0".
        - Positive values are binned into positive bins.

        This function allows for separate binning of negative and positive values, with customizable
        bin counts for both groups.
        """
        binned_df = self.df.copy()

        for col in self.features:
            try:
                # Handle columns with only one unique value
                if binned_df[col].nunique() <= 1:
                    unique_val = binned_df[col].dropna().unique()[0] if binned_df[col].notna().any() else "NaN"
                    binned_df[f"{col}_bin"] = f"SingleValue: {unique_val}"
                    continue

                zero_mask = binned_df[col] == 0
                negative_mask = binned_df[col] < 0
                positive_mask = binned_df[col] > 0

                zero_part = binned_df.loc[zero_mask].copy()
                negative_part = binned_df.loc[negative_mask].copy()
                positive_part = binned_df.loc[positive_mask].copy()

                binned_col = pd.Series(index=binned_df.index, dtype="object")

                # Handle negative values
                if not negative_part.empty:
                    neg_values = negative_part[col].round(2)
                    neg_bin_edges = np.round(np.linspace(neg_values.min(), neg_values.max(), neg_n_bins + 1), 2)

                    if len(np.unique(neg_bin_edges)) == 1:
                        negative_part[f"{col}_bin"] = f"SingleNegativeBin: {neg_values.min()}"
                    else:
                        cut_result = pd.cut(
                            neg_values,
                            bins=neg_bin_edges,
                            include_lowest=True
                        )
                        negative_part[f"{col}_bin"] = cut_result.astype(str).str.replace(
                            r'([\d\.-]+)', lambda m: f"{float(m.group()):.2f}", regex=True)

                    binned_col.loc[negative_part.index] = negative_part[f"{col}_bin"]

                # Handle zero values
                if not zero_part.empty:
                    binned_col.loc[zero_part.index] = "= 0"

                # Handle positive values
                if not positive_part.empty:
                    pos_values = positive_part[col].round(2)
                    pos_bin_edges = np.round(np.linspace(pos_values.min(), pos_values.max(), pos_n_bins + 1), 2)

                    if len(np.unique(pos_bin_edges)) == 1:
                        positive_part[f"{col}_bin"] = f"SinglePositiveBin: {pos_values.min()}"
                    else:
                        cut_result = pd.cut(
                            pos_values,
                            bins=pos_bin_edges,
                            include_lowest=True
                        )
                        positive_part[f"{col}_bin"] = cut_result.astype(str).str.replace(
                            r'([\d\.-]+)', lambda m: f"{float(m.group()):.2f}", regex=True
                        )

                    binned_col.loc[positive_part.index] = positive_part[f"{col}_bin"]

                binned_df[f"{col}_bin"] = binned_col

            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                binned_df[f"{col}_bin"] = "Error"

        if drop_original:
            binned_df.drop(columns=self.features, inplace=True)

        return binned_df

    # 3. Chi-square Test Function
    @timer
    def test_chi_square_segment_vs_rest(
        df: pd.DataFrame, 
        features: list[str], 
        target_cluster: str, 
        binary_target_prefix: str = "is_cluster_", 
        n_bins: int = 5, 
        bin_suffix: str = "_bin", 
        bin_type: str = 'handle_neg_zero_pos'
        ):
        """
        Perform Chi-square tests between binned features and cluster binary segments.

        The function bins the given features and splits the data into segments based on the specified 
        target cluster. It then performs a Chi-square test for independence between each binned feature 
        and the binary cluster segment. The results are returned as p-values in a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing features to be binned and analyzed.
            features (list of str): List of feature names to bin and analyze.
            target_cluster (str): Column name representing the target cluster.
            binary_target_prefix (str, optional): Prefix to identify binary cluster columns. Defaults to "is_cluster_".
            n_bins (int, optional): Number of bins to create for positive values. Defaults to 5.
            bin_suffix (str, optional): Suffix for the binned feature columns. Defaults to "_bin".
            bin_type (str, optional): Type of binning method. Options:
                - 'normal': standard binning with normal cut
                - 'handle_neg_zero_pos': handles negative, zero, and positive values separately (default).

        Returns:
            pd.DataFrame: DataFrame containing p-values for each binned feature with each cluster segment.

        Example:
            pval_df = test_chi_square_segment_vs_rest(
                df=my_df, 
                features=["feature1", "feature2"], 
                target_cluster="cluster_id")

        Notes:
            - A p-value less than 0.05 indicates a significant relationship between the feature and the segment.
        """
        try:
            # Step 1 Preprocess data to create binary columns for cluster segments
            df_with_binary = prep_binary_class(df=df, features=features, target_cluster=target_cluster)

            # Step 2 Bin features using the specified binning method
            if bin_type == 'normal':
                df_binned = bin_features(df=df_with_binary, features=features, n_bins=n_bins)
            else: # Default: 'handle_neg_zero_pos'
                df_binned = bin_features_neg_zero_pos(df=df_with_binary, features=features, pos_n_bins=n_bins, neg_n_bins=5)

            result_dict = {}

            # Step 3 Identify binary columns that represent the cluster segments
            binary_columns = [col for col in df_binned.columns if col.startswith(binary_target_prefix)]
            # Step 4: Identify binned feature columns
            binned_features = [col for col in df_binned.columns if col.endswith(bin_suffix)]

            print(f'List of binary class columns : {binary_columns}')
            print(f'List of binned features : {binned_features}')

            # Step 5: Loop over each binary segment column and test against binned features
            for bin_col in binary_columns:
                segment_label = bin_col.replace(binary_target_prefix, '')
                p_values = {}

                for feature in binned_features:
                    try:
                        contingency = pd.crosstab(df_binned[feature], df_binned[bin_col])
                        if contingency.shape[0] > 1 and contingency.shape[1] == 2:
                            _, p, _, _ = chi2_contingency(contingency)
                        else:
                            p = np.nan
                    except Exception as e:
                        warnings.warn(f"Chi-square failed for feature '{feature}' and segment '{segment_label}': {e}")
                        p = np.nan # If insufficient data, set p-value to NaN

                    p_values[feature] = p

                # Step 6: Store p-values for each segment in the result dictionary
                result_dict[segment_label] = p_values

            return pd.DataFrame(result_dict)

        except Exception as e:
            raise RuntimeError(f"[Chi-Square Segment Test] Failed due to: {e}")
