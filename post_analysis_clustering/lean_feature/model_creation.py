import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from post_analysis_clustering.utils import timer

class ModelCreation:
    """
    A class for evaluating feature importance across multiple models 
    using permutation importance on binary classification tasks derived 
    from multi-class cluster labels.

    This class supports ranking features by model-specific importance, 
    aggregating results across clusters, and generating bin-based summaries.

    Attributes:
        models (dict[str, sklearn.base.BaseEstimator]): 
            Dictionary of model names and scikit-learn classifier instances.
        n_rank (int): 
            Number of top-ranked features to consider for importance summary.
        pct_thres (float): 
            Threshold (0 to 100) for cumulative importance percentage used 
            in binning and scoring.
    """
    def __init__(self, 
                 models: dict,
                 n_rank: int=5,
                 pct_thres:float=80,
                ):
        """
        Initializes the ModelCreation class with the given models and configuration.

        Args:
            models (dict[str, sklearn.base.BaseEstimator]):
                A dictionary mapping model names to scikit-learn classifier objects.
                Example:
                    models = {
                            "Decision Tree": DecisionTreeClassifier(random_state=42),
                            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, early_stopping=True),
                            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
                            "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=42),
                            "Naive Bayes": GaussianNB()
                            }
            n_rank (int, optional):
                The number of top-ranked features to include in the rank summary. 
                Defaults to 5.
            pct_thres (float, optional):
                Threshold for cumulative importance binning (0-100). Determines 
                whether a feature is considered important across models. 
                Defaults to 80.
        
        Raises:
            TypeError: If models is not a dictionary.
            ValueError: If `n_rank` is not a positive integer or `pct_thres` is not within (0, 100].
        """
        self.models = models
        self.n_rank = n_rank
        self.pct_thres = pct_thres
        
    @timer
    def _calculate_permutation_importance(self, features, model, X_test, y_test):
        """
        Calculates permutation feature importance for a trained model.

        Args:
            features (list[str]): List of feature names.
            model (sklearn.base.BaseEstimator): Trained model to evaluate.
            X_test (pd.DataFrame): Test feature data.
            y_test (pd.Series): Test target labels.

        Returns:
            pd.DataFrame: A dataframe containing feature names and their corresponding importance scores.

        Example:
            importance_df = self._calculate_permutation_importance(features, model, X_test, y_test)
        """
        perm_importance = permutation_importance(model,
                                                 X_test, 
                                                 y_test, 
                                                 n_repeats=10, 
                                                 random_state=42)
        return pd.DataFrame({
            "Feature": features,
            "Importance": perm_importance.importances_mean
        }).sort_values(by="Importance", ascending=False)

    @timer
    def _train_and_evaluate_model(self,features, model, X_train, X_test, y_train, y_test):
        """
        Trains a model, evaluates performance, and calculates permutation importance.

        Args:
            features (list[str]): List of feature names.
            model (sklearn.base.BaseEstimator): Model to be trained.
            X_train (pd.DataFrame): Training feature data.
            X_test (pd.DataFrame): Test feature data.
            y_train (pd.Series): Training target labels.
            y_test (pd.Series): Test target labels.

        Returns:
            tuple:
                - pd.DataFrame: Feature importance.
                - float: Training accuracy.
                - float: Test accuracy.
                - dict: Classification report.

        Example:
            imp_df, train_acc, test_acc, report = self._train_and_evaluate_model(...)
        """
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        class_report = classification_report(y_test, test_pred, output_dict=True)
        importance_df = self._calculate_permutation_importance(features , model, X_test,y_test)

        return importance_df, train_acc, test_acc, class_report
    
    @timer
    def _run_all_models(self, features,X_train, X_test, y_train, y_test):
        """
        Trains all models, evaluates them, and calculates feature importance.

        Args:
            features (list[str]): List of feature names.
            X_train (pd.DataFrame): Training feature data.
            X_test (pd.DataFrame): Test feature data.
            y_train (pd.Series): Training target labels.
            y_test (pd.Series): Test target labels.

        Returns:
            tuple:
                - dict[str, pd.DataFrame]: Importance results per model.
                - dict[str, dict[str, float]]: Train/test accuracy for each model.
                - dict[str, dict]: Classification reports per model.
        """
        importance_results = {}
        performance = {}
        classification_reports = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            importance_df, train_acc, test_acc, report =  self._train_and_evaluate_model(features,model, X_train, X_test, y_train, y_test) 
            importance_results[name] = importance_df
            performance[name] = {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
            classification_reports[name] = report

        return importance_results, performance, classification_reports
    
    @timer
    def _prep_binary_class(self,df,features,target_cluster):
        """
        Creates binary columns for each cluster value (one-vs-all).

        Args:
            df (pd.DataFrame): Full dataset.
            features (list[str]): List of feature names.
            target_cluster (str): Target column representing clusters.

        Returns:
            pd.DataFrame: Modified dataframe with binary columns for each cluster.
        """

        binary_df = df.copy() 
        for cluster_label in sorted(df[target_cluster].unique()):
            binary_df[f'is_cluster_{cluster_label}'] = (df[target_cluster] == cluster_label).astype(int)
        return binary_df
    
    @timer
    def _cal_imp_one_binary_class(self,df, features,target_cluster,focus_segment):
        """
        Computes permutation importance for a specific cluster (one-vs-all binary classification).

        Args:
            df (pd.DataFrame): Full dataset.
            features (list[str]): Feature names.
            target_cluster (str): Target column for clusters.
            focus_segment (Any): The cluster value to treat as positive class.

        Returns:
            pd.DataFrame: Feature importances across all models for the specified segment.
        """
        binary_df = self._prep_binary_class(df,features,target_cluster)
        y = binary_df[f'is_cluster_{focus_segment}']
        X = binary_df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        importance_results, performance, classification_reports = self._run_all_models(features,X_train, X_test, y_train, y_test)

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


    @timer
    def _prep_rank_importance(self,df,features,target_cluster,focus_segment) -> pd.DataFrame:        
        """
        Prepares a long-form dataframe of feature importance ranks and cumulative contribution.
        Ranks features based on their importance scores across different models.

        Args:
            df (pd.DataFrame): Full dataset.
            features (list[str]): Feature names.
            target_cluster (str): Target cluster column.
            focus_segment (Any): Cluster value to compute importance for.

        Returns:
            pd.DataFrame: Melted and ranked importance with cumulative contribution.
        """
        importance_df = self._cal_imp_one_binary_class(df,features, target_cluster,focus_segment)
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

            
    @timer
    def _pivot_rank_importance(self,df,features,target_cluster,focus_segment):
        """
        Aggregates top-N ranked features across models into a pivot table.

        Args:
            df (pd.DataFrame): Dataset.
            features (list[str]): Feature names.
            target_cluster (str): Cluster column.
            focus_segment (Any): Cluster value of interest.

        Returns:
            pd.DataFrame: Pivot table of feature counts at each rank across models.
        """
        melt_df = self._prep_rank_importance(df,features,target_cluster,focus_segment)
        pvt_imp = pd.pivot_table(melt_df,
                                 index='Feature',
                                 columns='rank', # rank_abs
                                 values='variable',
                                 fill_value=0,
                                 aggfunc='count')
        top_n_pvt_imp = pvt_imp.loc[:, :self.n_rank]
        return top_n_pvt_imp


    @timer
    def _cal_imp_all_binary_class(self,df,features, target_cluster):   
        """
        Computes feature importance for each cluster segment using one-vs-all binary classification.

        Args:
            df (pd.DataFrame): Dataset.
            features (list[str]): Feature names.
            target_cluster (str): Column representing cluster labels.

        Returns:
            tuple:
                - pd.DataFrame: Combined feature importances for all segments.
                - pd.DataFrame: Aggregated top-N feature ranks for all segments.
                - pd.DataFrame: Cumulative contribution of each feature.
        """
        unique_segments = sorted(df[target_cluster].unique(), reverse=False)
        all_imps = []
        all_pvt_imps_score = []
        all_cumsum = []

        for segment in unique_segments:
            print(f"Processing segment {segment}")

            # Compute feature importance for the given segment
            importance_df = self._cal_imp_one_binary_class(df,features,target_cluster,focus_segment=segment)

            # get melt for cumsum method
            melt_df = self._prep_rank_importance(df,features,target_cluster,focus_segment=segment)
            melt_df['Segment'] = segment
            all_cumsum.append(melt_df)

            # Get top N important features for imp rank method
            pvt_imp = self._pivot_rank_importance(df,features,target_cluster,focus_segment=segment)

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

        return final_imp,final_pvt_imp_score,final_cumsum
    
    @timer
    def _bin_cumsum_percentiles(self,
                                final_cumsum, 
                                bin_size: int = 20
                               ) -> pd.DataFrame:         
        """
        Bins features into percentile-based groups and counts model occurrences per bin per feature & segment.

        Args:
            final_cumsum (pd.DataFrame): DataFrame with cumulative importance values.
            bin_size (int, optional): Percentile bin size. Defaults to 20.

        Returns:
            tuple:
                - pd.DataFrame: Binned model count for each feature and segment.
                - pd.DataFrame: Voting score summary vs threshold.

        Example:
            final_pvt_bin, final_vote_score = self._bin_cumsum_percentiles(cumsum_df)
        """
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

        final_pvt_cumsum_bin =  count_df.pivot_table(index=['Feature', 'Segment'], 
                                     columns='bin', 
                                     values='count', 
                                     fill_value=0, 
                                     observed=False).reset_index()

        final_pvt_cumsum_bin['single_imp'] = np.where(final_pvt_cumsum_bin['pct_100']==len(self.models),
                                               final_pvt_cumsum_bin['pct_100'],
                                               0).astype(int)        
        # Reorder columns: Feature, Segment, cnt_models_all_one, bin columns
        ordered_cols = ['Feature']+['single_imp'] + bin_labels +['Segment']
        final_pvt_cumsum_bin = final_pvt_cumsum_bin.reindex(columns=ordered_cols)
        final_pvt_cumsum_bin[bin_labels] = final_pvt_cumsum_bin[bin_labels].astype(int)
        final_pvt_cumsum_bin = final_pvt_cumsum_bin.sort_values(by=["Segment", "Feature"], ascending=[True, False]).reset_index(drop=True)
        
        #########################################################
        
        print(f"Using threshold of {self.pct_thres*100}% to determine feature importance across models.")
        
        # Create threshold bin labels directly from the df
        df['bin_thres'] = np.where(
            df['cumsum_pct'] * 100 < self.pct_thres*100,
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

        # Add 'single_imp' column from final_pvt_cumsum_bin
        final_pvt_cumsum_score = final_pvt_cumsum_score.merge(
            final_pvt_cumsum_bin[['Feature', 'Segment', 'single_imp']],
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
                
        return final_pvt_cumsum_bin , final_pvt_cumsum_score

    @timer
    def run(self,
            df: pd.DataFrame,
            features: list[str],
            target_cluster: str):
        """
        Main method to compute and summarize feature importance across models and clusters.

        Args:
            df (pd.DataFrame): Dataset.
            features (list[str]): List of feature names to evaluate.
            target_cluster (str): Column representing clusters.

        Returns:
            dict:
                - final_imp (pd.DataFrame): Feature importances for all segments.
                - final_imp_score (pd.DataFrame): Ranked importance (pivoted format).
                - final_cumsum_bin (pd.DataFrame): Percentile bin count summary.
                - final_cumsum_score (pd.DataFrame): Score summary with threshold filtering.

        Example:
            result = model_runner.run(df=data, features=feature_list, target_cluster="cluster")
        """
        
        final_imp,final_imp_score,final_cumsum = self._cal_imp_all_binary_class(df, features, target_cluster)
        final_cumsum_bin , final_cumsum_score = self._bin_cumsum_percentiles(final_cumsum)
        
        return dict(
                    final_imp=final_imp,
                    final_imp_score=final_imp_score,
                    final_cumsum_bin=final_cumsum_bin,
                    final_cumsum_score=final_cumsum_score
                    )