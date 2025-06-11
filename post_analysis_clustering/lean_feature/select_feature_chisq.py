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
                 thres_logworth: float = 1.301
                ):
        """
        Initialize LeanChiSquare instance.

        Args:
            df (pd.DataFrame): Input DataFrame containing the features and cluster labels.
            features (list[str]): List of feature column names to evaluate.
            target_cluster (str): Column name representing the cluster labels.
            thres_logworth (float, optional): Threshold of -log10(p-value) to determine significance. Defaults to 1.301 (~p < 0.05).
        """
        
        self.df = df
        self.features = features
        self.target_cluster = target_cluster
        self.thres_logworth = thres_logworth
        
        self._validate_chisq_attribute()
        
        # Call BaseLean init
        super().__init__(df, 
                         features, 
                         target_cluster)
        
    def _validate_chisq_attribute(self):
        """
        Validate the logworth threshold value.

        Raises:
            TypeError: If thres_logworth is not a float or int.
            ValueError: If thres_logworth is less than or equal to 0.
        """
        if not isinstance(self.thres_logworth, (float, int)):
            raise TypeError("thres_logworth must be a numeric value.")
        
        if self.thres_logworth <= 0:
            raise ValueError("thres_logworth must be greater than 0.")
            
    @timer    
    def PrepData(self,
                     method: str = "equal_range",  # or "neg_zero_pos"
                     n_bins: int = 5,
                     neg_n_bins: int = 5,
                     pos_n_bins: int = 5,
                     drop_original: bool = True
                     ) -> pd.DataFrame:
        """
        Bin numerical features using the specified method and prepare binary cluster indicators.

        Args:
            method (str, optional): Binning method: 'equal_range' or 'neg_zero_pos'. Defaults to 'equal_range'.
            n_bins (int, optional): Number of bins for equal_range method. Defaults to 5.
            neg_n_bins (int, optional): Number of bins for negative values in neg_zero_pos method. Defaults to 5.
            pos_n_bins (int, optional): Number of bins for positive values in neg_zero_pos method. Defaults to 5.
            drop_original (bool, optional): Whether to drop the original feature columns. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with binned features and binary cluster indicator columns.
        """
        binned_df = self.df.copy()
        
        # prep binary class
        for cluster_label in sorted(binned_df[self.target_cluster].unique()):
            binned_df[f'is_cluster_{cluster_label}'] = (binned_df[self.target_cluster] == cluster_label).astype(int)
        
        # prep bin
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
    
    @timer
    def TestChiSquare(self,
                     method: str = "equal_range",  # or "neg_zero_pos"
                     n_bins: int = 5,
                     neg_n_bins: int = 5,
                     pos_n_bins: int = 5,
                     drop_original: bool = True,
                     min_valid_p: float = 1e-300
                     ):
        """
        Perform Chi-square tests between binned features and binary cluster segments.

        Args:
            method (str, optional): Binning method: 'equal_range' or 'neg_zero_pos'. Defaults to 'equal_range'.
            n_bins (int, optional): Number of bins for equal_range method. Defaults to 5.
            neg_n_bins (int, optional): Number of bins for negative values in neg_zero_pos method. Defaults to 5.
            pos_n_bins (int, optional): Number of bins for positive values in neg_zero_pos method. Defaults to 5.
            drop_original (bool, optional): Whether to drop the original feature columns. Defaults to True.
            min_valid_p (float, optional): Minimum valid p-value to avoid log(0) during logworth computation. Defaults to 1e-300.

        Returns:
            tuple:
                - pd.DataFrame: DataFrame of raw p-values (features × clusters).
                - pd.DataFrame: DataFrame of -log10(p-values), also known as logworth scores.
        """
        # Step 1 Preprocess data to create binary columns for cluster segments
        binned_df = self.PrepData(method=method,
                                  n_bins=n_bins,
                                  neg_n_bins=neg_n_bins,
                                  pos_n_bins=pos_n_bins,
                                  drop_original=drop_original)
                                      
        result_dict = {}

        # Step 2 Identify binary columns that represent the cluster segments
        binary_columns = [col for col in binned_df.columns if col.startswith("is_cluster_")]
        # Step 3: Identify binned feature columns
        binned_features = [col for col in binned_df.columns if col.endswith("_bin")]

        print(f'List of binary class columns : {binary_columns}')
        print(f'List of binned features : {binned_features}')
        
        # Step 4: Loop over each binary segment column and test against binned features
        for bin_col in binary_columns:
            segment_label = bin_col.replace("is_cluster_", '')
            p_values = {}

            for feature in binned_features:
                try:
                    contingency = pd.crosstab(binned_df[feature], binned_df[bin_col])
                    if contingency.shape[0] > 1 and contingency.shape[1] == 2:
                        _, p, _, _ = chi2_contingency(contingency)
                    else:
                        p = np.nan
                except Exception as e:
                    warnings.warn(f"Chi-square failed for feature '{feature}' and segment '{segment_label}': {e}")
                    p = np.nan # If insufficient data, set p-value to NaN

                p_values[feature] = p
                
            # Step 5: Store p-values for each segment in the result dictionary
            result_dict[segment_label] = p_values
            
        pval_df= pd.DataFrame(result_dict)
        
        # Step 6 : Convert p-values into logworth scores
        logworth_df = -np.log10(pval_df.clip(lower=min_valid_p))  # clipping values to avoid log(0)
        logworth_df = logworth_df.round(4)
        
        print("\nLogWorth scores calculated. Higher values = more significant.")
        
        self.logworth_df =logworth_df
        self.pval_df = pval_df
        
        return self.pval_df , self.logworth_df
    
    @timer
    def GetLeanFeature(self,
                        logworth_df: pd.DataFrame=None
                       ):
        
        """
        Extract significant features per cluster based on logworth threshold.

        Args:
            logworth_df (pd.DataFrame, optional): Logworth score DataFrame. If None, internal Chi-square test will be performed. Defaults to None.

        Returns:
            tuple:
                - dict[str, list[str]]: Dictionary mapping each cluster to its list of significant features.
                - list[str]: Union of all significant features across clusters.
        """

        if logworth_df is None:
            if not hasattr(self, "logworth_df")  or self.logworth_df is None:
                print("No logworth_df provided. Testing Chi-Square for all segments...")
                self.TestChiSquare()
            logworth_df = self.logworth_df

        cluster_lean_features_dict = {}
        union_lean_feature_set = set()

        print(f'Threshold of logworth > {self.thres_logworth}')

        for cluster in logworth_df.columns:
            sig_features = logworth_df[cluster][logworth_df[cluster] > self.thres_logworth].dropna().index.tolist()
            sig_features = sorted(sig_features)
            union_lean_feature_set.update(sig_features)

            cluster_lean_features_dict[cluster] = sig_features

            print(f"Cluster {cluster}:")
            print(f"  Total features from raw: {len(logworth_df.index)}")
            print(f"  Total features remaining after threshold filter: {len(sig_features)}")

        union_lean_feature_list = sorted(list(union_lean_feature_set))
        print(f"\nUnion across all clusters:")
        print(f"  Total union features: {len(union_lean_feature_list)}")

        return cluster_lean_features_dict, union_lean_feature_list

        
    @timer
    def PlotHeatmapLogworth(self, 
                      logworth_df: pd.DataFrame = None, 
                      compare_type: str = 'Normalized'
                      ):
        """
        Plot a heatmap of logworth scores for each feature-cluster pair.

        Args:
            logworth_df (pd.DataFrame, optional): Logworth score DataFrame. If None, internal Chi-square test will be performed. Defaults to None.
            compare_type (str, optional): Comparison method for visualization. One of 'Global', 'Percentage', or 'Normalized'. Defaults to 'Normalized'.

        Raises:
            ValueError: If compare_type is not one of the allowed options.

        Returns:
            None
        """

        if logworth_df is None:
            if not hasattr(self, "logworth_df")  or self.logworth_df is None:
                print("No logworth_df provided. Testing Chi-Square for all segments...")
                self.TestChiSquare()
            logworth_df = self.logworth_df

        if compare_type not in ['Global', 'Percentage', 'Normalized']:
            raise ValueError("compare_type must be one of 'Global', 'Percentage', or 'Normalized'.")

        # Define the number of clusters and features
        clusters = logworth_df.columns
        features = logworth_df.index

        # Compute data for selected compare_type
        if compare_type == 'Global':
            show_data = logworth_df
        elif compare_type == 'Percentage':
            show_data = logworth_df.div(logworth_df.sum(axis=0), axis=1) * 100
        else:  # Default: 'Normalized'
            show_data = (logworth_df - logworth_df.min()) / (logworth_df.max() - logworth_df.min())

        # Plot the heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            show_data,
            cmap='Blues',
            annot=logworth_df,
            fmt=".2f",
            cbar=True,
            linewidths=0.4,
            linecolor='gray',
            mask=logworth_df.isna()
        )

        # Labels and title
        plt.title(f"LogWorth Heatmap ({compare_type} Column-scaled)", fontsize=14)
        plt.xlabel("Clusters")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
