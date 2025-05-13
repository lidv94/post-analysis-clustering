import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from post_analysis_clustering.utils import timer

# 1. Binning Function (equal range)
@timer
def bin_features(
    df: pd.DataFrame,
    features: list[str],
    n_bins: int = 5,
    bin_suffix: str = "_bin",
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Bin numerical features into equal-width intervals.

    If negative values are found, suggest using bin_features_neg_zero_pos instead.

    Args:
        df (pd.DataFrame): Input DataFrame containing features to bin.
        features (list of str): List of feature names to bin.
        n_bins (int, optional): Number of bins to create. Defaults to 5.
        bin_suffix (str, optional): Suffix to append to new binned feature names. Defaults to "_bin".
        drop_original (bool, optional): If True, drops the original features after binning. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with binned features added (or replaced if drop_original=True).
    """
    binned_df = df.copy()
    
    for col in features:
        try:
            # Handle edge cases: NaNs and identical values
            if binned_df[col].nunique() <= 1:
                binned_df[f"{col}{bin_suffix}"] = "SingleValue"
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
            binned_df[f"{col}{bin_suffix}"] = binned_series

        except Exception as e:
            print(f"Error processing column '{col}': {e}")
            binned_df[f"{col}{bin_suffix}"] = "Error"

    if drop_original:
        binned_df.drop(columns=features, inplace=True)

    return binned_df

@timer
def bin_features_neg_zero_pos(
    df: pd.DataFrame,
    features: list[str],
    pos_n_bins: int = 5,
    neg_n_bins: int = 5,
    bin_suffix: str = "_bin",
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Bin numerical features into separate intervals for negative, zero, and positive values.
    
    - Negative values are binned into negative bins.
    - Zero values are labeled as "= 0".
    - Positive values are binned into positive bins.
    
    This function allows for separate binning of negative and positive values, with customizable
    bin counts for both groups.

    Args:
        df (pd.DataFrame): Input DataFrame containing features to bin.
        features (list[str]): List of feature names to bin.
        pos_n_bins (int, optional): Number of bins to create for positive values. Defaults to 5.
        neg_n_bins (int, optional): Number of bins to create for negative values. Defaults to 5.
        bin_suffix (str, optional): Suffix to append to new binned feature names. Defaults to "_bin".
        drop_original (bool, optional): If True, drops the original features after binning. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with new binned features added. If drop_original is True, the original
        features are removed.
        
    Notes:
        - Columns with a single unique value will be labeled as `"SingleValue: <value>"`.
        - Zero values will be labeled as `= 0`.
        - If there are only negative or positive values, they will be binned accordingly.
    """
    binned_df = df.copy()

    for col in features:
        try:
            # Handle columns with only one unique value
            if binned_df[col].nunique() <= 1:
                unique_val = binned_df[col].dropna().unique()[0] if binned_df[col].notna().any() else "NaN"
                binned_df[f"{col}{bin_suffix}"] = f"SingleValue: {unique_val}"
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
                    negative_part[f"{col}{bin_suffix}"] = f"SingleNegativeBin: {neg_values.min()}"
                else:
                    negative_part[f"{col}{bin_suffix}"] = pd.cut(
                        neg_values,
                        bins=neg_bin_edges,
                        include_lowest=True
                    )

                binned_col.loc[negative_part.index] = negative_part[f"{col}{bin_suffix}"]

            # Handle zero values
            if not zero_part.empty:
                binned_col.loc[zero_part.index] = "= 0"

            # Handle positive values
            if not positive_part.empty:
                pos_values = positive_part[col].round(2)
                pos_bin_edges = np.round(np.linspace(pos_values.min(), pos_values.max(), pos_n_bins + 1), 2)

                if len(np.unique(pos_bin_edges)) == 1:
                    positive_part[f"{col}{bin_suffix}"] = f"SinglePositiveBin: {pos_values.min()}"
                else:
                    positive_part[f"{col}{bin_suffix}"] = pd.cut(
                        pos_values,
                        bins=pos_bin_edges,
                        include_lowest=True
                    )

                binned_col.loc[positive_part.index] = positive_part[f"{col}{bin_suffix}"]

            binned_df[f"{col}{bin_suffix}"] = binned_col

        except Exception as e:
            print(f"Error processing column '{col}': {e}")
            binned_df[f"{col}{bin_suffix}"] = "Error"

    if drop_original:
        binned_df.drop(columns=features, inplace=True)

    return binned_df



# 2. Add One-vs-Rest Binary Columns for Each Segment
@timer
def prep_binary_class(df: pd.DataFrame, features: list[str], target_cluster: str):
    """
    Prepares binary classification labels for each cluster segment by converting the target cluster into binary columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the features and target cluster column.
        features (list[str]): List of feature column names.
        target_cluster (str): The name of the target cluster column containing segment labels.

    Returns:
        pd.DataFrame: A DataFrame with new binary columns representing each cluster.
    """
    try:
        # Check if target_cluster exists in the DataFrame
        if target_cluster not in df.columns:
            raise ValueError(f"Column '{target_cluster}' not found in the DataFrame.")

        # Check if features are present in the DataFrame
        for feature in features:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found in the DataFrame.")
    
        binary_df = df.copy() 
        for cluster_label in sorted(df[target_cluster].unique()):
            binary_df[f'is_cluster_{cluster_label}'] = (df[target_cluster] == cluster_label).astype(int)
        return binary_df
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    except KeyError as ke:
        print(f"KeyError: {ke}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

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


# 4. Significance Test Function with H0/H1 Explanation
@timer
def interpret_pvalues(pval_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Interpret p-values from Chi-square tests and return a DataFrame with significance results.

    Args:
        pval_df (pd.DataFrame): DataFrame containing p-values from Chi-square tests.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        pd.DataFrame: DataFrame with the same schema as pval_df, but with interpreted significance results.
    
    Interpretation:
        - For each p-value:
            - If p < alpha, the result is "Significant (Reject H0)".
            - If p >= alpha, the result is "Not significant (Fail to reject H0)".
            - If p is NaN, the result is "Insufficient data".
    """
    result_df = pval_df.copy().astype('object') 
    print("\nChi-Square Significance Test Results (alpha =", alpha, ")")
    print("H0: Feature distribution in this segment is the same as the rest")
    print("H1: Feature distribution in this segment is different from the rest\n")

    # Iterate through each cell in the DataFrame to assign significance results
    for segment in result_df.columns:
        for feature, p in result_df[segment].items():
            if pd.isna(p):
                result_df.at[feature, segment] = "Insufficient data"
            elif p < alpha:
                result_df.at[feature, segment] = "Significant (Reject H0)"
            else:
                result_df.at[feature, segment] = "Not significant (Fail to reject H0)"

    return result_df

# 5. convert p-values into logworth scores
@timer
def calculate_logworth_scores(pval_df: pd.DataFrame, min_valid_p: float = 1e-300) -> pd.DataFrame:
    """
    Convert p-value DataFrame to LogWorth scores, which are the negative logarithm (base 10) of the p-values.
    LogWorth scores are used to rank variables based on their significance. A higher LogWorth score corresponds 
    to stronger significance (smaller p-values).

    The function ensures that p-values less than or equal to zero are replaced by `min_valid_p` to avoid issues with 
    taking the logarithm of zero or negative values.

    Args:
        pval_df (pd.DataFrame): DataFrame containing p-values for various features and segments.
        min_valid_p (float, optional): The minimum valid p-value to use when replacing any p-value <= 0. Defaults to 1e-300.

    Returns:
        pd.DataFrame: DataFrame containing the calculated LogWorth scores for each feature and segment, with the same shape as `pval_df`.
        
    LogWorth Calculation:
        - LogWorth = -log10(p-value)
        - Higher LogWorth values indicate stronger significance (smaller p-values).
        - Any p-values <= 0 are replaced with `min_valid_p` before calculation to avoid undefined log values.
    
    Example:
        If the p-value for a particular feature is 0.0001, the corresponding LogWorth score will be:
            LogWorth = -log10(0.0001) = 4.
        
    Note:
        LogWorth scores are useful for ranking features or segments by their level of significance, especially in
        areas such as bioinformatics, segmentation analysis, and marketing science.

    """
    try:
        # Ensure the input DataFrame is valid
        if not isinstance(pval_df, pd.DataFrame):
            raise TypeError("Input pval_df must be a pandas DataFrame.")
        
        # Calculate LogWorth scores
        logworth_df = -np.log10(pval_df.clip(lower=min_valid_p))  # Calculate LogWorth, clipping values to avoid log(0)
        logworth_df = logworth_df.round(4)  # Round to 4 decimal places for clarity
        
        print("\nLogWorth scores calculated. Higher values = more significant.")
        return logworth_df
        
    except TypeError as te:
        print(f"Error: {te}")
        return pd.DataFrame()  # Return an empty DataFrame on error
        
    except Exception as e:
        print(f"Unexpected error occurred in calculating LogWorth scores: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


@timer
def plot_heatmap_logworth_colscaled(logworth_df: pd.DataFrame, compare_type: str = 'Normalized'):
    """
    Plot a heatmap of LogWorth scores for multiple clusters with independent column-wise color scaling.

    This function visualizes the LogWorth scores across all clusters in a single heatmap, with each column 
    (representing a cluster) having its own color scale for better comparison of significance levels.

    Args:
        logworth_df (pd.DataFrame): 
            A DataFrame where each row represents a feature, and each column represents a cluster. 
            The values in the DataFrame are the LogWorth scores, which are the -log10 of p-values.
            Higher LogWorth values indicate stronger significance (lower p-values).
        
        compare_type (str, optional): 
            Defines how to scale the LogWorth scores across columns (clusters):
                - 'Global': No scaling, showing absolute LogWorth scores.
                - 'Percentage': Scales each column (cluster) based on the percentage of its values relative to the total sum.
                - 'Normalized': Scales each column independently between 0 and 1 (default option), 
                  normalizing values within each cluster for comparison across clusters.

    Returns:
        None: 
            The function displays a heatmap showing the LogWorth scores for each feature and cluster.

    Notes:
        - LogWorth scores are computed as -log10(p-value), where higher scores indicate stronger statistical significance.
        - The color scale for each cluster (column) is applied independently, allowing for a more accurate comparison of significance across clusters.
        - The heatmap visualizes all clusters in one plot, and a global colorbar is shown for reference.
        
    Example:
        plot_heatmap_logworth_colscaled(logworth_df, compare_type='Normalized')
    """
    try:
        # Validate inputs
        if not isinstance(logworth_df, pd.DataFrame):
            raise TypeError("logworth_df must be a pandas DataFrame.")
        
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
        
    except ValueError as ve:
        print(f"Error: {ve}")
        
    except TypeError as te:
        print(f"Error: {te}")
        
    except Exception as e:
        print(f"Unexpected error occurred while plotting the heatmap: {e}")


# 6. get lean features
@timer
def get_significant_features_by_pval(pval_df: pd.DataFrame, alpha: float = 0.05):
    """
    Extract significant features based on p-value threshold for each cluster.

    This function identifies the features that have p-values below the specified alpha for each cluster.
    It returns a dictionary of significant features per cluster, as well as a list of union features across all clusters.

    Args:
        pval_df (pd.DataFrame): DataFrame where rows represent features and columns represent clusters. 
                                 Values are p-values for each feature-cluster pair.
        alpha (float, optional): Significance level threshold. Features with p-values below this threshold are considered significant. Default is 0.05.

    Returns:
        tuple: 
            - cluster_lean_features_dict (dict): A dictionary with clusters as keys and lists of significant features as values.
            - union_lean_feature_list (list): A sorted list of all significant features across all clusters.
            - union_scaled_lean_feature_list (list): A sorted list of all significant features with 'SCALED_' prefix across all clusters.
    """
    try:
        # Validate inputs
        if not isinstance(pval_df, pd.DataFrame):
            raise TypeError("pval_df must be a pandas DataFrame.")
        
        if not isinstance(alpha, (float, int)):
            raise TypeError("alpha must be a numeric value.")
        
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1.")
            
        cluster_lean_features_dict = {}
        union_lean_feature_set = set()
        print(f"Total lean features from prior process: {len(pval_df.index)}")
        print(f"------Result after perform chi-square test------")

        for cluster in pval_df.columns:
            # Select significant features where p-value < alpha
            sig_features = pval_df[cluster][pval_df[cluster] < alpha].dropna().index.tolist()
            sig_features = sorted(sig_features)

            scaled_lean_feature_list = [f"SCALED_{f}" for f in sig_features]
            union_lean_feature_set.update(sig_features)

            cluster_lean_features_dict[cluster] = {
                "lean_feature_list": sig_features,
                "scaled_lean_feature_list": scaled_lean_feature_list
            }

            print(f"Cluster {cluster}:")
            print(f"  Total significant features: {len(sig_features)}")

        union_lean_feature_list = sorted(list(union_lean_feature_set))
        union_scaled_lean_feature_list = [f"SCALED_{f}" for f in union_lean_feature_list]

        print(f"\nUnion across all clusters:")
        print(f"  Total union features: {len(union_lean_feature_list)}")

        return cluster_lean_features_dict, union_lean_feature_list, union_scaled_lean_feature_list

    except ValueError as ve:
        print(f"Error: {ve}")
        return {}, [], []
        
    except TypeError as te:
        print(f"Error: {te}")
        return {}, [], []
        
    except Exception as e:
        print(f"Unexpected error occurred while extracting significant features by p-value: {e}")
        return {}, [], []
    
    
@timer
def get_significant_features_by_logworth(logworth_df: pd.DataFrame, thres_logworth: float = 1.301):
    """
    Extract significant features based on LogWorth threshold for each cluster.

    This function identifies features with LogWorth scores greater than the specified threshold for each cluster.
    LogWorth is computed as -log10(p-value), so higher values indicate stronger significance.

    Args:
        logworth_df (pd.DataFrame): DataFrame where rows represent features and columns represent clusters.
                                    Values are LogWorth scores for each feature-cluster pair.
        thres_logworth (float, optional): LogWorth threshold. Features with LogWorth values above this threshold are considered significant. Default is 1.301 (corresponding to p-value = 0.05).

    Returns:
        tuple:
            - cluster_lean_features_dict (dict): A dictionary with clusters as keys and lists of significant features as values.
            - union_lean_feature_list (list): A sorted list of all significant features across all clusters.
            - union_scaled_lean_feature_list (list): A sorted list of all significant features with 'SCALED_' prefix across all clusters.
    """
    try:
        # Validate inputs
        if not isinstance(logworth_df, pd.DataFrame):
            raise TypeError("logworth_df must be a pandas DataFrame.")
        
        if not isinstance(thres_logworth, (float, int)):
            raise TypeError("thres_logworth must be a numeric value.")
        
        if thres_logworth <= 0:
            raise ValueError("thres_logworth must be greater than 0.")
            
        cluster_lean_features_dict = {}
        union_lean_feature_set = set()
        print(f"Total lean features from prior process: {len(logworth_df.index)}")
        print(f"------Result after perform chi-square test------")

        for cluster in logworth_df.columns:
            sig_features = logworth_df[cluster][logworth_df[cluster] > thres_logworth].dropna().index.tolist()
            sig_features = sorted(sig_features)

            scaled_lean_feature_list = [f"SCALED_{f}" for f in sig_features]
            union_lean_feature_set.update(sig_features)

            cluster_lean_features_dict[cluster] = {
                "lean_feature_list": sig_features,
                "scaled_lean_feature_list": scaled_lean_feature_list
            }

            print(f"Cluster {cluster}:")
            print(f"  Total significant features: {len(sig_features)}")

        union_lean_feature_list = sorted(list(union_lean_feature_set))
        union_scaled_lean_feature_list = [f"SCALED_{f}" for f in union_lean_feature_list]

        print(f"\nUnion across all clusters:")
        print(f"  Total union features: {len(union_lean_feature_list)}")

        return cluster_lean_features_dict, union_lean_feature_list, union_scaled_lean_feature_list
    
    except ValueError as ve:
        print(f"Error: {ve}")
        return {}, [], []
        
    except TypeError as te:
        print(f"Error: {te}")
        return {}, [], []
        
    except Exception as e:
        print(f"Unexpected error occurred while extracting significant features by LogWorth: {e}")
        return {}, [], []