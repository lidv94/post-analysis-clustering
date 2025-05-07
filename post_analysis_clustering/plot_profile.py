import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from post_analysis_clustering.utils import timer

@timer
def prep_dist(raw_df: pd.DataFrame, features: list, primary_key: str, target_cluster: str):
    """
    Prepares data for cluster-wise analysis by splitting into overall and segment-specific DataFrames.

    Args:
        raw_df (pd.DataFrame): The raw input DataFrame containing all features and cluster assignments.
        features (list of str): List of feature column names to keep.
        primary_key (str): Name of the primary key column (string).
        target_cluster (str): Name of the cluster assignment column (string).

    Returns:
        tuple:
            all_df (pd.DataFrame): DataFrame containing all features across all clusters (without the cluster column).
            segment_dfs (dict of {int or str: pd.DataFrame}): Dictionary mapping each unique cluster label to its corresponding DataFrame.

    Example:
        >>> all_df, segment_dfs = prep_dist(raw_df=rich_raw_df, features=['colA', 'colB'], primary_key='party_id', target_cluster='cluster')
    """
    
    try:
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("`raw_df` must be a pandas DataFrame.")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("`features` must be a list of strings.")
        if not isinstance(primary_key, str) or primary_key not in raw_df.columns:
            raise ValueError("`primary_key` must be a string and exist in `raw_df` columns.")
        if not isinstance(target_cluster, str) or target_cluster not in raw_df.columns:
            raise ValueError("`target_cluster` must be a string and exist in `raw_df` columns.")

        prep_df = raw_df.loc[:, [primary_key] + features + [target_cluster]]
        unique_segments = sorted(prep_df[target_cluster].unique(), reverse=False) 
        segment_dfs = {} 
        for segment in unique_segments:
            print(f"Processing segment {segment}")
            segment_df = prep_df.loc[prep_df[target_cluster] == segment, :].drop(columns=target_cluster).copy()
            segment_dfs[segment] = segment_df

        all_df = prep_df.drop(columns=target_cluster)
        return all_df, segment_dfs
    except Exception as e:
        print(f"Error in prep_dist : {e}")
        return None, {}

@timer
def prep_frequency_feature(data: pd.DataFrame, col: str, binning_keywords: list = None, n_bins: int = 100):
    """
    Prepare a frequency table for a single feature, optionally binning based on keywords.

    Args:
        data (pd.DataFrame): Input dataframe.
        col (str): Name of the column to analyze.
        binning_keywords (list, optional): List of substrings to determine if binning is needed. Defaults to None.
        n_bins (int, optional): Number of bins for numeric binning. Defaults to 100.

    Returns:
        pd.DataFrame: Frequency table with columns [feature, count, percentage(%)].

    Example:
        >>> prep_frequency_feature(df, 'BENE_PREMIUM', binning_keywords=['BENE'], n_bins=10)
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas DataFrame.")
        if not isinstance(col, str) or col not in data.columns:
            raise ValueError(f"`col` must be a string and exist in DataFrame columns.")
        if binning_keywords is not None and (not isinstance(binning_keywords, list) or not all(isinstance(b, str) for b in binning_keywords)):
            raise TypeError("`binning_keywords` must be a list of strings or None.")
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("`n_bins` must be a positive integer.")
            
        proxy = data.copy()
        is_in_target_list = any(substring.lower() in col.lower() for substring in (binning_keywords or []))

        if is_in_target_list:
            try:
                zero_mask = proxy[col] == 0
                zero_part = proxy.loc[zero_mask].copy()
                nonzero_part = proxy.loc[~zero_mask].copy()
                nonzero_values = nonzero_part[col].round(2)
                bin_edges = np.round(np.linspace(nonzero_values.min(), nonzero_values.max(), n_bins + 1), 2)
                nonzero_part[col] = pd.cut(nonzero_values, bins=bin_edges, include_lowest=True)
                zero_part[col] = '= 0'
                proxy = pd.concat([zero_part, nonzero_part])

            except Exception as e:
                print(f"Could not bin {col}: {e}")
                return pd.DataFrame(columns=[col, 'count', 'percentage(%)'])

        # Count and percentage
        proxy = proxy[col].value_counts(dropna=False).reset_index()
        proxy.columns = [col, 'count']
        proxy['percentage(%)'] = round((proxy['count'] / proxy['count'].sum()) * 100, 2)

        # Sort bins
        if is_in_target_list:
            def sort_key(val):
                if val == '= 0':
                    return float('-inf')
                if isinstance(val, pd.Interval):
                    return val.left
                return float('inf')

            proxy = proxy.sort_values(by=col, key=lambda x: x.map(sort_key))
            proxy[col] = proxy[col].apply(lambda x: str(x) if x == '= 0' else f'({x.left:.2f}, {x.right:.2f}]')

        else:
            proxy = proxy.sort_values(by=col)

        return proxy.reset_index(drop=True)
    
    except Exception as e:
        print(f"Error in prep_frequency_feature : {e}")
        return pd.DataFrame(columns=[col, 'count', 'percentage(%)'])

@timer
def plot_feature_distributions(
    df_dict: dict, 
    features: list, 
    primary_key: str = 'PARTY_RK', 
    binning_keywords: list = None, 
    n_bins: int = 100
    ):
    """
    Plot feature distributions.

    Args:
        df_dict (dict): Dictionary with segment names as keys and DataFrames as values.
        features (list): List of feature column names to plot.
        primary_key (str, optional): Primary key column name. Defaults to 'PARTY_RK'.
        binning_keywords (list, optional): List of substrings indicating which features need binning. Defaults to None.
        n_bins (int, optional): Number of bins to use for binning. Defaults to 100.

    Returns:
        None

    Example:
        >>> plot_feature_distributions(
        >>>     df_dict=df_dict_all,
        >>>     features=feature_list_rich_raw,
        >>>     primary_key='PARTY_RK',
        >>>     binning_keywords=['BENE', 'PREMIUM'],
        >>>     n_bins=10
        >>> )
    """
    try:
        if not isinstance(df_dict, dict) or not all(isinstance(k, (str, int)) and isinstance(v, pd.DataFrame) for k, v in df_dict.items()):
            raise TypeError("`df_dict` must be a dictionary of {segment: DataFrame}.")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("`features` must be a list of strings.")
        if not isinstance(primary_key, str):
            raise TypeError("`primary_key` must be a string.")
        if binning_keywords is not None and (not isinstance(binning_keywords, list) or not all(isinstance(b, str) for b in binning_keywords)):
            raise TypeError("`binning_keywords` must be a list of strings or None.")
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("`n_bins` must be a positive integer.")

        n = len(features)
        cols = 2
        rows = (n + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = axes.flatten()

        last_key = list(df_dict.keys())[-1]

        for key, data in df_dict.items():
            for i, col in enumerate(features):
                ax = axes[i]

                proxy = prep_frequency_feature(data, col, binning_keywords=binning_keywords, n_bins=n_bins)
                if proxy.empty:
                    continue

                sns.barplot(
                    x=col, 
                    y='count', 
                    data=proxy,
                    ax=ax, 
                    color='#1f77b4',
                )

                ax.set_title(f"{col} ({key})", fontsize=10)
                ax.set_ylabel('Count')
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=90)

        # Remove unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'{last_key} : Distribution of features', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    except Exception as e:
        print(f"Error in plot_feature_distributions: {e}")
    
'''
Example Usage:

all_df, segment_dfs = prep_dist(raw_df=rich_raw_df,features=feature_list_rich_raw,primary_key='PARTY_RK',target_cluster='K3')
df_dict_all = {'all segment': all_df}
df_dict_0 = {'segment 0' : segment_dfs[0]}
df_dict_1 =  {'segment 1' : segment_dfs[1]}
df_dict_2 =  {'segment 2' : segment_dfs[2]}

plot_feature_distributions(
    df_dict=df_dict_all,
    features=feature_list_rich_raw,
    primary_key='PARTY_RK',
    binning_keywords=['BENE','PREMIUM'],
    n_bins=10
)
plot_feature_distributions(
    df_dict=df_dict_0,
    features=feature_list_rich_raw,
    primary_key='PARTY_RK',
    binning_keywords= ['BENE','PREMIUM'],
    n_bins=10
)
plot_feature_distributions(
    df_dict=df_dict_1,
    features=feature_list_rich_raw,
    primary_key='PARTY_RK',
    binning_keywords= ['BENE','PREMIUM'], 
    n_bins=10
)
plot_feature_distributions(
    df_dict=df_dict_2,
    features=feature_list_rich_raw,
    primary_key='PARTY_RK',
    binning_keywords= ['BENE','PREMIUM'],
    n_bins=10
)

'''