import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from post_analysis_clustering.utils import timer, get_palette

@timer
def plot_pie_cluster(df_in: pd.DataFrame ,target_cluster:str):
    """
    Plots a pie chart showing the proportion of each cluster in the target column.

    Args:
        df_in (pd.DataFrame): DataFrame containing the target cluster column.
        target_cluster (str): Column name representing the clusters.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If target_cluster is not found in the DataFrame.
    """
    try:
        if not isinstance(df_in, pd.DataFrame):
            raise TypeError("df_in must be a pandas DataFrame.")
        if not isinstance(target_cluster, str):
            raise TypeError("target_cluster must be a string.")
        if target_cluster not in df_in.columns:
            raise ValueError(f"'{target_cluster}' not found in dataframe columns.")    
    
        prod_counts = df_in[target_cluster].value_counts()
        custom_colors = get_palette(target_cluster, df_in)
        colors = [custom_colors[label] for label in prod_counts.index]

        plt.figure(figsize=(5, 6))
        plt.pie(
            prod_counts, 
            labels=[f'Cluster {label}\n n={n} ({n / prod_counts.sum() * 100:.1f}%)' for label, n in zip(prod_counts.index, prod_counts.values)],
            autopct='',  
            colors=colors
        )
        plt.title(f'Cluster Size for {target_cluster}, N = {prod_counts.sum()}')
        plt.rc('font', size=8)
        plt.show()
    except Exception as e:
        print(f"Error in plot_pie_cluster : {e}")    

@timer
def get_descriptive_stats(raw_df: pd.DataFrame, features: list[str], target_cluster: str, filter_col_keywords: list[str] = None):
    """
    Compute and print descriptive statistics for multiple features grouped by target_cluster.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): A list of feature names to compute descriptive statistics for.
        target_cluster (str): The name of the column representing the clusters.
        filter_col_keywords (list[str], optional): A list of keywords used to filter feature names (case-insensitive). Defaults to None.

    Returns:
        None: Prints descriptive statistics for each feature in the filtered list.

    Example:
        get_descriptive_stats(raw_df, features, 'cluster_col', filter_col_keywords=['keyword1', 'keyword2'])
    """
    
    try:
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("raw_df must be a pandas DataFrame.")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("features must be a list of strings.")
        if not isinstance(target_cluster, str):
            raise TypeError("target_cluster must be a string.")
        if filter_col_keywords is not None and not all(isinstance(k, str) for k in filter_col_keywords):
            raise TypeError("filter_col_keywords must be a list of strings or None.")

        # Filter features case-insensitively based on filter_col_keywords
        filtered_features = features if filter_col_keywords is None else [
            col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)
        ]
        for feature in filtered_features:
            desc_stats = raw_df.groupby(target_cluster)[feature].describe(percentiles=[.25, .5, .75])
            desc_stats = desc_stats.round(2)
            desc_stats = desc_stats.T
            print(f"\n📊 Descriptive Statistics for {feature}:\n", desc_stats)
    except Exception as e:
        print(f"Error in get_descriptive_stats : {e}")
        
@timer
def plot_violin(raw_df: pd.DataFrame, features: list[str], target_cluster: str, filter_col_keywords: list[str] = None):
    """
    Plots violin plots for each feature in the provided list.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): A list of feature names to plot.
        target_cluster (str): The name of the column representing the clusters.
        filter_col_keywords (list[str], optional): A list of keywords used to filter feature names (case-insensitive). Defaults to None.

    Returns:
        None: Displays violin plots for each filtered feature.

    Example:
        plot_violin(raw_df, features, 'cluster_col', filter_col_keywords=['keyword1'])
    """
    try:
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("raw_df must be a pandas DataFrame.")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("features must be a list of strings.")
        if not isinstance(target_cluster, str):
            raise TypeError("target_cluster must be a string.")
        if filter_col_keywords is not None and not all(isinstance(k, str) for k in filter_col_keywords):
            raise TypeError("filter_col_keywords must be a list of strings or None.")    
        custom_palette = get_palette(target_cluster, raw_df)

        if filter_col_keywords is None:
            filtered_features = features
        else:
            filtered_features = [
                col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)
            ]

        num_features = len(filtered_features)
        num_cols = 2
        num_rows = (num_features + 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for idx, feature in enumerate(filtered_features):
            sns.violinplot(
                x=raw_df[target_cluster], 
                y=raw_df[feature], 
                ax=axes[idx], 
                hue=raw_df[target_cluster], 
                palette=custom_palette,
                dodge=False
            )
            axes[idx].set_title(f'Violin Plot: {target_cluster} vs {feature}')
            axes[idx].set_xlabel(target_cluster)
            axes[idx].set_ylabel(feature)
            axes[idx].legend([], frameon=False)  # Remove legend (redundant hue)

        for j in range(idx + 1, num_rows * num_cols):
            fig.delaxes(axes[j])  # Remove extra empty subplots

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_violin : {e}")

@timer
def plot_violin_plotly(raw_df: pd.DataFrame, 
                       features: list[str], 
                       target_cluster: str, 
                       filter_col_keywords: list[str] = None):
    """
    Plots interactive violin plots using Plotly.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): A list of feature names to plot.
        target_cluster (str): The name of the column representing the clusters.
        filter_col_keywords (list[str], optional): A list of keywords used to filter feature names (case-insensitive). Defaults to None.

    Returns:
        None: Displays interactive violin plots for each filtered feature.

    Example:
        plot_violin_plotly(raw_df = rich_raw_df, features = union_lean_feature_list,target_cluster='K3', filter_col_keywords = ['PREMIUM'])
    """
    custom_palette = get_palette(target_cluster, raw_df)
    sorted_categories = sorted(raw_df[target_cluster].unique())

    # Filter features case-insensitively based on filter_col_keywords
    filtered_features = features if filter_col_keywords is None else [
        col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)
    ]

    # Loop through features and generate violin plots
    for feature in filtered_features:
        fig = px.violin(
            raw_df, 
            x=target_cluster, 
            y=feature, 
            box=True,  
            title=f"Violin plot: {target_cluster} vs {feature}",
            color=target_cluster,  
            color_discrete_map=custom_palette,
            category_orders={target_cluster: sorted_categories}
        )
        fig.update_traces(width=0.8, scalemode="count")  # Adjust width & scaling
        fig.show()


@timer
def plot_box_plotly(raw_df: pd.DataFrame, features: list[str], target_cluster: str, filter_col_keywords: list[str] = None):
    """
    Plots interactive box plots using Plotly.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): A list of feature names to plot.
        target_cluster (str): The name of the column representing the clusters.
        filter_col_keywords (list[str], optional): A list of keywords used to filter feature names (case-insensitive). Defaults to None.

    Returns:
        None: Displays interactive box plots for each filtered feature.

    Example:
        plot_box_plotly(raw_df = rich_raw_df, features = union_lean_feature_list,target_cluster='K3', filter_col_keywords = ['PREMIUM'])
    """
    custom_palette = get_palette(target_cluster, raw_df)  # Get color mapping
    sorted_categories = sorted(raw_df[target_cluster].unique())
    
    # Filter features case-insensitively based on filter_col_keywords
    filtered_features = features if filter_col_keywords is None else [
        col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)
    ]
    
    for feature in filtered_features:
        fig = px.box(
            raw_df, 
            x=target_cluster, 
            y=feature, 
            color=target_cluster, 
            color_discrete_map=custom_palette,
            title=f'Box Plot: {target_cluster} vs {feature}',
            category_orders={target_cluster: sorted_categories}  # Sort legend categories
        )
        fig.show()
        

        
@timer
def plot_crosstab(raw_df: pd.DataFrame,
                  features: list[str], 
                  target_cluster: str, 
                  filter_col_keywords: list[str] = None, 
                  compare_type: str = 'Global', 
                  annot_type: str = 'Actual'
                 ):
    """
    Plots a heatmap for the cross-tabulation of features against clusters.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): A list of feature names to include in the crosstab.
        target_cluster (str): The name of the column representing the clusters.
        filter_col_keywords (list[str], optional): A list of keywords used to filter feature names (case-insensitive). Defaults to None.
        compare_type (str, optional): Specifies how values are scaled for coloring in the heatmap. 
            Options are 'Global', 'Normalized', or 'Percentage'. Defaults to 'Global'.
        annot_type (str, optional): Specifies the annotation type. Options are 'Actual' or 'Percentage'. Defaults to 'Actual'.

    Returns:
        None: Displays the heatmap.

    Example:
        plot_crosstab(raw_df, features, 'cluster_col', filter_col_keywords=['keyword1'], compare_type='Percentage', annot_type='Actual')
    """
    # If filter_col_keywords is None, use all features
    if filter_col_keywords is None:
        filtered_features = features
    else:
        filtered_features = [
            col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)
        ]
    num_features = len(filtered_features)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for idx, col in enumerate(filtered_features):
        crosstab_result = pd.crosstab(raw_df[col], raw_df[target_cluster]).sort_index(ascending=False) # Sort the rows (y-axis) in descending order
        percent_freq = crosstab_result.div(crosstab_result.sum(axis=0), axis=1) # * 100
        
        # Compute data for selected compare_type
        if compare_type == 'Normalized':
            show_data =  (crosstab_result - crosstab_result.min()) / (crosstab_result.max() - crosstab_result.min())
        elif compare_type == 'Percentage':
            show_data = percent_freq
        else:  # Default: 'Global'
            show_data = crosstab_result
            
        ############################################
        
        if annot_type == 'Percentage':
            custom_annot = percent_freq
        else: # default : 'Actual'
            custom_annot = crosstab_result
        
        
        sns.heatmap(show_data, 
                    annot = custom_annot, 
                    fmt= '.2%' ,#,'d',
                    cmap='Blues', 
                    cbar=True,
                    linewidths=0.5,
                    ax=axes[idx]
                   )
        axes[idx].set_title(f"{compare_type} Heatmap for {col}", fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(f"{col} Value", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(f"Cluster - {target_cluster}", fontsize=11, fontweight='bold')

    for j in range(idx + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
####################################################################################################
    
@timer
def prep_bin_heatmap(raw_df: pd.DataFrame, column: str, target_cluster: str) -> pd.DataFrame:
    """
    Prepare a binned heatmap of a continuous feature column based on quantiles.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        column (str): The name of the continuous feature to bin.
        target_cluster (str): The name of the column representing the clusters.

    Returns:
        pd.DataFrame: A pivot table with binned values in rows and clusters in columns.

    Example:
        prep_bin_heatmap(raw_df, 'feature_column', 'cluster_col')
    """
    df = raw_df.copy()

    # Define quantiles for binning
    quantiles = np.arange(0, 1.1, 0.2)
    non_zero_mask = df[column] != 0
    quantile_values = df.loc[non_zero_mask, column].quantile(quantiles).unique()

    # Fallback if quantiles are not distinct
    if len(quantile_values) < 2:
        quantile_values = np.linspace(df[column].min(), df[column].max(), num=3)

    # Bin the values into quantiles
    df['bin'] = pd.cut(df[column], bins=quantile_values, include_lowest=True)

    # Handle zero values separately
    df['bin'] = df['bin'].astype('category')
    df['bin'] = df['bin'].cat.add_categories(['0'])
    df.loc[df[column] == 0, 'bin'] = '0'

    # Sort bins: Zero first, then numeric bins in order
    ordered_bins = ['0'] + sorted([cat for cat in df['bin'].cat.categories if cat != '0'],
                                     key=lambda x: x.left if isinstance(x, pd.Interval) else float('-inf'))

    df['bin'] = df['bin'].cat.reorder_categories(ordered_bins, ordered=True)

    return df.pivot_table(columns=target_cluster, index='bin', aggfunc='size', fill_value=0,observed=False)


@timer
def plot_bin_heatmap(raw_df: pd.DataFrame, 
                     features: list[str], 
                     target_cluster: str, 
                     filter_col_keywords: list[str] = None, 
                     annot_type: str = 'Percentage'):
    """
    Plots binned heatmaps for continuous features based on quantiles, with optional filtering based on keywords.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): A list of continuous feature names to plot.
        target_cluster (str): The name of the column representing the clusters.
        filter_col_keywords (list[str], optional): A list of keywords to filter feature names (case-insensitive). Defaults to None.
        annot_type (str, optional): The type of annotation to display. Options are 'Actual' or 'Percentage'. Defaults to 'Percentage'.

    Returns:
        None: Displays a grid of binned heatmaps for each filtered feature.

    Example:
        plot_bin_heatmap(raw_df, features, 'cluster_col', filter_col_keywords=['keyword1'], annot_type='Actual')
    """
    # Filter features case-insensitively based on filter_col_keywords
    if filter_col_keywords is not None:
        filtered_features = [col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)]
    else:
        filtered_features = features
    
    num_features = len(filtered_features)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for idx, col in enumerate(filtered_features):
        df_bin = prep_bin_heatmap(raw_df=raw_df, column=col, target_cluster=target_cluster)
        df_bin_actual = df_bin.loc[df_bin.index[::-1]]  # reverse the order of the rows (bins)
        df_bin_percent = df_bin.divide(df_bin.sum(axis=0), axis=1)
        df_bin_percent = df_bin_percent.loc[df_bin.index[::-1]]  # reverse rows again

        # Select annotation type
        if annot_type == 'Actual':
            custom_annot = df_bin_actual
            custom_fmt = 'd'
        else:  # Default is 'Percentage'
            custom_annot = df_bin_percent
            custom_fmt = '.2%'

        # Plot heatmap for the feature
        sns.heatmap(df_bin_percent, 
                    annot=custom_annot, 
                    cmap='Blues', 
                    fmt=custom_fmt, 
                    linewidths=0.5, 
                    linecolor='white', 
                    ax=axes[idx])
        axes[idx].set_title(f'Percentage Heatmap for {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(f'Cluster - {target_cluster}', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Quantile Binned Value', fontsize=11, fontweight='bold')

    for j in range(idx + 1, num_rows * num_cols):
        fig.delaxes(axes[j])  # Remove extra empty subplots

    plt.tight_layout()
    plt.show()

    
    
####################################################################################################
    
@timer
def plot_snake_scaled(raw_df: pd.DataFrame, features: list[str], target_cluster: str, primary_key: str = 'PARTY_RK', filter_col_keywords: list[str] = None):
    """
    Plots a snake plot to visualize the standardized trend of features, segmented by clusters.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): List of continuous feature names to plot.
        target_cluster (str): The name of the cluster column.
        primary_key (str, optional): The name of the primary key column. Defaults to 'PARTY_RK'.
        filter_col_keywords (list[str], optional): A list of keywords to filter feature names (case-insensitive). Defaults to None.

    Returns:
        None: Displays the snake plot.

    Example:
        plot_snake_scaled(raw_df, features, 'cluster_col', filter_col_keywords=['keyword1'])
    """
    custom_palette = get_palette(target_cluster, raw_df)

    # Filter features based on filter_col_keywords, case-insensitively
    if filter_col_keywords is None:
        filtered_features = features
    else:
        filtered_features = [col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)]

    X = raw_df[filtered_features]
    y = raw_df[target_cluster]  

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Merge with primary key and target cluster
    tmp = pd.concat([raw_df[[primary_key, target_cluster]], X_scaled_df], axis=1)
    df_melted = pd.melt(tmp, 
                        id_vars=[primary_key, target_cluster],
                        value_vars=filtered_features,
                        var_name='Attribute',
                        value_name='Value')
    
    # Plot the snake plot
    plt.figure(figsize=(8, 4))
    sns.lineplot(x='Attribute', y='Value', hue=target_cluster, data=df_melted, marker='o', palette=custom_palette)
    
    plt.axhline(y=0, linestyle='--', color='gray', linewidth=1)
    plt.xticks(rotation=90)
    plt.title('Snake Plot: Cluster-wise Trend of Standardized Feature Values', fontsize=14, fontweight='bold')
    plt.xlabel("Features", fontsize=12, fontweight='bold')  
    plt.ylabel("Standardized Value (Z-Scores)", fontsize=12, fontweight='bold')
    plt.show()


@timer
def plot_relative_imp(raw_df: pd.DataFrame, features: list[str], target_cluster: str, filter_col_keywords: list[str] = None, compare_type: str = 'Normalized'):
    """
    Plots a heatmap of relative feature importance based on cluster averages versus population averages.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): List of feature names to analyze.
        target_cluster (str): The name of the cluster column.
        filter_col_keywords (list[str], optional): A list of keywords to filter feature names (case-insensitive). Defaults to None.
        compare_type (str, optional): Determines how values are scaled for coloring in the heatmap. 
                                      Options are 'Normalized' or 'Global'. Defaults to 'Normalized'.

    Returns:
        None: Displays the heatmap.

    Example:
        plot_relative_imp(raw_df, features, 'cluster_col', filter_col_keywords=['keyword1'])
    """
    # Filter features based on filter_col_keywords, case-insensitively
    if filter_col_keywords is None:
        filtered_features = features
    else:
        filtered_features = [col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)]        
    
    X = raw_df[filtered_features]
    y = raw_df[target_cluster]  

    # Concatenate data for cluster averages and population averages
    tmp = pd.concat([X, y], axis=1)
    cluster_avg = tmp.groupby([target_cluster]).mean()  # Calculate cluster averages
    population_avg = X.mean()  # Calculate population averages

    # Calculate relative importance (percentage change from population average)
    relative_imp = cluster_avg / population_avg - 1
    relative_imp = relative_imp.astype(float)

    # Normalize columns if necessary
    col_norm = relative_imp.apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
    
    # Determine the data to show based on compare_type
    if compare_type == 'Global':
        show_data = relative_imp
    else:
        show_data = col_norm

    # Plot the heatmap
    plt.figure(figsize=(8, 2))
    plt.title(f'Relative Importance for Features with Actual Imp Annotation @Center=0&colormap:{compare_type}', fontsize=10, fontweight='bold')
    sns.heatmap(data=show_data, 
                annot=relative_imp.round(2), 
                fmt='.2f', 
                cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True),
                center=0,
                cbar=False)
    plt.xticks(rotation=90)
    plt.xlabel("Features", fontsize=12, fontweight='bold')  
    plt.ylabel(f"Cluster - {target_cluster}", fontsize=12, fontweight='bold')
    plt.show()

    
    
#############################################################################################

def sort_key(val):
    if val == '= 0':
        return float('-inf')
    if isinstance(val, pd.Interval):
        return val.left
    return float('inf')


@timer
def plot_grouped_bar_by_bins(
    raw_df: pd.DataFrame, 
    features: list[str], 
    target_cluster: str, 
    filter_col_keywords: list[str] = None,
    n_bins: int = 10
):
    """
    Plots grouped bar charts showing the frequency of values in bins for each feature, 
    split by the target_cluster. The bins are created using `pd.cut`, and zero values 
    are separated. The chart is displayed in a subplot grid (2 columns).

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): List of feature names to bin and plot.
        target_cluster (str): The name of the target cluster column.
        filter_col_keywords (list[str], optional): A list of keywords to filter feature names 
                                                    (case-insensitive). Defaults to None.
        n_bins (int, optional): The number of bins to create for non-zero values. Defaults to 10.

    Returns:
        None: Displays the grouped bar chart.
    
    Example:
        plot_grouped_bar_by_bins(raw_df, features, 'cluster_col', filter_col_keywords=['keyword1'])
    """
    # Filter features based on filter_col_keywords, case-insensitively
    if filter_col_keywords is None:
        filtered_features = features
    else:
        filtered_features = [col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)]

    custom_colors = get_palette(target_cluster, raw_df)

    # Setup subplots
    num_features = len(filtered_features)
    num_cols = 2
    num_rows = (num_features + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 5))
    axes = axes.flatten()

    for idx, col in enumerate(filtered_features):
        proxy = raw_df[[col, target_cluster]].copy()

        zero_mask = proxy[col] == 0
        zero_part = proxy.loc[zero_mask].copy()
        nonzero_part = proxy.loc[~zero_mask].copy()

        try:
            nonzero_values = nonzero_part[col].round(2)
            bin_edges = np.round(np.linspace(nonzero_values.min(), nonzero_values.max(), n_bins + 1), 2)
            nonzero_part['bin'] = pd.cut(nonzero_values, bins=bin_edges, include_lowest=True)
            zero_part['bin'] = '= 0'
        except Exception as e:
            print(f"Skipping {col} due to binning error: {e}")
            continue

        parts = [df for df in [zero_part, nonzero_part] if not df.empty]
        binned_df = pd.concat(parts) if parts else pd.DataFrame()

        # Sort bins
        binned_df['bin'] = binned_df['bin'].astype("category")
        binned_df['bin'] = binned_df['bin'].cat.set_categories(
            sorted(binned_df['bin'].unique(), key=sort_key), ordered=True)

        # Group and pivot
        grouped = binned_df.groupby(['bin', target_cluster], observed=False).size().unstack(fill_value=0)
        grouped = grouped.loc[sorted(grouped.index, key=sort_key)]

        # Plot on current subplot
        ax = axes[idx]
        grouped.plot(
            kind='bar', 
            stacked=False, 
            ax=ax,
            color=[custom_colors.get(label, 'gray') for label in grouped.columns]
        )
        ax.set_title(f"Grouped Bar Chart of {col}")
        ax.set_xlabel("Binned Ranges")
        ax.set_ylabel("Frequency")
        ax.legend(title=target_cluster)
        ax.tick_params(axis='x', rotation=45)

    # Remove unused axes if any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


@timer
def plot_stacked_bar_by_bins(
    raw_df: pd.DataFrame, 
    features: list[str], 
    target_cluster: str, 
    filter_col_keywords: list[str] = None,
    n_bins: int = 10,
    percent_by: str = 'segment',  # 'segment' or 'bin'
    reference_stat: str = 'mean'  # 'mean' or 'median'
):
    """
    Plots stacked bar charts showing the percentage distribution of binned values for each feature, 
    split by the target_cluster. The chart is displayed with an optional reference line based on 
    the mean or median of each bin.

    Args:
        raw_df (pd.DataFrame): The dataframe containing the raw data.
        features (list[str]): List of feature names to bin and plot.
        target_cluster (str): The name of the target cluster column.
        filter_col_keywords (list[str], optional): A list of keywords to filter feature names 
                                                    (case-insensitive). Defaults to None.
        n_bins (int, optional): The number of bins to create for non-zero values. Defaults to 10.
        percent_by (str, optional): How percentages are calculated. Options are 'segment' or 'bin'. Defaults to 'segment'.
        reference_stat (str, optional): The statistic to use for the reference line. Options are 'mean' or 'median'. Defaults to 'mean'.

    Returns:
        None: Displays the stacked bar chart with optional reference lines.
    
    Example:
        plot_stacked_bar_by_bins(raw_df, features, 'cluster_col', filter_col_keywords=['keyword1'])
    """
    # Filter features based on filter_col_keywords, case-insensitively
    if filter_col_keywords is None:
        filtered_features = features
    else:
        filtered_features = [col for col in features if any(keyword.lower() in col.lower() for keyword in filter_col_keywords)]

    # Get color map from user-defined function
    color_map = get_palette(target_cluster, raw_df)
    
    for col in filtered_features:
        proxy = raw_df[[col, target_cluster]].copy()

        zero_mask = proxy[col] == 0
        zero_part = proxy.loc[zero_mask].copy()
        nonzero_part = proxy.loc[~zero_mask].copy()

        try:
            nonzero_values = nonzero_part[col].round(2)
            bin_edges = np.round(np.linspace(nonzero_values.min(), nonzero_values.max(), n_bins + 1), 2)
            nonzero_part['bin'] = pd.cut(nonzero_values, bins=bin_edges, include_lowest=True)
            zero_part['bin'] = '= 0'
        except Exception as e:
            print(f"Skipping {col} due to binning error: {e}")
            continue

        parts = [df for df in [zero_part, nonzero_part] if not df.empty]
        binned_df = pd.concat(parts) if parts else pd.DataFrame()

        # Sort bins
        binned_df['bin'] = binned_df['bin'].astype("category")
        binned_df['bin'] = binned_df['bin'].cat.set_categories(
            sorted(binned_df['bin'].unique(), key=sort_key), ordered=True)

        # Frequency counts per bin per segment
        counts = binned_df.groupby(['bin', target_cluster], observed=False).size().reset_index(name='count')

        # % within segment or % by bin
        if percent_by == 'segment':
            total = counts.groupby(target_cluster)['count'].transform('sum')
        else:
            total = counts.groupby('bin')['count'].transform('sum')
        counts['percentage'] = counts['count'] / total * 100

        # Global reference (mean or median across segments for each bin)
        pivot = counts.pivot(index='bin', columns=target_cluster, values='percentage')
        if reference_stat == 'mean':
            reference_line = pivot.mean(axis=1)
        elif reference_stat == 'median':
            reference_line = pivot.median(axis=1)
        else:
            raise ValueError("`reference_stat` must be 'mean' or 'median'")
        global_map = reference_line.to_dict()

        # Plot
        plot_df = counts.copy()
        unique_bins = sorted(binned_df['bin'].dropna().unique(), key=sort_key)

        g = sns.FacetGrid(
            data=plot_df, 
            col=target_cluster, 
            col_wrap=1, 
            sharex=True, 
            sharey=False, 
            height=3.5, 
            aspect=2
        )

        # Plot bars using correct color for each segment
        def barplot_with_custom_color(data, color, **kwargs):
            segment = data[target_cluster].iloc[0]
            sns.barplot(
                data=data, 
                x='bin', y='percentage', 
                color=color_map.get(segment, 'gray'), 
                order=unique_bins, 
                **kwargs
            )

        g.map_dataframe(barplot_with_custom_color)

        # Add horizontal reference bars
        for ax, segment in zip(g.axes.flat, g.col_names):
            for idx, bin_label in enumerate(unique_bins):
                if bin_label in global_map:
                    y = global_map[bin_label]
                    ax.bar(
                        x=idx, height=y, width=0.8,
                        fill=False, edgecolor='red', linestyle='--', linewidth=1.5
                    )
                    ax.text(
                        idx, y + 1, f"{y:.1f}%", 
                        ha='center', va='bottom', color='red', fontsize=8
                    )

        g.set_axis_labels("Binned Values", "Frequency Percentage (%)")
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle(f"Binned Percentage Distribution of {col} with {reference_stat} reference line", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()
        

        
#############################################################################################

@timer
def cluster_feature_stats_table(
    raw_df: pd.DataFrame,
    features: list[str],
    target_cluster: str,
    filter_col_keywords: list[str] = None,
    stats: list[str] = ['count','mean', 'mode', 'median', 'std']
) -> pd.DataFrame:
    """
    Create a summary table of statistics for each feature and each cluster.

    Args:
        df (pd.DataFrame): Input dataframe.
        feature_list (list[str]): List of feature columns to consider.
        target_cluster (str): Name of the cluster column.
        filter_col_keywords (list[str], optional): Filter features by keywords (case-insensitive). Defaults to None.
        stats (list[str], optional): List of statistics to compute. Options: 'count','mean', 'mode', 'median', 'std'. Defaults to all.

    Returns:
        pd.DataFrame: MultiIndex dataframe with features and stats per cluster as columns.
    """
    df = raw_df.copy()

    if filter_col_keywords is not None:
        filtered_features = [f for f in features if any(k.lower() in f.lower() for k in filter_col_keywords)]
    else:
        filtered_features = features

    clusters = sorted(df[target_cluster].unique())

    results = {}

    for cluster in clusters:
        cluster_df = df[df[target_cluster] == cluster][filtered_features]

        cluster_stats = {}

        if 'mean' in stats:
            cluster_stats['mean'] = cluster_df.mean()
        if 'std' in stats:
            cluster_stats['std'] = cluster_df.std()
        if 'count' in stats:
            cluster_stats['count'] = cluster_df.count()
        if 'mode' in stats:
            cluster_stats['mode'] = cluster_df.mode().iloc[0]
        if 'median' in stats:
            cluster_stats['median'] = cluster_df.median()

        cluster_df_stats = pd.concat(cluster_stats, axis=1)
        cluster_df_stats.columns = pd.MultiIndex.from_product([[cluster], cluster_df_stats.columns])
        results[cluster] = cluster_df_stats

    final_df = pd.concat(results.values(), axis=1)

    final_df = final_df.round(2)

    return final_df