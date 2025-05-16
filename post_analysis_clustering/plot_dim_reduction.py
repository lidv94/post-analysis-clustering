import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from post_analysis_clustering.utils import timer, get_palette

def plot_dim_reduction(
    raw_df: pd.DataFrame,
    features: list[str],
    target_cluster: str,
    method: str = 'pca',  # 'pca' or 'tsne'
    n_components: int = 2,
    title: str = None,
    scale: bool = True,
    sample_size: int = None,
    stratify: bool = True  # preserve cluster proportions when downsampling
):
    """
    Visualize clusters using PCA or t-SNE dimensionality reduction.
    Description:
    - Each point corresponds to one data sample projected from high-dimensional feature space down to 2 or 3 dimensions for visualization.
    - PCA (Principal Component Analysis) is a linear technique that preserves directions of maximum global variance. 
      It provides a broad overview by approximating distances well, but loses some detail by discarding components with less variance, 
      potentially missing subtle or non-linear patterns.
      PCA shows the big overall patterns quickly but might miss tiny details.
    - t-SNE (t-distributed Stochastic Neighbor Embedding) is a non-linear method that preserves local neighborhoods, 
      making it effective for revealing fine cluster structures, although global distances may be distorted.
      t-SNE shows detailed, close-knit groups clearly but can distort the big picture.
    - Well-separated, compact clusters suggest good cluster quality, while overlapping clusters may indicate less distinct groups.   

    Args:
        raw_df (pd.DataFrame): Input dataframe.
        features (list[str]): List of feature columns to use for dimensionality reduction.
        target_cluster (str): Column name for cluster/segment labels.
        method (str): 'pca' or 'tsne'. Default is 'pca'.
        n_components (int): Number of dimensions to reduce to (2 or 3 recommended). Default is 2.
        title (str): Optional plot title.
        scale (bool): Whether to standardize features before reduction.
        sample_size (int, optional): Total number of samples to use. If None, uses all data.
        stratify (bool): Whether to stratify sampling to preserve cluster proportions.

    Returns:
        None. Displays a 2D or 3D scatter plot of the data.
    """
    df_in = raw_df.copy()
    
    # Downsample with optional stratification
    if sample_size is not None and sample_size < len(df_in):
        stratify_col = df_in[target_cluster] if stratify else None
        df_in, _ = train_test_split(df_in, 
                                    train_size=sample_size, 
                                    stratify=stratify_col, 
                                    random_state=42)
        
    X = df_in[features]
    y = df_in[target_cluster]

    # Standardize features
    if scale:
        X = StandardScaler().fit_transform(X)

    # Select reduction method
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(X)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(X)
    else:
        raise ValueError("Invalid method. Use 'pca' or 'tsne'.")

    # Prepare plot data
    plot_df = pd.DataFrame(embedding, columns=[f"Component {i+1}" for i in range(n_components)])
    plot_df[target_cluster] = y.values

    # Use custom palette
    custom_colors = get_palette(target_cluster, df_in)
    
    # Title handling with scaling and sampling info
    scaling_note = " (scaled)" if scale else " (unscaled)"

    if sample_size is not None and sample_size < len(raw_df):
        sampling_note = f" - downsampled to {sample_size} samples"
    else:
        sampling_note = " - full dataset"

    final_title = title or f"{method.upper()} Projection{scaling_note}{sampling_note}"

    # Plot
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x='Component 1',
            y='Component 2',
            hue=target_cluster,
            data=plot_df,
            palette=custom_colors,
            s=50,
            edgecolor='k'
        )
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for name, group in plot_df.groupby(target_cluster):
            ax.scatter(
                group['Component 1'],
                group['Component 2'],
                group['Component 3'],
                label=name,
                color=custom_colors.get(name, 'gray'),
                s=50,
                edgecolor='k'
            )
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    else:
        raise ValueError("Only 2 or 3 components are supported for plotting.")

    plt.title(final_title) 
    # "Projection" : the process of mapping high-dimensional data into a lower-dimensional space.
    plt.legend()
    plt.tight_layout()
    plt.show()
