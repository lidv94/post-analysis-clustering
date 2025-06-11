import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from post_analysis_clustering.utils import timer, get_palette
from post_analysis_clustering.visualize.base import BaseVis

class DimensionalityReduction(BaseVis):
    """
    A class for applying dimensionality reduction techniques (e.g., PCA, t-SNE, UMAP)
    to high-dimensional cluster data for visualization purposes.

    Inherits from:
        BaseVis: A base class containing common attributes and utilities for clustering visualization.

    Attributes:
        df (pd.DataFrame): The input dataframe.
        features (list[str]): A list of column names to use for dimensionality reduction.
        target_cluster (str): Name of the column indicating cluster labels.
        primary_key (str): The column name that uniquely identifies each record.
        scale (bool): Whether to standardize the features before reduction. Defaults to True.
        sample_size (int or None): Optional limit for the number of samples to process. 
            If set, the data is sampled (optionally stratified). Defaults to None.
        stratify (bool): Whether to stratify sampling based on `target_cluster`. Defaults to True.

    Raises:
        TypeError: If `scale` or `stratify` is not a boolean.
        ValueError: If `sample_size` is not None or a positive integer.

    Example:
        reducer = DimensionalityReduction(
            df=data, 
            features=numeric_cols, 
            target_cluster='cluster', 
            primary_key='customer_id',
            scale=True,
            sample_size=5000,
            stratify=True
        )
    """
    def __init__(self, 
                 df, 
                 features, 
                 target_cluster, 
                 primary_key,
                 scale: bool = True,
                 sample_size: int = None,
                 stratify: bool = True):
        super().__init__(df, features, target_cluster, primary_key)
        self.scale = scale
        self.sample_size = sample_size
        self.stratify = stratify
        
        self._validate_attributes()

    def _validate_attributes(self):
        if not isinstance(self.scale, bool):
            raise TypeError(f"Expected 'scale' to be bool, got {type(self.scale).__name__}")
        
        if self.sample_size is not None:
            if not (isinstance(self.sample_size, int) and self.sample_size > 0):
                raise ValueError(f"'sample_size' must be a positive integer or None, got {self.sample_size}")
        
        if not isinstance(self.stratify, bool):
            raise TypeError(f"Expected 'stratify' to be bool, got {type(self.stratify).__name__}")
    
    @timer
    def plot_dim_reduction(
        self,
        method: str = 'pca',  # 'pca' or 'tsne'
        n_components: int = 2, # 2 or 3
        title: str = None,
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
            method (str): 'pca' or 'tsne'. Default is 'pca'.
            n_components (int): Number of dimensions to reduce to (2 or 3 recommended). Default is 2.
            title (str): Optional plot title.
            scale (bool): Whether to standardize features before reduction.
            sample_size (int, optional): Total number of samples to use. If None, uses all data.
            stratify (bool): Whether to stratify sampling to preserve cluster proportions.

        Returns:
            None. Displays a 2D or 3D scatter plot of the data.
        """
        df_in = self.df.copy()

        # Downsample with optional stratification
        if self.sample_size is not None and self.sample_size < len(df_in):
            stratify_col = df_in[self.target_cluster] if self.stratify else None
            df_in, _ = train_test_split(df_in, 
                                        train_size=self.sample_size, 
                                        stratify=stratify_col, 
                                        random_state=42)

        X = df_in[self.features]
        y = df_in[self.target_cluster]

        # Standardize features
        if self.scale:
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
        plot_df[self.target_cluster] = y.values

        # Use custom palette
        custom_colors = get_palette(self.target_cluster, df_in)

        # Title handling with scaling and sampling info
        scaling_note = " (scaled)" if self.scale else " (unscaled)"

        if self.sample_size is not None and self.sample_size < len(self.df):
            sampling_note = f" - downsampled to {self.sample_size} samples"
        else:
            sampling_note = " - full dataset"

        final_title = title or f"{method.upper()} Projection{scaling_note}{sampling_note}"

        # Plot
        if n_components == 2:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x='Component 1',
                y='Component 2',
                hue=self.target_cluster,
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
            for name, group in plot_df.groupby(self.target_cluster):
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
        