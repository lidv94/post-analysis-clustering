import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from post_analysis_clustering.utils import timer, get_palette
from post_analysis_clustering.visualize.base import BaseVis

class PerformanceMetrics(BaseVis):
    """
    A class for evaluating the quality of clustering results using internal validation metrics 
    and visualizing the performance across commonly used indices.

    Inherits:
        BaseVis: Provides foundational attributes like the input DataFrame, selected features, 
                 cluster labels, and primary key.

    Attributes:
        df (pd.DataFrame): The input dataset containing features and cluster labels.
        features (list[str]): List of feature column names used for metric evaluation.
        target_cluster (str): Name of the column indicating cluster membership.
        primary_key (str): Unique identifier for each row.
        scale (bool): Whether to standardize features using z-score normalization before evaluation.
        sample_size (int or None): If specified, limits computation to a random subset of this size.
        stratify (bool): If sampling is applied, maintains cluster proportions in the subsample.

    Methods:
        _compute_cluster_metrics():
            Computes three internal clustering evaluation metrics:
            - Silhouette Score (higher is better)
            - Davies-Bouldin Index (lower is better)
            - Calinski-Harabasz Index (higher is better)

        plot_cluster_metric_ranges():
            Visualizes each clustering metric with annotated score lines and ideal targets 
            on adaptive horizontal plots.

    Raises:
        TypeError: If `scale` or `stratify` is not a boolean.
        ValueError: If `sample_size` is not a positive integer or None.

    Example:
        >>> metrics = PerformanceMetrics(
                df=clustered_data,
                features=feature_cols,
                target_cluster="cluster",
                primary_key="id",
                scale=True,
                sample_size=10000,
                stratify=True
            )
        >>> metrics.plot_cluster_metric_ranges()
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
    def _compute_cluster_metrics(self) -> dict:
        """
        Compute internal clustering validation metrics to assess cluster quality.

        This function calculates three widely-used clustering evaluation metrics:

        - **Silhouette Score**: Measures how similar an object is to its own cluster 
          compared to other clusters. Ranges from -1 to 1 (higher is better).

        - **Davies-Bouldin Index (DBI)**: Computes the average similarity between each cluster 
          and its most similar one. Lower values indicate better clustering (ideal is 0).

        - **Calinski-Harabasz Index (CHI)**: Measures the ratio of between-cluster dispersion 
          to within-cluster dispersion. Higher values indicate more distinct clusters.

        Optionally supports feature standardization and stratified sampling for large datasets.

        Returns:
            dict: Dictionary with keys:
                - 'Silhouette Score'
                - 'Davies-Bouldin Index'
                - 'Calinski-Harabasz Index'
              and corresponding float values or None if computation fails.
        """
        df_in = self.df.copy()

        if self.sample_size is not None and self.sample_size < len(df_in):
            stratify_col = df_in[self.target_cluster] if self.stratify else None
            df_in, _ = train_test_split(
                df_in,
                train_size=self.sample_size,
                stratify=stratify_col,
                random_state=42
            )

        X = df_in[self.features]
        y = df_in[self.target_cluster]

        X = StandardScaler().fit_transform(X) if self.scale else X.values

        metrics = {}
        try:
            metrics['Silhouette Score'] = silhouette_score(X, y)
        except Exception:
            metrics['Silhouette Score'] = None

        try:
            metrics['Davies-Bouldin Index'] = davies_bouldin_score(X, y)
        except Exception:
            metrics['Davies-Bouldin Index'] = None

        try:
            metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X, y)
        except Exception:
            metrics['Calinski-Harabasz Index'] = None

        return metrics

    @timer
    def plot_cluster_metric_ranges(self):
        """
        Compute and visualize internal clustering metrics with annotated ranges and ideal targets.

        This function generates horizontal plots for three key clustering metrics:

        - **Silhouette Score**: Indicates cohesion and separation; range [-1, 1]. Higher is better.
        - **Davies-Bouldin Index**: Measures inter/intra cluster similarity; lower is better. Ideal is 0.
        - **Calinski-Harabasz Index**: Measures cluster compactness and separation; higher is better.

        Visual annotations include:
          - Blue vertical line for the actual score.
          - Green dashed line for the ideal target.
          - Adaptive x-axis limits (e.g., `score + 3` for DBI, `score * 1.2` for CHI).

        Returns:
            None: Displays matplotlib plots of each clustering metric with range annotations.
        """
        metric_results = self._compute_cluster_metrics()

        metric_info = {
            "Silhouette Score": {"range": (-1, 1), "best": 1},
            "Davies-Bouldin Index": {"range": (0, float('inf')), "best": 0},
            "Calinski-Harabasz Index": {"range": (0, float('inf')), "best": "higher"}
        }

        height_per_plot = 1.5  
        fig, axes = plt.subplots(nrows=len(metric_results), figsize=(7, height_per_plot * len(metric_results)))
        if len(metric_results) == 1:
            axes = [axes]

        for ax, (metric, score) in zip(axes, metric_results.items()):
            info = metric_info.get(metric)
            if info is None or score is None:
                continue

            r_min, r_max = info["range"]
            best = info["best"]

            plot_r_max = r_max
            if r_max == float('inf'):
                if metric == "Davies-Bouldin Index":
                    plot_r_max = max(score + 3, r_min + 1)
                elif metric == "Calinski-Harabasz Index":
                    plot_r_max = max(score * 1.2, r_min + 100)

            # Plot ideal line
            if best == "higher":
                ideal_x = plot_r_max * 0.99
                ax.axvline(x=ideal_x, color='green', linestyle='--', label='Ideal (Higher Better)', linewidth=1.5)
            else:
                ideal_x = best
                offset = 0.02 * (plot_r_max - r_min)
                if ideal_x == r_min:
                    ideal_x_plot = ideal_x + offset
                elif ideal_x == plot_r_max:
                    ideal_x_plot = ideal_x - offset
                else:
                    ideal_x_plot = ideal_x
                ax.axvline(x=ideal_x_plot, color='green', linestyle='--', label='Ideal', linewidth=1.5)

            ax.axvline(x=score, color='blue', linestyle='-', label=f'Score', linewidth=2)

            ax.set_xlim(r_min, plot_r_max)

            r_max_str = "∞" if r_max == float('inf') else f"{r_max}"
            ax.set_title(f"{metric} (Score: {score:.2f})\nRange: [{r_min}, {r_max_str}]", fontsize=9)
            ax.set_xlabel("Metric Value", fontsize=8)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_yticks([])
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, axis='x', linestyle=':', alpha=0.6)

        plt.tight_layout(h_pad=1.0)
        plt.show()