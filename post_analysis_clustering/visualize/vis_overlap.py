import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import chi2
from post_analysis_clustering.utils import timer, get_palette
from post_analysis_clustering.visualize.base import BaseVis

class OverlapPairPlot(BaseVis):
    """
    A class for visualizing overlap between clusters or segments using convex hulls 
    and centroid plots across pairs of selected features.

    This class supports:
    - Visualizing the boundary (convex hull) of each cluster in 2D feature spaces.
    - Showing centroids computed from full feature space projected into 2D.
    - Creating a lower-triangle pairplot with polygon overlays for all feature pairs.

    Inherits:
        BaseVis: Provides the base DataFrame, selected feature names, target cluster column, and primary key.

    Attributes:
        df (pd.DataFrame): The input dataset containing features and cluster labels.
        features (list[str]): List of numerical feature columns used in the pairwise analysis.
        target_cluster (str): Column name indicating cluster/segment assignments.
        primary_key (str): Unique identifier for each row in the dataset.

    Methods:
        plot_segment_overlap(Xi, Xj):
            Plots convex hulls and centroids in a single 2D scatter plot using two features.

        _plot_segment_overlap_polygon_ax(Xi, Xj, ax):
            Internal helper for plotting convex hulls and centroids on a specific matplotlib axis.

        plot_custom_polygon_pairplot():
            Displays a lower-triangle pairplot across all feature combinations, with hulls and centroids per segment.

    Example:
        >>> overlap = OverlapPairPlot(
                df=my_df, 
                features=["feature1", "feature2", "feature3"], 
                target_cluster="cluster", 
                primary_key="id"
            )
        >>> overlap.plot_segment_overlap("feature1", "feature2")
        >>> overlap.plot_custom_polygon_pairplot()
    """
    def __init__(self, 
                 df, 
                 features, 
                 target_cluster, 
                 primary_key):
        super().__init__(df, 
                         features, 
                         target_cluster, 
                         primary_key)
                
    @timer
    def plot_segment_overlap(self, 
                             Xi: str, 
                             Xj: str):
        """
        Plot convex hulls and centroids (computed as means) for each segment using two selected features.

        Args:
            Xi (str): Name of the feature to be plotted on the x-axis.
            Xj (str): Name of the feature to be plotted on the y-axis.

        Returns:
            None: Displays a matplotlib plot.
        """  
        try:
            plot_df = self.df[[Xi, Xj, self.target_cluster]]
            unique_segments = plot_df[self.target_cluster].unique()

            # Compute centroids directly from mean of each cluster
            centroid_df = self.df.groupby(self.target_cluster)[self.features].mean()

            # Get indices of Xi and Xj in feature_list
            Xi_idx = self.features.index(Xi)
            Xj_idx = self.features.index(Xj)

            custom_palette = get_palette(self.target_cluster, self.df)

            plt.figure(figsize=(8, 6))

            for segment in unique_segments:
                segment_data = plot_df[plot_df[self.target_cluster] == segment][[Xi, Xj]].values
                centroid = centroid_df.loc[segment].values  # Centroid as 1D array

                if len(segment_data) >= 3:
                    try:
                        hull = ConvexHull(segment_data)
                        hull_vertices = np.append(hull.vertices, hull.vertices[0])  # Close the polygon

                        plt.plot(segment_data[hull_vertices, 0], segment_data[hull_vertices, 1], 
                                 label=f'Segment {segment}', color=custom_palette[segment])

                        plt.scatter(centroid[Xi_idx], centroid[Xj_idx], marker='x',
                                    color=custom_palette[segment], s=100, label=f'Centroid {segment}')
                    except QhullError:
                        print(f"Error: Convex Hull failed for segment {segment}.")
                        continue

            plt.xlabel(Xi)
            plt.ylabel(Xj)
            plt.legend()
            plt.title(f'Segment Overlap for {Xi} vs {Xj}')
            plt.show()

        except Exception as e:
            print(f"Unexpected error occurred while plotting segment overlap: {e}")

    #################################################################

    @timer
    def _plot_segment_overlap_polygon_ax(self,
                                        Xi: str, 
                                        Xj: str,  
                                        ax=None) -> None:
        """
        Plots convex hulls (boundaries) and centroids (cluster means) for each segment on a given axis.

        Args:
            Xi (str): Name of the feature to be plotted on the x-axis.
            Xj (str): Name of the feature to be plotted on the y-axis.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot the graph on. Defaults to None.

        Returns:
            None: Displays convex hulls and centroids on the provided matplotlib axis.
        """
        plot_df = self.df[[Xi, Xj, self.target_cluster]]
        unique_segments = plot_df[self.target_cluster].unique()
        custom_palette = get_palette(self.target_cluster, self.df)

        centroid_df = self.df.groupby(self.target_cluster)[self.features].mean()

        for segment in unique_segments:
            segment_data = plot_df[plot_df[self.target_cluster] == segment][[Xi, Xj]].values

            if len(segment_data) >= 3:
                try:
                    hull = ConvexHull(segment_data)
                    hull_vertices = np.append(hull.vertices, hull.vertices[0])

                    ax.plot(segment_data[hull_vertices, 0], segment_data[hull_vertices, 1], 
                            label=f'{self.target_cluster} = {segment}', color=custom_palette[segment])

                    centroid = centroid_df.loc[segment]
                    ax.scatter(centroid[Xi], centroid[Xj], marker='x', color=custom_palette[segment], s=100)

                except QhullError as qhull_error:
                    print(f"Error computing convex hull for segment {segment}: {qhull_error}")
                    continue
                except Exception as e:
                    print(f"An unexpected error occurred while plotting convex hull for segment {segment}: {e}")
                    continue

        ax.set_xlabel(Xi)
        ax.set_ylabel(Xj)

    @timer
    def plot_custom_polygon_pairplot(self) -> None:
        """
        Creates a customized lower-triangle pairplot showing convex hulls and mean centroids for each segment.

        Returns:
            None: Displays a pairplot with hulls and centroids.
        """
        num_features = len(self.features)
        fig, axes = plt.subplots(num_features, num_features, figsize=(num_features * 3, num_features * 3))
        legend_handles = None

        for i in range(num_features):
            for j in range(num_features):
                ax = axes[i, j]
                if i < j or i == j:
                    ax.set_visible(False)
                else:
                    Xi, Xj = self.features[j], self.features[i]
                    try:
                        self._plot_segment_overlap_polygon_ax( Xi, Xj, ax)

                        if i == num_features - 1:
                            ax.set_xlabel(Xi)
                        else:
                            ax.set_xticklabels([])

                        if j == 0:
                            ax.set_ylabel(Xj)
                        else:
                            ax.set_yticklabels([])

                        if legend_handles is None:
                            legend_handles, legend_labels = ax.get_legend_handles_labels()

                        if ax.get_legend():
                            ax.get_legend().remove()

                    except Exception as e:
                        print(f"Error plotting subplot ({Xj} vs {Xi}): {e}")
                        ax.set_visible(False)

        fig.suptitle(f"Segment Overlap Pairplot with Polygon", fontsize=16)

        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='upper right', title='Segments')

        plt.tight_layout()
        plt.show()


    #################################################################
    ######################### END OF POLYGON ########################
    #################################################################
    ######################### START ELLIPSES ########################
    #################################################################       
    def _validate_attributes(self):
        """
        Validate the `confidence_level` attribute to ensure it is within the acceptable range.

        This method checks that:
        - If `confidence_level` is not None:
            - It must be of type `float` or `int`.
            - Its value must be strictly greater than 0 and less than or equal to 1.

        Raises:
            TypeError: If `confidence_level` is not a float or int.
            ValueError: If `confidence_level` is not within the range (0, 1].

        Returns:
            None
        """
        if self.confidence_level is not None:        
            if not isinstance(self.confidence_level, (float, int)) or not (0 < self.confidence_level <= 1):
                raise ValueError("`confidence_level` must be a float between 0 and 1.")
                
    @timer
    def plot_segment_ellipse(
        self,
        Xi: str,
        Xj: str,
        confidence_level: float = 0.95
    ):
        """
        Plot confidence ellipses and centroids for each segment using two selected features.

        Args:
            Xi (str): Feature for x-axis.
            Xj (str): Feature for y-axis.
            confidence_level (float, optional): Confidence level for ellipse size.

        Returns:
            None: Displays the plot.
        """
        try:
            self.confidence_level = confidence_level
            self._validate_attributes()
            
            plot_df = self.df[[Xi, Xj, self.target_cluster]]
            unique_segments = plot_df[self.target_cluster].unique()
            custom_palette = get_palette(self.target_cluster, self.df)
            plotted_segments = set()

            # If no feature_list provided, just use Xi and Xj for centroid
            if self.features is None:
                centroid_df = self.df.groupby(self.target_cluster)[[Xi, Xj]].mean()
                use_full_features = False
            else:
                centroid_df = self.df.groupby(self.target_cluster)[self.features].mean()
                use_full_features = True

            plt.figure(figsize=(8, 6))
            ax = plt.gca()

            for segment in unique_segments:
                segment_data = plot_df[plot_df[self.target_cluster] == segment][[Xi, Xj]].values

                if len(segment_data) >= 2:
                    # Calculate ellipse parameters
                    mean = np.mean(segment_data, axis=0)
                    cov = np.cov(segment_data.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = eigvals.argsort()[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]

                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                    width, height = 2 * np.sqrt(eigvals * chi2.ppf(confidence_level, 2))

                    ellipse = Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        edgecolor=custom_palette[segment],
                        facecolor='none',
                        lw=2,
                        label=f'Segment {segment}' if segment not in plotted_segments else None
                    )
                    ax.add_patch(ellipse)

                    # Plot centroid
                    if use_full_features:
                        centroid = centroid_df.loc[segment]
                        cx, cy = centroid[Xi], centroid[Xj]
                    else:
                        cx, cy = mean[0], mean[1]

                    ax.scatter(cx, cy, marker='x', color=custom_palette[segment], s=100, label=f'Centroid {segment}' if segment not in plotted_segments else None)

                    plotted_segments.add(segment)

            ax.set_xlabel(Xi)
            ax.set_ylabel(Xj)
            ax.set_title(f'Segment Ellipse Overlap for {Xi} vs {Xj}')
            ax.legend(loc='upper right')

            plt.show()

        except Exception as e:
            print(f"Error in plot_segment_ellipse: {e}")

    @timer
    def _plot_segment_ellipse_ax(
        self,
        Xi: str,
        Xj: str,
        ax=None,
        confidence_level: float = 0.95
    ):
        """
        Plots confidence ellipses for each segment on a specified axis using the specified confidence level.

        Args:
            Xi (str): Name of the feature to be plotted on the x-axis.
            Xj (str): Name of the feature to be plotted on the y-axis.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. Defaults to None.
            confidence_level (float, optional): Confidence level for the ellipses. Defaults to 0.95.

        Returns:
            None
        """
        try:
            plot_df = self.df[[Xi, Xj, self.target_cluster]]
            unique_segments = plot_df[self.target_cluster].unique()
            custom_palette = get_palette(self.target_cluster, self.df)
            plotted_segments = set()

            for segment in unique_segments:
                segment_data = plot_df[plot_df[self.target_cluster] == segment][[Xi, Xj]].values

                if len(segment_data) >= 2:
                    # Mean as cluster center
                    mean = np.mean(segment_data, axis=0)
                    cov = np.cov(segment_data.T)

                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = eigvals.argsort()[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]

                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                    width, height = 2 * np.sqrt(eigvals * chi2.ppf(confidence_level, 2))

                    ellipse = Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        edgecolor=custom_palette[segment],
                        facecolor='none',
                        lw=2,
                        label=f"{self.target_cluster} = {segment}" if segment not in plotted_segments else None 
                    )
                    ax.add_patch(ellipse)

                    # Plot mean as centroid
                    label = segment if segment not in plotted_segments else None
                    ax.scatter(mean[0], mean[1], marker='x', color=custom_palette[segment], s=100)
                    plotted_segments.add(segment)

            ax.set_xlabel(Xi)
            ax.set_ylabel(Xj)

        except Exception as e:
            print(f"Error plotting ellipse for ({Xj} vs {Xi}): {e}")
            if ax:
                ax.set_visible(False)

    @timer
    def plot_custom_ellipse_pairplot(
        self,
        confidence_level: float = 0.95
    ):
        """
        Creates a customized lower-triangle pairplot with confidence ellipses representing segment boundaries.

        Args:
            confidence_level (float, optional): Confidence level for the ellipses. Defaults to 0.95.

        Returns:
            None
        """
        self.confidence_level = confidence_level
        self._validate_attributes()
        
        num_features = len(self.features)
        fig, axes = plt.subplots(num_features, num_features, figsize=(num_features * 3, num_features * 3))
        legend_handles = None

        for i in range(num_features):
            for j in range(num_features):
                ax = axes[i, j]
                if i < j or i == j:
                    ax.set_visible(False)
                else:
                    Xi, Xj = self.features[j], self.features[i]
                    try:
                        self._plot_segment_ellipse_ax( Xi, Xj, ax=ax, confidence_level=confidence_level)

                        if i == num_features - 1:
                            ax.set_xlabel(Xi)
                        else:
                            ax.set_xticklabels([])

                        if j == 0:
                            ax.set_ylabel(Xj)
                        else:
                            ax.set_yticklabels([])

                        if legend_handles is None:
                            legend_handles, legend_labels = ax.get_legend_handles_labels()

                        if ax.get_legend():
                            ax.get_legend().remove()

                    except Exception as e:
                        print(f"Error in subplot ({Xj} vs {Xi}): {e}")
                        ax.set_visible(False)

        fig.suptitle(f"Segment Overlap Pairplot with Ellipses (confidence_level = {confidence_level})", fontsize=16)

        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='upper right', title='Segments')

        plt.tight_layout()
        plt.show()