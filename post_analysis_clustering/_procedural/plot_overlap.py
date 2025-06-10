import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import chi2
from post_analysis_clustering.utils import timer, get_palette

@timer
def plot_segment_overlap(data: pd.DataFrame, 
                         Xi: str, 
                         Xj: str, 
                         segments: str, 
                         feature_list: list[str]):
    """
    Plot convex hulls and centroids (computed as means) for each segment using two selected features.

    Args:
        data (pd.DataFrame): The dataset containing the features and segment labels.
        Xi (str): Name of the feature to be plotted on the x-axis.
        Xj (str): Name of the feature to be plotted on the y-axis.
        segments (str): Column name in the DataFrame indicating the segment or cluster labels.
        feature_list (list[str]): List of all feature names to compute centroids.

    Returns:
        None: Displays a matplotlib plot.
    """  
    try:
        plot_df = data[[Xi, Xj, segments]]
        unique_segments = plot_df[segments].unique()

        # Compute centroids directly from mean of each cluster
        centroid_df = data.groupby(segments)[feature_list].mean()

        # Get indices of Xi and Xj in feature_list
        Xi_idx = feature_list.index(Xi)
        Xj_idx = feature_list.index(Xj)
        
        custom_palette = get_palette(segments, data)

        plt.figure(figsize=(8, 6))

        for segment in unique_segments:
            segment_data = plot_df[plot_df[segments] == segment][[Xi, Xj]].values
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
def plot_segment_overlap_polygon_ax(data: pd.DataFrame, 
                                    Xi: str, 
                                    Xj: str, 
                                    segments: str, 
                                    feature_list: list[str], 
                                    ax=None) -> None:
    """
    Plots convex hulls (boundaries) and centroids (cluster means) for each segment on a given axis.

    Args:
        data (pd.DataFrame): DataFrame containing the dataset.
        Xi (str): Name of the feature to be plotted on the x-axis.
        Xj (str): Name of the feature to be plotted on the y-axis.
        segments (str): Column name representing segment/cluster labels.
        feature_list (list[str]): List of all feature names used to compute the mean centroids.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot the graph on. Defaults to None.

    Returns:
        None: Displays convex hulls and centroids on the provided matplotlib axis.
    """
    plot_df = data[[Xi, Xj, segments]]
    unique_segments = plot_df[segments].unique()
    custom_palette = get_palette(segments, data)

    centroid_df = data.groupby(segments)[feature_list].mean()

    for segment in unique_segments:
        segment_data = plot_df[plot_df[segments] == segment][[Xi, Xj]].values
        
        if len(segment_data) >= 3:
            try:
                hull = ConvexHull(segment_data)
                hull_vertices = np.append(hull.vertices, hull.vertices[0])
                
                ax.plot(segment_data[hull_vertices, 0], segment_data[hull_vertices, 1], 
                        label=f'{segments} = {segment}', color=custom_palette[segment])
                
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
def plot_custom_polygon_pairplot(data: pd.DataFrame, 
                                 features: list[str], 
                                 segments: str) -> None:
    """
    Creates a customized lower-triangle pairplot showing convex hulls and mean centroids for each segment.

    Args:
        data (pd.DataFrame): DataFrame containing the full dataset.
        features (list[str]): List of feature names to include in the pairplot.
        segments (str): Column name representing cluster assignments.

    Returns:
        None: Displays a pairplot with hulls and centroids.
    """
    num_features = len(features)
    fig, axes = plt.subplots(num_features, num_features, figsize=(num_features * 3, num_features * 3))
    legend_handles = None

    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i, j]
            if i < j or i == j:
                ax.set_visible(False)
            else:
                Xi, Xj = features[j], features[i]
                try:
                    plot_segment_overlap_polygon_ax(data, Xi, Xj, segments, features, ax)

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

@timer
def plot_segment_ellipse(
    data: pd.DataFrame,
    Xi: str,
    Xj: str,
    segments: str,
    confidence_level: float = 0.95,
    feature_list: list[str] = None
):
    """
    Plot confidence ellipses and centroids for each segment using two selected features.

    Args:
        data (pd.DataFrame): Dataset containing features and segment labels.
        Xi (str): Feature for x-axis.
        Xj (str): Feature for y-axis.
        segments (str): Column for segment labels.
        confidence_level (float, optional): Confidence level for ellipse size.
        feature_list (list[str], optional): List of features for centroid calculation (default uses Xi and Xj only).

    Returns:
        None: Displays the plot.
    """
    try:
        plot_df = data[[Xi, Xj, segments]]
        unique_segments = plot_df[segments].unique()
        custom_palette = get_palette(segments, data)
        plotted_segments = set()

        # If no feature_list provided, just use Xi and Xj for centroid
        if feature_list is None:
            centroid_df = data.groupby(segments)[[Xi, Xj]].mean()
            use_full_features = False
        else:
            centroid_df = data.groupby(segments)[feature_list].mean()
            use_full_features = True

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        for segment in unique_segments:
            segment_data = plot_df[plot_df[segments] == segment][[Xi, Xj]].values

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
def plot_segment_ellipse_ax(
    data: pd.DataFrame,
    Xi: str,
    Xj: str,
    segments: str,
    ax=None,
    confidence_level: float = 0.95
):
    """
    Plots confidence ellipses for each segment on a specified axis using the specified confidence level.

    Args:
        data (pd.DataFrame): DataFrame containing feature columns and segment labels.
        Xi (str): Name of the feature to be plotted on the x-axis.
        Xj (str): Name of the feature to be plotted on the y-axis.
        segments (str): Name of the column containing segment or cluster labels.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. Defaults to None.
        confidence_level (float, optional): Confidence level for the ellipses. Defaults to 0.95.

    Returns:
        None
    """
    try:
        plot_df = data[[Xi, Xj, segments]]
        unique_segments = plot_df[segments].unique()
        custom_palette = get_palette(segments, data)
        plotted_segments = set()

        for segment in unique_segments:
            segment_data = plot_df[plot_df[segments] == segment][[Xi, Xj]].values

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
                    label=f"{segments} = {segment}" if segment not in plotted_segments else None 
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
    data: pd.DataFrame,
    features: list[str],
    segments: str,
    confidence_level: float = 0.95
):
    """
    Creates a customized lower-triangle pairplot with confidence ellipses representing segment boundaries.

    Args:
        data (pd.DataFrame): Dataset containing features and segment labels.
        features (list[str]): List of feature names to include in the pairplot.
        segments (str): Column name representing segment or cluster labels.
        confidence_level (float, optional): Confidence level for the ellipses. Defaults to 0.95.

    Returns:
        None
    """
    num_features = len(features)
    fig, axes = plt.subplots(num_features, num_features, figsize=(num_features * 3, num_features * 3))
    legend_handles = None

    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i, j]
            if i < j or i == j:
                ax.set_visible(False)
            else:
                Xi, Xj = features[j], features[i]
                try:
                    plot_segment_ellipse_ax(data, Xi, Xj, segments, ax=ax, confidence_level=confidence_level)

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