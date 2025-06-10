import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from post_analysis_clustering.utils import timer,get_palette
from post_analysis_clustering.visualize.base import BaseVis

class ProfileData(BaseVis):
    def __init__(self, 
                 df, 
                 features, 
                 target_cluster, 
                 primary_key):
        super().__init__(df, 
                         features, 
                         target_cluster, 
                         primary_key)
        
        self._validate_dtypes()

    def _validate_dtypes(self):
        invalid_types = [f for f in self.features if not pd.api.types.is_numeric_dtype(self.df[f]) and not pd.api.types.is_categorical_dtype(self.df[f])]
        if invalid_types:
            raise ValueError(f"The following features are neither numeric nor categorical: {invalid_types}")

    def _prep_dist(self):
        prep_df = self.df.loc[:, [self.primary_key] + self.features + [self.target_cluster]].copy()
        segment_dfs = {
            segment: prep_df[prep_df[self.target_cluster] == segment].drop(columns=self.target_cluster)
            for segment in sorted(prep_df[self.target_cluster].unique())
        }
        all_df = prep_df.drop(columns=self.target_cluster)
        return all_df, segment_dfs

    def _prep_frequency_feature(self, 
                               data: pd.DataFrame, 
                               col: str, 
                               binning_keywords: list[str] = None, 
                               n_bins: int = 100) -> pd.DataFrame:
        
        if binning_keywords is not None and (not isinstance(binning_keywords, list) or not all(isinstance(b, str) for b in binning_keywords)):
            raise TypeError("`binning_keywords` must be a list of strings or None.")
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("`n_bins` must be a positive integer.")
            
        proxy = data.copy()
        is_in_target_list = any(sub.lower() in col.lower() for sub in (binning_keywords or []))

        if is_in_target_list:
            try:
                zero_mask = proxy[col] == 0
                zero_part = proxy.loc[zero_mask].copy()
                nonzero_part = proxy.loc[~zero_mask].copy()
                bin_edges = np.round(np.linspace(nonzero_part[col].min(), nonzero_part[col].max(), n_bins + 1), 2)
                nonzero_part[col] = pd.cut(nonzero_part[col], bins=bin_edges, include_lowest=True)
                zero_part[col] = '= 0'
                proxy = pd.concat([zero_part, nonzero_part])
            except Exception as e:
                print(f"Could not bin {col}: {e}")
                return pd.DataFrame(columns=[col, 'count', 'percentage(%)'])

        proxy = proxy[col].value_counts(dropna=False).reset_index()
        proxy.columns = [col, 'count']
        proxy['percentage(%)'] = round((proxy['count'] / proxy['count'].sum()) * 100, 2)

        if is_in_target_list:
            def sort_key(val):
                if val == '= 0': return float('-inf')
                if isinstance(val, pd.Interval): return val.left
                return float('inf')
            proxy = proxy.sort_values(by=col, key=lambda x: x.map(sort_key))
            proxy[col] = proxy[col].apply(lambda x: str(x) if x == '= 0' else f'({x.left:.2f}, {x.right:.2f}]')
        else:
            proxy = proxy.sort_values(by=col)

        return proxy.reset_index(drop=True)

    def _plot_feature_distributions(self, 
                                   df_dict: dict[str, pd.DataFrame], 
                                   binning_keywords: list[str] = None, 
                                   n_bins: int = 100):
        n = len(self.features)
        cols = 2
        rows = (n + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = axes.flatten()

        last_key = list(df_dict.keys())[-1]

        for key, data in df_dict.items():
            for i, col in enumerate(self.features):
                ax = axes[i]
                proxy = self._prep_frequency_feature(data, col, binning_keywords, n_bins)
                if proxy.empty:
                    continue
                sns.barplot(x=col, y='count', data=proxy, ax=ax, color='#1f77b4')
                ax.set_title(f"{col} ({key})", fontsize=10)
                ax.set_ylabel('Count')
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=90)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'{last_key} : Distribution of features', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def PlotAllFeatureDist(self, binning_keywords: list[str] = None, n_bins: int = 100):
        all_df, segment_dfs = self._prep_dist()
        df_dict_all = {'all segment': all_df}
        # Create dict for each segment, keys as strings for readability
        df_dict_segments = {f'segment {seg}': df for seg, df in segment_dfs.items()}

        # Plot all segments individually
        for segment_name, segment_df in df_dict_segments.items():
            self._plot_feature_distributions({segment_name: segment_df}, binning_keywords, n_bins)

        # Plot all data combined
        self._plot_feature_distributions(df_dict_all, binning_keywords, n_bins)
