# post-analysis-clustering

> A Python package for visualizing and interpreting clustering results using statistical tests, feature importance, and insightful plots.

[![PyPI version](https://img.shields.io/pypi/v/post-analysis-clustering)](https://pypi.org/project/post-analysis-clustering/)
[![License](https://img.shields.io/github/license/lidv94/post-analysis-clustering)](https://github.com/lidv94/post-analysis-clustering/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/post-analysis-clustering)](https://pypi.org/project/post-analysis-clustering/)
![GitHub last commit](https://img.shields.io/github/last-commit/lidv94/post-analysis-clustering)

---

## 📦 Installation

Install via pip:

```bash
pip install post-analysis-clustering
```

# 📦 post-analysis-clustering

## 📊 Overview

The `post-analysis-clustering` package is designed to help **analyze, validate, and interpret clustering results**. It provides tools to:

- Visualize feature distributions across clusters  
- Identify distinguishing features using statistical tests  
- Plot heatmaps, snake plots, importance charts, and more  
- Evaluate inter-cluster separation and overlap  

It is especially useful for clustering results from customer segmentation, fraud detection, or other unsupervised learning pipelines.

---

## 🔧 Features

- 📈 **Box, Violin, and Distribution Plots** for feature-by-cluster analysis  
- 🧮 **Permutation Importance Heatmaps** across multiple classifiers  
- 📊 **Crosstab and Binned Heatmaps** to explore categorical and continuous variables   
- ✅ **Chi-square tests** with human-readable significance interpretation  
- 🎨 Custom color palettes for consistent cluster visualization  

---

## 🚀 Usage

Basic usage example:

```python
from post_analysis_clustering import plot_bin_heatmap

plot_bin_heatmap(
    raw_df=df,
    features=["age", "income", "purchase_amount"],
    target_cluster="cluster",
    annot_type="Percentage"
)
```
For a complete usage example, check out the [dev.ipynb](https://github.com/lidv94/post-analysis-clustering/blob/main/dev.ipynb).

---

## 👤 Author
- 🔗 [GitHub Profile](https://github.com/lidv94)  
- 💼 [LinkedIn](https://www.linkedin.com/in/alice-varakamin/)  
