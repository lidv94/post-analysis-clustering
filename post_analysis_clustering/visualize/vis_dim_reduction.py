import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from post_analysis_clustering.utils import timer, get_palette

from typing import List

class DimensionalityReduction:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def validate_dtypes(self, df: pd.DataFrame, features: List[str]) -> None:
        pass

    def method1(self, df: pd.DataFrame, features: List[str]) -> None:
        return "method1"

    def method2(self, df: pd.DataFrame, features: List[str]) -> None:
        return "method2"

    def method3(self, df: pd.DataFrame, features: List[str]) -> None:
        return "method3"

    def method4(self, df: pd.DataFrame, features: List[str]) -> None:
        return "method4"