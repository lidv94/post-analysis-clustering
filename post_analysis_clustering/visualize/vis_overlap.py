import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import chi2
from post_analysis_clustering.utils import timer, get_palette

from typing import List

class OverlapPairPlot:
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