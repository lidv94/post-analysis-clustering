from .lean_feature.model_creation import ModelCreation
from .lean_feature.base import BaseLean


class LeanImportanceRank(BaseLean):
    def __init__(self, df):
        super.__init__(df=df)
        
    def run(self, df1, df2, model_creation:ModelCreation):
        ...
        
    def plot(self):
        self._validate_inputs()
        
# class LeanImportanceThreshold:
#     def __init__(self):
#         ...
        
#     def run(self, param1, param2, param3:ModelCreation):
#         if param3.result is None:
#             param3.run()
#         param3.result
        
#     def plot(self):
#         ...
        
# class LeanChiSquare:
#     def __init__(self):
#         ...
        
#     def run(self):
#         ...
        
#     def _plot(self):
#         ...
        
#     def plot(self):
#         ...