import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap
from post_analysis_clustering.utils import timer

@timer
def calculate_permutation_importance(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    features: list[str], 
    n_repeats: int = 10
    ):
    """
    Calculate Permutation Importance for a given model.

    Args:
        model: The model to evaluate.
        X_test (pd.DataFrame): Test data features.
        y_test (pd.Series): True labels for the test data.
        features (List[str]): List of feature names for the model.
        n_repeats (int, optional): Number of times to shuffle each feature. Default is 10.

    Returns:
        pd.DataFrame: DataFrame containing features and their corresponding importance.
    """
    if not isinstance(X_test, (pd.DataFrame)):
        raise TypeError(f"X_test must be a DataFrame, got {type(X_test)}")
    if not isinstance(y_test, (pd.Series)):
        raise TypeError(f"y_test must be a Series, got {type(y_test)}")
        
    n_features = X_test.shape[1]
    if len(features) != n_features:
        raise ValueError(f"Length of features list ({len(features)}) does not match number of features in X_test ({n_features}).")    
    try:    
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": perm_importance.importances_mean
        }).sort_values(by="Importance", ascending=False)

        return importance_df
    except Exception as e:
        raise RuntimeError(f"Failed to compute permutation importance: {e}")


@timer
def train_and_evaluate_model(
    model, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    features: list[str]):
    """
    Train a given model, evaluate accuracy, and compute permutation importance.

    Args:
        model: The model to train and evaluate.
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data labels.
        X_test (pd.DataFrame): Test data features.
        y_test (pd.Series): Test data labels.
        features (List[str]): List of feature names for permutation importance.

    Returns:
        Tuple[pd.DataFrame, float, float, Dict[str, Union[str, Dict]]]:
            - pd.DataFrame: Permutation importance for the model.
            - float: Training accuracy.
            - float: Test accuracy.
            - Dict: Classification report for the model.
    """
    
    try:            
        # Validate input types
        for name, arr in [("X_train", X_train), ("X_test", X_test)]:
            if not isinstance(arr, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame, got {type(arr)}")
        for name, arr in [("y_train", y_train), ("y_test", y_test)]:
            if not isinstance(arr, pd.Series):
                raise TypeError(f"{name} must be a pandas Series, got {type(arr)}")

        # Feature name handling
        if not features:
            features = list(X_test.columns)
        elif len(features) != X_test.shape[1]:
            raise ValueError("Provided feature names do not match number of columns in X_test")

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Compute accuracy
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        # Compute classification report
        class_report = classification_report(y_test, test_pred, output_dict=True)

        # Compute permutation importance
        perm_importance_df = calculate_permutation_importance(model, X_test, y_test, features, n_repeats=10)

        return perm_importance_df, train_acc, test_acc, class_report
    
    except Exception as e:
        raise RuntimeError(f"Error in train_and_evaluate_model: {e}")



@timer
def train_and_evaluate_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    features: list[str]):
    """
    Train multiple models, evaluate performance, and calculate permutation importance.

    Args:
        X_train (pd.DataFrame): Training data features.
        X_test (pd.DataFrame): Test data features.
        y_train (pd.Series): Training data labels.
        y_test (pd.Series): Test data labels.
        features (List[str]): List of feature names.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]], Dict[str, Dict[str, Union[str, Dict]]]]:
            - Dict of model names as keys and their corresponding permutation importance DataFrames.
            - Dict of model names as keys and their corresponding accuracy scores (train and test).
            - Dict of model names as keys and their classification reports.
    """    
    
    try:
        # Validate input types
        for name, arr in [("X_train", X_train), ("X_test", X_test)]:
            if not isinstance(arr, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame, got {type(arr)}")
        for name, arr in [("y_train", y_train), ("y_test", y_test)]:
            if not isinstance(arr, pd.Series):
                raise TypeError(f"{name} must be a pandas Series, got {type(arr)}")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("`features` must be a list of strings")    
    
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            # "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42), 
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, early_stopping=True),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=42),
            "Naive Bayes": GaussianNB()
        }

        importance_results = {}
        performance = {}
        classification_reports = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            perm_importance_df, train_acc, test_acc, class_report = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, features)

            importance_results[name] = perm_importance_df
            performance[name] = {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
            classification_reports[name] = class_report

        return importance_results, performance, classification_reports
    
    except Exception as e:
        print(f"Error in train_and_evaluate_models : {e}")
        raise
        
###########################################################################################################
######################################## BINARY-CLASS #####################################################
###########################################################################################################

@timer
def prep_binary_class(df: pd.DataFrame, features: list[str], target_cluster: str):
    """
    Prepares binary classification labels for each cluster segment by converting the target cluster into binary columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the features and target cluster column.
        features (list[str]): List of feature column names.
        target_cluster (str): The name of the target cluster column containing segment labels.

    Returns:
        pd.DataFrame: A DataFrame with new binary columns representing each cluster.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame.")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("`features` must be a list of strings.")
        if target_cluster not in df.columns:
            raise ValueError(f"`{target_cluster}` not found in DataFrame columns.")
            
        binary_df = df.copy() 
        for cluster_label in sorted(df[target_cluster].unique()):
            binary_df[f'is_cluster_{cluster_label}'] = (df[target_cluster] == cluster_label).astype(int)
        return binary_df
    except Exception as e:
        print(f"Error in prep_binary_class : {e}")
        raise

@timer
def cal_imp_one_binary_class(df: pd.DataFrame, features: list[str], target_cluster: str, focus_segment: int):
    """
    Performs binary classification to evaluate feature importance for a specific cluster segment.

    This function preprocesses the data to perform a one-vs-all binary classification, where the 
    focus is on a specific segment of the target cluster. It then trains a model using the provided 
    features and evaluates the model's performance, returning the feature importance results.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and target cluster column.
        features (list[str]): List of feature column names used for classification.
        target_cluster (str): Name of the column containing cluster labels.
        focus_segment (int): The specific cluster segment to focus on for the classification task.

    Returns:
        pd.DataFrame: A DataFrame containing feature importance values for each trained model. 
                       The importance values are represented in a table with features as rows 
                       and models as columns.
    """
    try:
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"'df' must be a pandas DataFrame, got {type(df)}")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("'features' must be a list of strings")
        if target_cluster not in df.columns:
            raise ValueError(f"'{target_cluster}' is not a column in the DataFrame")
        if focus_segment not in df[target_cluster].unique():
            raise ValueError(f"focus_segment {focus_segment} not found in '{target_cluster}' values")

        binary_df = prep_binary_class(df=df, features=features, target_cluster=target_cluster)
        y = binary_df[f'is_cluster_{focus_segment}']
        X = binary_df[features]

        if X.isnull().any().any():
            raise ValueError("NaN values detected in feature data")
        if y.isnull().any():
            raise ValueError("NaN values detected in target data")        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        importance_results, performance, classification_reports = train_and_evaluate_models(X_train, X_test, y_train, y_test, features)

        print(f"\n### Feature Importance : One-vs-All Classification for Cluster {focus_segment} ###")
        # Extract all unique features
        all_features = set()
        for df in importance_results.values():
            all_features.update(df['Feature'])

        importance_df = pd.DataFrame(index=sorted(all_features))

        for model, df in importance_results.items():
            model_importance = df.set_index('Feature')['Importance']
            importance_df[model] = importance_df.index.map(model_importance)

        importance_df.reset_index(inplace=True)
        importance_df.rename(columns={'index': 'Feature'}, inplace=True)

        return importance_df # only one segment
    except Exception as e:
        print(f"Runtime error in cal_imp_one_binary_class : {e}")
        raise 

@timer
def prep_rank_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks features based on their importance scores across different models.

    This function melts the provided importance DataFrame, which contains feature importance values
    across multiple models, and ranks the features in ascending order of importance. The most important features
    receive the lowest rank (1), and the least important features receive the highest rank.

    Args:
        importance_df (pd.DataFrame): A DataFrame containing features and their corresponding importance scores
                                      for different models. The 'Feature' column contains feature names, and
                                      the other columns represent models with their respective importance values.

    Returns:
        pd.DataFrame: A DataFrame with features, their importance values, and their corresponding ranks across models.
    """
    try:
        if not isinstance(importance_df, pd.DataFrame):
            raise TypeError(f"'importance_df' must be a pandas DataFrame, got {type(importance_df)}")
        if "Feature" not in importance_df.columns:
            raise ValueError("'importance_df' must contain a 'Feature' column")
        value_columns = importance_df.drop("Feature", axis=1).columns.tolist()
        if not value_columns:
            raise ValueError("No model columns found to rank")
    
        melt_df = pd.melt(importance_df, id_vars="Feature", value_vars=importance_df.drop("Feature", axis=1).columns.tolist())  
        rank_df = melt_df.groupby(['variable'])['value'].rank(method="dense", ascending=False)
        melt_df['rank'] = rank_df.astype(int)  
        return melt_df
    
    except Exception as e:
        print(f"Runtime error in prep_rank_importance : {e}")
        raise

@timer
def pivot_rank_importance(importance_df: pd.DataFrame, n_rank: int):
    """
    Creates a pivot table summarizing the ranks of features across models.

    This function takes the ranked importance values for each feature and generates a pivot table that displays
    the frequency of each rank for the features. It then limits the table to the top 'n_rank' ranks.

    Args:
        importance_df (pd.DataFrame): A DataFrame containing feature importance values and their corresponding ranks.
                                      This should include a 'Feature' column and one column for each model's rank.
        n_rank (int): The number of top ranks to retain in the pivot table. Only ranks from 1 to 'n_rank' are included.

    Returns:
        pd.DataFrame: A pivot table with features as rows, ranks as columns, and the count of models achieving each rank.
    """
    try:
        if not isinstance(importance_df, pd.DataFrame):
            raise TypeError(f"'importance_df' must be a pandas DataFrame, got {type(importance_df)}")
        if not isinstance(n_rank, int) or n_rank <= 0:
            raise ValueError("'n_rank' must be a positive integer")
            
        melt_df = prep_rank_importance(importance_df=importance_df)
        
        if "rank" not in melt_df.columns or "Feature" not in melt_df.columns or "variable" not in melt_df.columns:
            raise ValueError("Expected columns missing after melting and ranking")

        pvt_imp = pd.pivot_table(melt_df,
                                 index='Feature',
                                 columns='rank', # rank_abs
                                 values='variable',
                                 fill_value=0,
                                 aggfunc='count')
        top_n_pvt_imp = pvt_imp.loc[:, :n_rank]
        return top_n_pvt_imp

    except Exception as e:
        print(f"Runtime error in pivot_rank_importance : {e}")
        raise

######################### DONE-ONE-SEGMENT ################################
###################### START-COMBINE-ALL-SEGMENT ###########################

@timer
def cal_imp_all_binary_class(df: pd.DataFrame, features: list[str], target_cluster: str, n_rank: int = 5):
    """
    Computes feature importance for all unique segments in the target cluster column using a binary classification approach.

    This function processes each unique cluster (segment) in the target column, performs a one-vs-all binary classification
    for each segment, and calculates the feature importance. The results are combined into two DataFrames:
    - One contains raw feature importance scores for all segments.
    - The other contains a pivot table summarizing the top-n ranked features per segment.

    Args:
        df (pd.DataFrame): The input DataFrame containing feature columns and a target column representing cluster or segment labels.
        features (List[str]): A list of feature column names to evaluate for importance.
        target_cluster (str): The column name that contains the cluster or segment labels.
        n_rank (int): The number of top-ranked features to retain per segment. Defaults to 5.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        - A DataFrame containing the raw feature importance scores for each segment.
        - A DataFrame containing the pivot table of top-n ranked features for each segment, with counts of model ranks.
    """    
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame.")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("`features` must be a list of strings.")
        if not isinstance(target_cluster, str):
            raise TypeError("`target_cluster` must be a string.")
        if target_cluster not in df.columns:
            raise ValueError(f"`target_cluster` '{target_cluster}' not found in DataFrame columns.")
        if not isinstance(n_rank, int) or n_rank < 1:
            raise ValueError("`n_rank` must be a positive integer.")
    
        unique_segments = sorted(df[target_cluster].unique(), reverse=False)
        all_imps = []
        all_pvt_imps_score = []

        for segment in unique_segments:
            print(f"Processing segment {segment}")

            # Compute feature importance for the given segment
            importance_df = cal_imp_one_binary_class(df=df, features=features, target_cluster=target_cluster, focus_segment=segment)

            # Get top N important features
            pvt_imp = pivot_rank_importance(importance_df, n_rank=n_rank)

            # Add segment identifier
            importance_df["Segment"] = segment
            all_imps.append(importance_df)

            # Add segment identifier
            pvt_imp["Segment"] = segment
            all_pvt_imps_score.append(pvt_imp)

        # Combine results
        final_imp = pd.concat(all_imps, axis=0).reset_index(drop=True)
        final_pvt_imp_score = pd.concat(all_pvt_imps_score, axis=0).reset_index()
        final_pvt_imp_score.columns.name = None

        # Sort columns: Feature, 1, 2, ..., Segment
        fixed_cols = ['Feature']
        numeric_cols = sorted([col for col in final_pvt_imp_score.columns if isinstance(col, int)])
        fixed_cols += numeric_cols + ['Segment']
        final_pvt_imp_score = final_pvt_imp_score[fixed_cols]

        # Fill NaNs with 0 and convert numeric columns to integers
        final_pvt_imp_score = final_pvt_imp_score.fillna(0)
        cols_to_convert = [col for col in final_pvt_imp_score.columns if col not in ['Feature', 'Segment']]
        final_pvt_imp_score[cols_to_convert] = final_pvt_imp_score[cols_to_convert].astype(int)

        return final_imp, final_pvt_imp_score
    
    except Exception as e:
        print(f"Error in cal_imp_all_binary_class : {e}")
        raise

#############################################################        

@timer
def plot_heatmap_imp_all_binary_class(final_imp: pd.DataFrame, compare_type: str = 'Normalized'):
    """
    Plots a heatmap of feature importance scores for multiple models, segmented by cluster.

    This function visualizes the feature importance scores for different segments (clusters) and models using a heatmap.
    The color scale and the values displayed can be customized based on the `compare_type` argument. It supports three modes:
    - 'Normalized': Scales the values per model (column-wise) within each segment, so the color scale is relative to each model.
    - 'Global': Uses raw values, with a consistent color scale across all models.
    - 'Percentage': Displays the values as percentages of the total for each model within a segment.

    Args:
        final_imp (pd.DataFrame): A DataFrame containing feature importance scores with columns representing features and rows representing models and segments.
        compare_type (str, optional): Specifies how to scale the values in the heatmap:
                                      - 'Normalized' (default): Normalize values per model within each segment.
                                      - 'Global': Use raw values across all models.
                                      - 'Percentage': Show values as a percentage of the total for each model.
    
    Returns:
        None: The function generates a heatmap plot for each segment in the data.
    """
    
    try:
        if not isinstance(final_imp, pd.DataFrame):
            raise TypeError("`final_imp` must be a pandas DataFrame.")
        if compare_type not in ['Global', 'Percentage', 'Normalized']:
            raise ValueError("`compare_type` must be one of ['Global', 'Percentage', 'Normalized'].")

        df = final_imp.copy()
        if "Segment" not in df.columns or "Feature" not in df.columns:
            raise ValueError("`final_imp` must contain 'Segment' and 'Feature' columns.")

        unique_segments = sorted(df["Segment"].unique(), reverse=False)

        for segment in unique_segments:
            print(f"Plotting heatmap for segment {segment}")
            segment_data = df[df['Segment'] == segment]
            segment_data = segment_data.drop(columns='Segment').set_index("Feature")
            rank_data = segment_data.rank(ascending=False, axis=0).astype(int)

            # Compute data for selected compare_type
            if compare_type == 'Global':
                show_data = segment_data
            elif compare_type == 'Percentage':
                show_data = segment_data.div(segment_data.sum(axis=0), axis=1) * 100
            else:  # Default: 'Normalized'
                show_data = (segment_data - segment_data.min()) / (segment_data.max() - segment_data.min())

            # Plot heatmap
            plt.figure(figsize=(6, 5))
            sns.heatmap(show_data, 
                        annot=rank_data, 
                        fmt='d' if compare_type != 'Percentage' else '.0f', 
                        cmap=sns.light_palette("seagreen", as_cmap=True),
                        cbar=True,
                        linewidths=0.5,
                       )

            # Add title and labels
            plt.title(f"{compare_type} Heatmap of Permutation Importance from Models for Cluster {segment} with Rank Annotation",
                      fontsize=10, fontweight='bold')
            plt.xlabel("Classification Models", fontsize=12, fontweight='bold')  
            plt.ylabel("Features", fontsize=12, fontweight='bold')
            plt.show()
    except Exception as e:
        print(f"Error in plot_heatmap_imp_all_binary_class : {e}")
        raise

##############################################################

@timer
def plot_vote_result_all_binary_class(final_pvt_imp_score: pd.DataFrame) -> None:
    """
    Plots heatmaps for each segment using precomputed feature importance scores, with colors representing ranked importance.

    This function visualizes the voting results of feature importance from multiple models using heatmaps. The heatmaps
    display the importance scores for each feature within each segment (cluster), with a custom color palette that 
    reflects the rank of the importance scores.

    The color palette consists of distinct colors that map to ranks (from low to high). This allows for a visual
    comparison of how features are ranked within each segment, based on their importance scores.

    Parameters:
    - final_pvt_imp_score (DataFrame): A DataFrame containing precomputed feature importance scores. Each row represents
      a feature, and the columns represent the scores for each segment and model. The DataFrame must include a column 
      "Segment" to identify the segments (clusters).

    Returns:
    - None: The function generates and displays a heatmap for each segment in the data, with color-coded rankings.
    """
    
    try:
        if not isinstance(final_pvt_imp_score, pd.DataFrame):
            raise TypeError("`final_pvt_imp_score` must be a pandas DataFrame.")
        if "Segment" not in final_pvt_imp_score.columns or "Feature" not in final_pvt_imp_score.columns:
            raise ValueError("`final_pvt_imp_score` must contain 'Segment' and 'Feature' columns.")
    
        unique_segments = sorted(final_pvt_imp_score["Segment"].unique(), reverse=False)

        custom_order_palette = {
            0: "#ffffff",  # 0 - white
            1: "#e7f0f9",  # 1 - very lighter blue
            2: "#d0e4f7",  # 2 - very light blue
            3: "#a6c8ec",  # 3 - light blue
            4: "#7badde",  # 4 - medium light blue
            5: "#5192ce",  # 5 - medium blue
            6: "#2a77be",  # 6 - deep blue
            7: "#1c5f9f",  # 7 - dark blue
            8: "#144b85",  # 8 - very dark blue
            9: "#103c6c",  # 9 - almost black blue
            10: "#0e2f52", # 10 - deep navy
            11: "#0b2340", # 11 - dark navy
            12: "#081f30", # 12 - midnight blue
            13: "#041624", # 13 - blackened blue
            14: "#02101a", # 14 - charcoal blue
            15: "#01080f"  # 15 - almost black
        }

        # Extracting the colors as a list
        custom_colors = [custom_order_palette[i] for i in sorted(custom_order_palette.keys())]
        custom_cmap = ListedColormap(custom_colors)

        for segment in unique_segments:
            print(f"Plotting heatmap for segment {segment}")

            # Extract data for the current segment
            segment_data = final_pvt_imp_score[final_pvt_imp_score["Segment"] == segment].drop(columns=["Segment"]).set_index("Feature")

            # Plot heatmap
            plt.figure(figsize=(6, 5))
            sns.heatmap(segment_data, 
                        annot=True, 
                        cmap=custom_cmap, 
                        cbar=False,
                        # center=0, 
                        linewidths=0.5,
                        # linecolor='grey',
                        vmin=0, vmax=15) 

            # Add title and labels
            plt.title(f"Voting Results of Permutation Importance from Models for Cluster {segment} ", fontsize=10, fontweight='bold')
            plt.xlabel("Rank", fontsize=12, fontweight='bold')  
            plt.ylabel("Feature", fontsize=12, fontweight='bold')
            plt.show()
    except Exception as e:
        print(f"Error in plot_vote_result_all_binary_class : {e}")
        raise

        
#################################################################

@timer
def filter_thres_features(final_pvt_imp_score: pd.DataFrame, thres_score: int):
    """
    Filters features based on a threshold score for importance and returns the remaining features and their scaled versions.

    This function processes the precomputed feature importance scores from multiple models and selects features that 
    have an aggregated importance score greater than or equal to a specified threshold. It also creates a list of 
    scaled versions of the selected features (by adding a prefix 'SCALED_') for further processing.

    Parameters:
    - final_pvt_imp_score (DataFrame): A DataFrame containing feature importance scores for all segments and models.
      It must include a 'Feature' column and columns representing importance scores for each model.
    - thres_score (int): The threshold score for selecting features based on the sum of their importance across models.

    Returns:
    - tuple: A tuple containing:
        - lean_feature_list (list): A list of features with importance greater than or equal to the threshold.
        - scaled_lean_feature_list (list): A list of scaled versions of the selected features, prefixed with 'SCALED_'.

    Example:
    - lean_feature_list, scaled_lean_feature_list = get_thres_features(final_pvt_imp_score, 3)
    """
    try:
        if not isinstance(final_pvt_imp_score, pd.DataFrame):
            raise TypeError("`final_pvt_imp_score` must be a pandas DataFrame.")
        if not isinstance(thres_score, int) or thres_score < 0:
            raise ValueError("`thres_score` must be a non-negative integer.")    

        df = final_pvt_imp_score.copy()
        df['sum_top_model'] = df.iloc[:, 1:-1].sum(axis=1)
        df = df.loc[df['sum_top_model']>=thres_score,:]
        lean_feature_list = list(set(df['Feature'].to_list()))

        print(f"Total features from raw : {len(set(final_pvt_imp_score['Feature'].to_list()))}")
        print(f"Total features remaining after lean from vote results: {len(lean_feature_list)}")

        scaled_lean_feature_list = [f"SCALED_{col}" for col in lean_feature_list]

        return lean_feature_list, scaled_lean_feature_list
    except Exception as e :
        print(f"[Error in filter_thres_features]: {e}")
        raise
        
@timer
def filter_thres_features_by_cluster(final_pvt_imp_score: pd.DataFrame, thres_score: int):
    """
    Filters features by cluster and a threshold score for importance, and returns the remaining features for each cluster.

    This function processes the precomputed feature importance scores for each cluster, selecting features that 
    have an aggregated importance score greater than or equal to a specified threshold. It also creates a list of 
    scaled versions of the selected features for each cluster and provides a union of all features across clusters.

    Parameters:
    - final_pvt_imp_score (DataFrame): A DataFrame containing feature importance scores for all segments and models.
      It must include a 'Segment' column to indicate the cluster and a 'Feature' column for feature names.
    - thres_score (int): The threshold score for selecting features based on the sum of their importance across models.

    Returns:
    - tuple: A tuple containing:
        - cluster_lean_features_dict (dict): A dictionary where each key is a cluster segment, and the value is another
          dictionary with:
            - 'lean_feature_list' (list): Features with importance greater than or equal to the threshold for the cluster.
            - 'scaled_lean_feature_list' (list): Scaled versions of the selected features for the cluster.
        - union_lean_feature_list (list): The union of all selected features across clusters.
        - union_scaled_lean_feature_list (list): Scaled versions of the union of all selected features across clusters.

    Example:
    - cluster_lean_features_dict, union_lean_feature_list, union_scaled_lean_feature_list = get_thres_features_by_cluster(final_pvt_imp_score, 3)
    """
    try:
        if not isinstance(final_pvt_imp_score, pd.DataFrame):
            raise TypeError("`final_pvt_imp_score` must be a pandas DataFrame.")
        if not isinstance(thres_score, int) or thres_score < 0:
            raise ValueError("`thres_score` must be a non-negative integer.")
        if 'Segment' not in final_pvt_imp_score.columns or 'Feature' not in final_pvt_imp_score.columns:
            raise ValueError("`final_pvt_imp_score` must contain 'Segment' and 'Feature' columns.")    
    
        df = final_pvt_imp_score.copy()
        unique_segments = sorted(df['Segment'].unique(), reverse=False)
        cluster_lean_features_dict = {}
        union_lean_feature_set = set()

        for cluster in unique_segments:
            df_cluster = df[df['Segment'] == cluster].copy()
            df_cluster['sum_top_model'] = df_cluster.iloc[:, 1:-1].sum(axis=1)
            df_cluster = df_cluster.loc[df_cluster['sum_top_model'] >= thres_score, :]
            lean_feature_list = sorted(df_cluster['Feature'].to_list())
            scaled_lean_feature_list = sorted([f"SCALED_{col}" for col in lean_feature_list])

            union_lean_feature_set.update(lean_feature_list)

            cluster_lean_features_dict[cluster] = {
                "lean_feature_list": lean_feature_list,
                "scaled_lean_feature_list": scaled_lean_feature_list
            }

            print(f"Cluster {cluster}:")
            print(f"  Total features from raw: {len(set(final_pvt_imp_score['Feature'].to_list()))}")
            print(f"  Total features remaining after lean from vote results: {len(lean_feature_list)}")

        # Convert union set to list and print
        union_lean_feature_list = sorted(list(union_lean_feature_set))
        union_scaled_lean_feature_list = sorted([f"SCALED_{col}" for col in union_lean_feature_list])

        print(f"\nUnion across all clusters:")
        print(f"  Total union features: {len(union_lean_feature_list)}")

        return cluster_lean_features_dict, union_lean_feature_list, union_scaled_lean_feature_list
    except Exception as e:
        print(f"Error in filter_thres_features_by_cluster : {e}")
        
###########################################################################################################
######################################## MULTI-CLASS ######################################################
###########################################################################################################

@timer
def cal_imp_multi_class(df: pd.DataFrame, features: list[str], target_cluster: str, n_rank: int = 5):
    """
    Calculates feature importance for a multi-class classification problem and returns a DataFrame with the importance scores.

    This function trains and evaluates classification models for a specified target cluster using the given features 
    and then computes the feature importance using permutation importance. It aggregates the results into a DataFrame 
    where each column represents a model's importance scores for each feature.

    Parameters:
    - df (DataFrame): A DataFrame containing the features and the target cluster column. 
      The target cluster column specifies the segment or class to predict.
    - features (list[str]): A list of feature column names to use for classification.
    - target_cluster (str): The column name representing the target cluster or class to predict.

    Returns:
    - DataFrame: A DataFrame containing the feature importance scores for each model. The rows represent features,
      and the columns represent different classification models. The values indicate the importance score of each feature 
      as computed by the model.

    Example:
    - importance_df = cal_imp_multi_class(df, features, target_cluster)
    """
    try:
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise TypeError("features must be a list of strings.")
        if not isinstance(target_cluster, str):
            raise TypeError("target_cluster must be a string.")
        if target_cluster not in df.columns:
            raise ValueError(f"'{target_cluster}' is not a column in the provided DataFrame.")
        all_df = df.copy()
        y = all_df[target_cluster]
        X = all_df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        importance_results, performance, classification_reports = train_and_evaluate_models(X_train, X_test, y_train, y_test, features)

        print(f"\n### Feature Importance : Multi-Classification for {target_cluster}")

        # Extract all unique features
        all_features = set()
        for df in importance_results.values():
            all_features.update(df['Feature'])

        # Create an empty dataframe with features as index
        importance_df = pd.DataFrame(index=sorted(all_features))

        # Fill in feature importance for each model
        for model, df in importance_results.items():
            model_importance = df.set_index('Feature')['Importance']
            importance_df[model] = importance_df.index.map(model_importance)

        importance_df.reset_index(inplace=True)
        importance_df.rename(columns={'index': 'Feature'}, inplace=True)

        final_imp = importance_df.reset_index(drop=True)
        final_pvt_imp_score = pivot_rank_importance(final_imp, n_rank=n_rank).reset_index()
        final_pvt_imp_score.columns.name = None
        return final_imp,final_pvt_imp_score
    except Exception as e:
        print(f"Error in cal_imp_multi_class : {e}")
        
#############################################################

@timer
def plot_heatmap_imp_multi_class(final_imp: pd.DataFrame, compare_type: str = 'Normalized'):
    """
    Plot a heatmap of feature importances for multi-class classification with rank annotations.

    Args:
        final_imp (pd.DataFrame): DataFrame containing feature importance scores across models.
            It must include a 'Feature' column and one column for each model's importance scores.
        compare_type (str, optional): How to scale the values for coloring. Options are:
            - 'Normalized': Normalize values per model (column-wise) between 0 and 1 (default).
            - 'Global': Use raw values without normalization.
            - 'Percentage': Show values as a percentage of each model's total.

    Returns:
        None: Displays a seaborn heatmap with feature ranks annotated.

    Example:
        plot_heatmap_imp_multi_class(final_imp, compare_type='Normalized')
    """
    try:
        if not isinstance(final_imp, pd.DataFrame):
            raise TypeError("final_imp must be a pandas DataFrame.")
        if 'Feature' not in final_imp.columns:
            raise ValueError("'final_imp' must contain a 'Feature' column.")
        segment_data = final_imp.copy().set_index("Feature")
        print(f"Plotting heatmap for multiclass")
        rank_data = segment_data.rank(ascending=False, axis=0).astype(int)

        # Compute data for selected compare_type
        if compare_type == 'Global':
            show_data = segment_data
        elif compare_type == 'Percentage':
            show_data = segment_data.div(segment_data.sum(axis=0), axis=1) * 100
        else:  # Default: 'Normalized'
            show_data = (segment_data - segment_data.min()) / (segment_data.max() - segment_data.min())

        # Plot heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(show_data, 
                    annot=rank_data, 
                    fmt='d' if compare_type != 'Percentage' else '.0f', 
                    cmap=sns.light_palette("seagreen", as_cmap=True),
                    cbar=True,
                    linewidths=0.5,
                   )

        # Add title and labels
        plt.title(f"{compare_type} Heatmap of Permutation Importance from Models for Multiclass with Rank Annotation",
                  fontsize=10, fontweight='bold')
        plt.xlabel("Classification Models", fontsize=12, fontweight='bold')  
        plt.ylabel("Features", fontsize=12, fontweight='bold')
        plt.show()
    except Exception as e:
        print(f"Error in plot_heatmap_imp_multi_class : {e}")

##############################################################

@timer
def plot_vote_result_multi_class(final_pvt_imp_score: pd.DataFrame):
    """
    Plot a voting heatmap showing feature ranks for multi-class classification models.

    Args:
        final_pvt_imp_score (pd.DataFrame): DataFrame where rows are features and columns represent 
            the number of times a feature achieved a certain rank across models (e.g., top 1, top 2, etc.).

    Returns:
        None: Displays a seaborn heatmap colored by rank votes.

    Example:
        plot_vote_result_multi_class(final_pvt_imp_score)
    """
    try:
        if not isinstance(final_pvt_imp_score, pd.DataFrame):
            raise TypeError("final_pvt_imp_score must be a pandas DataFrame.")
        if 'Feature' not in final_pvt_imp_score.columns:
            raise ValueError("final_pvt_imp_score must contain a 'Feature' column.")

        custom_order_palette = {
            0: "#ffffff",  # 0 - white
            1: "#e7f0f9",  # 1 - very lighter blue
            2: "#d0e4f7",  # 2 - very light blue
            3: "#a6c8ec",  # 3 - light blue
            4: "#7badde",  # 4 - medium light blue
            5: "#5192ce",  # 5 - medium blue
            6: "#2a77be",  # 6 - deep blue
            7: "#1c5f9f",  # 7 - dark blue
            8: "#144b85",  # 8 - very dark blue
            9: "#103c6c",  # 9 - almost black blue
            10: "#0e2f52", # 10 - deep navy
            11: "#0b2340", # 11 - dark navy
            12: "#081f30", # 12 - midnight blue
            13: "#041624", # 13 - blackened blue
            14: "#02101a", # 14 - charcoal blue
            15: "#01080f"  # 15 - almost black
        }

        # Extracting the colors as a list
        custom_colors = [custom_order_palette[i] for i in sorted(custom_order_palette.keys())]
        custom_cmap = ListedColormap(custom_colors)

        print(f"Plotting heatmap for multiclass")

        # Extract data for the current segment
        segment_data = final_pvt_imp_score.set_index("Feature")

        # Plot heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(segment_data, 
                    annot=True, 
                    cmap=custom_cmap, 
                    cbar=False,
                    # center=0, 
                    linewidths=0.5,
                    # linecolor='grey',
                    vmin=0, vmax=15)

        # Add title and labels
        plt.title(f"Voting Results of Permutation Importance from Models for Multiclass", fontsize=10, fontweight='bold')
        plt.xlabel("Rank", fontsize=12, fontweight='bold')  
        plt.ylabel("Feature", fontsize=12, fontweight='bold')
        plt.show()
    except Exception as e:
        print(f"Error in plot_vote_result_multi_class : {e}")

