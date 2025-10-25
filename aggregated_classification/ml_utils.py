# ml_utils.py

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Literal

# Scikit-learn imports
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Imbalanced-learn imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Define a type for our method choice for better code hinting and clarity
MethodType = Literal['rfe', 'pca_live', 'pca_precomputed', 'select_k_best']

def load_and_preprocess_data(
    method: MethodType,
    raw_file_path: Path,
    pca_file_path: Path,
    n_pca_features_to_use: int = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and preprocesses data based on the selected feature reduction method.
    - For 'rfe' or 'pca_live', it loads the raw data and performs full preprocessing.
    - For 'pca_precomputed', it loads the already transformed PCA data.
    Handles both full datasets and small dummy data cases.

    Args:
        method (MethodType): The chosen feature reduction method.
        raw_file_path (Path): Path to the raw, high-dimensional data file.
        pca_file_path (Path): Path to the precomputed PCA data file.
        n_pca_features_to_use (int, optional): Number of PCA features to use if method is 'pca_precomputed'.

    Returns:
        A tuple containing the feature DataFrame (X) and the target Series (y).
    """
    print(f"Loading data for method: '{method}'")
    # --- Case 1: Raw data methods ---
    # For methods that need raw data, preprocess it
    if method in ['rfe', 'pca_live', 'select_k_best']:
        print(f"Loading raw data from '{raw_file_path}'...")
        df = pd.read_csv(pca_file_path, sep="\t")

        print("Preprocessing raw data...")
        y = df['is.expert']
        X = df.drop(columns=['Participant_unique', 'is.expert'])

        # One-hot encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)

        # Drop constant features (no variance)
        constant_features = [col for col in X_encoded.columns if X_encoded[col].nunique() == 1]
        if constant_features:
            X_final = X_encoded.drop(columns=constant_features)
            print(f"Removed {len(constant_features)} constant features.")
        else:
            X_final = X_encoded

    # --- Case 2: Precomputed PCA data ---
    elif method == 'pca_precomputed':
        print(f"Loading pre-computed PCA data from '{pca_file_path}'...")
        df = pd.read_csv(pca_file_path, sep="\t")

        y = df['is.expert']
        X_final = df.drop(columns=['Participant_unique', 'is.expert'])

        # Adjust to requested number of PCA features (safe fallback)
        if n_pca_features_to_use is not None:
            available = X_final.shape[1]
            if n_pca_features_to_use > available:
                print(f"Requested {n_pca_features_to_use} PCA features, but only {available} available.")
                print(f"Falling back to {available} components for small/dummy dataset.")
                n_pca_features_to_use = available

            print(f"Selecting the first {n_pca_features_to_use} principal components.")
            X_final = X_final.iloc[:, :n_pca_features_to_use]

        # --- Dummy data fallback: if only a few samples exist ---
        if X_final.shape[0] < 10:
            print(f"Warning: Only {X_final.shape[0]} samples detected. Using diagnostic fallback mode.")
            print("This dataset is too small for cross-validation; metrics will be illustrative only.")


    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Final shape of feature matrix X: {X_final.shape}")
    return X_final, y


def build_pipeline(method: MethodType, classifier: Any) -> ImbPipeline:
    """Builds the machine learning pipeline based on the selected method."""
    steps = [('scaler', StandardScaler())]

    if method == 'rfe':
        estimator = DecisionTreeClassifier(random_state=42)
        steps.append(('feature_selection', RFE(estimator=estimator, step=10)))


    elif method == 'select_k_best':
        steps.append(('variance_threshold', VarianceThreshold()))
        steps.append(('feature_selection', SelectKBest(score_func=f_classif)))

    elif method == 'pca_live':
        steps.append(('pca', PCA()))

    # For 'pca_precomputed', we add no feature reduction step

    steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('classifier', classifier))

    return ImbPipeline(steps)

def evaluate_final_pipeline_with_loocv(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    method: str,
    classifier: any,
    best_params: dict,
    model_name: str,
    average: Literal['macro', 'weighted'] = 'macro'
) -> Dict[str, Any]:
    """
    Evaluates a final pipeline using LOOCV and calculates overall performance
    metrics with a specified averaging method.
    Automatically switches to a simple diagnostic fit/predict mode for small datasets.

    Args:
        X_full: The complete feature dataset.
        y_full: The complete target series.
        method: The modeling method used.
        classifier: The classifier instance.
        best_params: The dictionary of best hyperparameters for the pipeline.
        model_name: The name of the model being evaluated.
        average (str): The averaging method for precision, recall, and F1.
                       Options: 'macro', 'weighted'. Defaults to 'macro'.

    Returns:
        A dictionary containing the overall performance metrics and LOOCV details.
    """
    print(f"\n--- Running Final LOOCV Evaluation for: {model_name} ---")
    print(f"Using Method: '{method}' | Average: '{average}'")
    print(f"With Parameters: {best_params}")

    pipeline = build_pipeline(method, classifier) # Assumes your build_pipeline is defined

    # Filter params for the current pipeline
    existing_step_names = [step[0] for step in pipeline.steps]
    params_to_set = {
        key: value for key, value in best_params.items()
        if key.split('__')[0] in existing_step_names
    }
    pipeline.set_params(**params_to_set)

    n_samples = len(X_full)

    # --- Fallback for tiny datasets ---
    if n_samples < 10:
        print(f"Warning: Only {n_samples} samples available. Skipping LOOCV and running single fit/predict.")
        pipeline.fit(X_full, y_full)
        y_predictions = pipeline.predict(X_full)
    else:
        loo = LeaveOneOut()
        y_predictions = cross_val_predict(pipeline, X_full, y_full, cv=loo, n_jobs=-1)

    print(f"\nEvaluation complete. Calculating overall metrics from {n_samples} predictions.")


    # --- Compute metrics ---
    # Calculate overall metrics using the specified average method
    accuracy = accuracy_score(y_full, y_predictions)
    precision = precision_score(y_full, y_predictions, average=average, zero_division=0)
    recall = recall_score(y_full, y_predictions, average=average, zero_division=0)
    f1 = f1_score(y_full, y_predictions, average=average, zero_division=0)

    print(f"Overall Accuracy:             {accuracy:.4f}")
    print(f"Overall Precision ({average.capitalize()}): {precision:.4f}")
    print(f"Overall Recall ({average.capitalize()}):    {recall:.4f}")
    print(f"Overall F1-Score ({average.capitalize()}):  {f1:.4f}")
    print("-" * 60)

    # Return a dictionary with keys that reflect the averaging method
    return {
        "model_name": model_name,
        "method": method,
        "average_method": average,
        "best_params": params_to_set,
        "loocv_accuracy": accuracy,
        f"loocv_precision_{average}": precision,
        f"loocv_recall_{average}": recall,
        f"loocv_f1_{average}": f1,
        "loocv_predictions": y_predictions.tolist()
    }
    print(f"\nLOOCV results saved to: {file_path}")