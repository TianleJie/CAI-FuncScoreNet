import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.utils import shuffle
import shap


def load_complexity_features(excel_path: str) -> pd.DataFrame:
    """
    Load and label muscle synergy complexity features from two groups (CAI and healthy).
    """
    cai_df = pd.read_excel(excel_path, sheet_name='Sheet1')
    healthy_df = pd.read_excel(excel_path, sheet_name='Sheet2')

    cai_df['label'] = 1
    healthy_df['label'] = 0

    combined_df = pd.concat([cai_df, healthy_df], ignore_index=True)
    return combined_df


def load_aligned_probability_features(spatial_prob_path: str,
                                      spatial_label_path: str,
                                      temporal_prob_path: str,
                                      temporal_label_path: str) -> pd.DataFrame:
    """
    Load probability features from integrated classifiers and reorder to align with
    complexity feature labels (CAI first, then healthy).
    """
    spatial_probs = np.load(spatial_prob_path)
    spatial_labels = np.load(spatial_label_path)

    temporal_probs = np.load(temporal_prob_path)
    temporal_labels = np.load(temporal_label_path)

    # Align ordering: CAI samples (label==1) first, then healthy (label==0)
    spatial_probs_ordered = np.concatenate([
        spatial_probs[spatial_labels == 1],
        spatial_probs[spatial_labels == 0]
    ])
    temporal_probs_ordered = np.concatenate([
        temporal_probs[temporal_labels == 1],
        temporal_probs[temporal_labels == 0]
    ])

    probability_df = pd.DataFrame({
        'spatial_prob': spatial_probs_ordered,
        'temporal_prob': temporal_probs_ordered
    })

    return probability_df


def merge_all_features(complexity_df: pd.DataFrame,
                       probability_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Merge complexity features and classifier probabilities into a single feature matrix.
    """
    complexity_df = complexity_df.reset_index(drop=True)
    probability_df = probability_df.reset_index(drop=True)

    feature_df = pd.concat([complexity_df.drop(columns=['label']), probability_df], axis=1)
    labels = complexity_df['label']

    return feature_df, labels


def split_dataset(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.6, seed: int = 42):
    """
    Shuffle and split dataset into training and testing sets (stratified).
    """
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def train_random_forest_classifier(X_train, y_train) -> GridSearchCV:
    """
    Train Random Forest with 5-fold cross-validation using grid search.
    """
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'min_samples_leaf': [5, 10, 20, 30, 40, 50],
        'min_samples_split': [5, 10, 20, 30, 40, 50],
        'max_depth': [2, 4, 6, 8, 10, 12],
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model_performance(model: RandomForestClassifier,
                               X: pd.DataFrame,
                               y_true: pd.Series,
                               dataset_name: str):
    """
    Evaluate and print classification metrics and functional probability scores.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # Probability of class 1

    print(f"\n📊 Evaluation Results on {dataset_name}:")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_true, y_prob):.4f}")

    # Optional: Output functional scores
    print(f"\n🧠 Functional Scores (Probability of CAI) on {dataset_name}:")
    print(y_prob)


def compute_shap_importance(model: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHAP values and return feature-wise mean absolute SHAP values (unnormalized and unsorted).

    Parameters:
    - model: Trained RandomForestClassifier.
    - X: Input features (DataFrame) used for SHAP explanation.

    Returns:
    - A DataFrame with two columns: 'feature' and 'mean_abs_shap'.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[1]  # Class 1: CAI

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': mean_abs_shap
    })

    return shap_df


def main():
    # File paths
    complexity_file = 'muscle_synergy_complexity_features.xlsx'
    spatial_probs_file = 'integrated_spatial_classifier_output_probs.npy'
    spatial_labels_file = 'integrated_spatial_classifier_output_labels.npy'
    temporal_probs_file = 'integrated_temporal_classifier_output_probs.npy'
    temporal_labels_file = 'integrated_temporal_classifier_output_labels.npy'

    # Step 1: Load features
    complexity_df = load_complexity_features(complexity_file)
    probability_df = load_aligned_probability_features(
        spatial_probs_file,
        spatial_labels_file,
        temporal_probs_file,
        temporal_labels_file
    )

    # Step 2: Merge features
    X, y = merge_all_features(complexity_df, probability_df)

    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Step 4: Train model
    model_cv = train_random_forest_classifier(X_train, y_train)
    best_model = model_cv.best_estimator_

    # Step 5: Evaluate on train and test
    evaluate_model_performance(best_model, X_train, y_train, dataset_name="Training Set")
    evaluate_model_performance(best_model, X_test, y_test, dataset_name="Testing Set")

    # Step 6: Compute SHAP importance
    shap_df = compute_shap_importance(best_model, X)


if __name__ == '__main__':
    main()