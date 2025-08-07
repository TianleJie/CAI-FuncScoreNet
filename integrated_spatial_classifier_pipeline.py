import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def load_and_merge_data(spatial_path, integrated_path, sheet_CAI, sheet_normal):
    # Load CAI group
    spatial_CAI = pd.read_excel(spatial_path, sheet_name=sheet_CAI)
    integrated_CAI = pd.read_excel(integrated_path, sheet_name=sheet_CAI)
    CAI_data = pd.concat([spatial_CAI, integrated_CAI], axis=1)
    CAI_data['Label'] = 1

    # Load Normal group
    spatial_normal = pd.read_excel(spatial_path, sheet_name=sheet_normal)
    integrated_normal = pd.read_excel(integrated_path, sheet_name=sheet_normal)
    normal_data = pd.concat([spatial_normal, integrated_normal], axis=1)
    normal_data['Label'] = 0

    # Combine and shuffle
    full_data = pd.concat([CAI_data, normal_data], axis=0).reset_index(drop=True)
    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return full_data


def split_dataset(data):
    # 60% for model training and 40% for final hold-out test
    data_train_val, data_test_final = train_test_split(
        data, test_size=0.4, stratify=data['Label'], random_state=42
    )

    # From the 60%, further split into 40% (part1), 30% (part2), 30% (part3)
    part1, temp = train_test_split(
        data_train_val, test_size=0.6, stratify=data_train_val['Label'], random_state=42
    )
    part2, part3 = train_test_split(
        temp, test_size=0.5, stratify=temp['Label'], random_state=42
    )

    return part1, part2, part3, data_test_final


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'min_samples_leaf': [5, 10, 20, 30, 40, 50],
        'min_samples_split': [5, 10, 20, 30, 40, 50],
        'max_depth': [2, 4, 6, 8, 10, 12],
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }
    return results, y_prob

def calculate_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer

def main():
    # ==== Step 1: Load and merge data ====
    spatial_file = 'spatial_features.xlsx'
    integrated_file = 'spatial_integrated_features.xlsx'
    sheet_CAI = 'CAI'
    sheet_normal = 'Normal'

    full_data = load_and_merge_data(spatial_file, integrated_file, sheet_CAI, sheet_normal)

    # ==== Step 2: Split dataset ====
    part1, part2, part3, part4 = split_dataset(full_data)

    # ==== Step 3: Train spatial classifier ====
    spatial_features = full_data.columns[:21]
    integrated_features = full_data.columns[21:25]

    X_train_spatial = part1[spatial_features]
    y_train_spatial = part1['Label']
    X_test_spatial = part2[spatial_features]
    y_test_spatial = part2['Label']

    spatial_model = train_random_forest(X_train_spatial, y_train_spatial)
    print("✅ Spatial classifier trained.")

    # Predict probabilities
    prob_train_part1 = spatial_model.predict_proba(part1[spatial_features])[:, 1]
    prob_test_part2 = spatial_model.predict_proba(part2[spatial_features])[:, 1]
    prob_train_part3 = spatial_model.predict_proba(part3[spatial_features])[:, 1]

    # ==== Step 4: Build training and testing sets for integrated classifier ====
    X_train_integrated_part1 = part1[integrated_features].reset_index(drop=True)
    X_train_integrated_part1['Spatial_Prob'] = prob_train_part1

    X_train_integrated_part3 = part3[integrated_features].reset_index(drop=True)
    X_train_integrated_part3['Spatial_Prob'] = prob_train_part3

    X_train_integrated = pd.concat([X_train_integrated_part1, X_train_integrated_part3], axis=0).reset_index(drop=True)
    y_train_integrated = pd.concat([part1['Label'], part3['Label']], axis=0).reset_index(drop=True)

    X_test_integrated = part2[integrated_features].reset_index(drop=True)
    X_test_integrated['Spatial_Prob'] = prob_test_part2
    y_test_integrated = part2['Label'].reset_index(drop=True)

    # ==== Step 5: Train and evaluate integrated classifier ====
    integrated_model = train_random_forest(X_train_integrated, y_train_integrated)
    results, y_prob_integrated = evaluate_model(integrated_model, X_test_integrated, y_test_integrated)
    np.save('integrated_temporal_classifier_output_probs.npy', y_prob_integrated)

    print("\n🎯 Evaluation Results (Integrated Classifier):")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # === Step 6: SHAP Explainability ===
    shap_values, explainer = calculate_shap(integrated_model, X_test_integrated) # reserved for later SHAP analysis


if __name__ == '__main__':
    main()