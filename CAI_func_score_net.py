"""
CAI-FuncScoreNet: Hierarchical Multi-Domain Fusion Framework
------------------------------------------------------------
Official implementation of the interpretable, data-driven, hierarchical
multi-domain fusion framework for classification and continuous motor
function scoring in Chronic Ankle Instability (CAI).

Author: Tianle Jie et al. (Advanced Science)
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
import warnings

# Suppress minor warnings for cleaner console output
warnings.filterwarnings('ignore')


class CAIFuncScoreNet:
    def __init__(self, random_state=42):
        """
        Initialize the multi-stage cascaded Random Forest framework with
        optimal hyperparameters specified in the paper (Table 1).
        """
        self.random_state = random_state

        # Stage 1: Base Classifiers
        self.spatial_base = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5, min_samples_split=20, random_state=self.random_state)
        self.temporal_base = RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=5, min_samples_split=5, random_state=self.random_state)

        # Stage 2: Ensemble Classifiers
        self.spatial_ensemble = RandomForestClassifier(
            n_estimators=50, max_depth=6, min_samples_leaf=5, min_samples_split=5, random_state=self.random_state)
        self.temporal_ensemble = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, min_samples_split=5, random_state=self.random_state)

        # Stage 3: Multi-Domain Fusion Classifier
        self.fusion_classifier = RandomForestClassifier(
            n_estimators=50, max_depth=6, min_samples_leaf=5, min_samples_split=5, random_state=self.random_state)

    def fit(self, X_S, X_T, S_S, S_T, C_features, y, groups=None):
        """
        Train the hierarchical pipeline using the subject-wise 4:3:3 cascaded
        splitting strategy (Section 5.8).
        """
        if groups is None:
            groups = np.arange(len(y))

        # Split 1: 40% for Base models, 60% for the rest (Ens + Fus)
        gss_base = GroupShuffleSplit(n_splits=1, test_size=0.60, random_state=self.random_state)
        idx_base, idx_rest = next(gss_base.split(X_S, y, groups))

        # Split 2: Divide the remaining 60% into two 30% subsets for Ens and Fus
        gss_ens = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=self.random_state)
        idx_ens, idx_fus = next(gss_ens.split(X_S[idx_rest], y[idx_rest], groups[idx_rest]))

        # Map relative indices back to absolute indices
        idx_ens = idx_rest[idx_ens]
        idx_fus = idx_rest[idx_fus]

        # Stage 1: Train Base Classifiers (40%)
        self.spatial_base.fit(X_S[idx_base], y[idx_base])
        self.temporal_base.fit(X_T[idx_base], y[idx_base])

        # Stage 2: Train Ensemble Classifiers (30%)
        p_S_ens = self.spatial_base.predict_proba(X_S[idx_ens])[:, 1]
        p_T_ens = self.temporal_base.predict_proba(X_T[idx_ens])[:, 1]

        E_S_train = np.column_stack((p_S_ens, S_S[idx_ens]))
        E_T_train = np.column_stack((p_T_ens, S_T[idx_ens]))

        self.spatial_ensemble.fit(E_S_train, y[idx_ens])
        self.temporal_ensemble.fit(E_T_train, y[idx_ens])

        # Stage 3: Train Multi-Domain Fusion Classifier (30%)
        p_S_fus = self.spatial_base.predict_proba(X_S[idx_fus])[:, 1]
        p_T_fus = self.temporal_base.predict_proba(X_T[idx_fus])[:, 1]

        E_S_fus = np.column_stack((p_S_fus, S_S[idx_fus]))
        E_T_fus = np.column_stack((p_T_fus, S_T[idx_fus]))

        Y_S_fus = self.spatial_ensemble.predict_proba(E_S_fus)[:, 1]
        Y_T_fus = self.temporal_ensemble.predict_proba(E_T_fus)[:, 1]

        M_F_train = np.column_stack((Y_S_fus, Y_T_fus, C_features[idx_fus]))
        self.fusion_classifier.fit(M_F_train, y[idx_fus])

        return self

    def predict_proba(self, X_S, X_T, S_S, S_T, C_features):
        """
        Predict pathological probabilities for new, unseen testing data.
        """
        p_S = self.spatial_base.predict_proba(X_S)[:, 1]
        p_T = self.temporal_base.predict_proba(X_T)[:, 1]

        E_S = np.column_stack((p_S, S_S))
        E_T = np.column_stack((p_T, S_T))
        Y_S = self.spatial_ensemble.predict_proba(E_S)[:, 1]
        Y_T = self.temporal_ensemble.predict_proba(E_T)[:, 1]

        M_F = np.column_stack((Y_S, Y_T, C_features))
        P_final = self.fusion_classifier.predict_proba(M_F)[:, 1]

        return P_final

    def generate_motor_function_score(self, P_final):
        """Map predicted probabilities to a continuous motor function score (0-100)."""
        scores = np.round(P_final * 100, 2)
        return np.clip(scores, 0, 100)

    def assign_clinical_grade(self, scores):
        """Classify continuous scores into clinical severity levels."""
        grading = []
        for score in scores:
            if score < 53.7:
                grading.append("Healthy / Normal")
            elif 53.7 <= score < 62.0:
                grading.append("Sub-clinical / Minimal Impairment")
            elif 62.0 <= score < 67.6:
                grading.append("Mild Impairment (CAI)")
            else:
                grading.append("Severe Impairment (CAI)")
        return np.array(grading)


# =====================================================================
# Pipeline Execution
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAI-FuncScoreNet Training and Evaluation Pipeline")
    parser.add_argument('--data_dir', type=str, default='./features',
                        help="Directory containing the user-extracted feature CSV files.")
    args = parser.parse_args()

    print("Initializing CAI-FuncScoreNet Pipeline...\n")

    model = CAIFuncScoreNet(random_state=42)

    print(f"Loading extracted features from '{args.data_dir}'...")
    try:
        # Expected file structure from user's feature extraction process
        df_spatial = pd.read_csv(os.path.join(args.data_dir, "spatial_features.csv"))
        df_temporal = pd.read_csv(os.path.join(args.data_dir, "temporal_features.csv"))
        df_similarity = pd.read_csv(os.path.join(args.data_dir, "similarity_scores.csv"))
        df_complexity = pd.read_csv(os.path.join(args.data_dir, "complexity_features.csv"))
        df_labels = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))

        subject_ids = df_labels['Subject_ID'].values
        y_true = df_labels['Label'].values

        X_S = df_spatial.drop(columns=['Subject_ID']).values
        X_T = df_temporal.drop(columns=['Subject_ID']).values
        S_S = df_similarity['Spatial_Similarity'].values
        S_T = df_similarity['Temporal_Similarity'].values
        C_feat = df_complexity[['VAF', 'N_opt']].values

        print("Feature files successfully loaded.\n")

    except FileNotFoundError as e:
        print("\n[Error] Required feature files not found.")
        print("To run this pipeline, please complete the following steps:")
        print("  1. Download the raw sEMG data from our Figshare repository.")
        print(
            "  2. Process the signals and execute the NNMF feature extraction as detailed in the Methodology section of the paper.")
        print("  3. Save the extracted features as CSVs in the specified '--data_dir' (default: './features').")
        print(f"\nMissing file: {e.filename}")
        exit(1)

    # Core training and evaluation workflow
    print("Training cascaded hierarchical model with subject-wise 4:3:3 split...")
    model.fit(X_S, X_T, S_S, S_T, C_feat, y_true, groups=subject_ids)

    print("Generating predictions and continuous motor function scores...")
    probabilities = model.predict_proba(X_S, X_T, S_S, S_T, C_feat)
    motor_scores = model.generate_motor_function_score(probabilities)
    clinical_grades = model.assign_clinical_grade(motor_scores)

    # Display results
    results = pd.DataFrame({
        'Subject_ID': subject_ids,
        'True_Label': y_true,
        'Predicted_Probability': probabilities,
        'Motor_Function_Score': motor_scores,
        'Severity_Grade': clinical_grades
    })

    print("\n--- Model Output Preview (Top 10) ---")
    print(results.head(10).to_string(index=False))