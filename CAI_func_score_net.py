"""
CAI-FuncScoreNet: Hierarchical Multi-Domain Fusion Framework
------------------------------------------------------------
Official implementation of the interpretable, data-driven, hierarchical
multi-domain fusion framework for classification and continuous motor
function scoring in Chronic Ankle Instability (CAI).

Author: Tianle Jie et al. (Advanced Science)
"""

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

        Parameters:
        -----------
        X_S : ndarray of shape (n_samples, 3) - Spatial features [F_w, F_wf, OVR]
        X_T : ndarray of shape (n_samples, 19) - Temporal features
        S_S : ndarray of shape (n_samples, 1) - Spatial synergy similarity scores
        S_T : ndarray of shape (n_samples, 1) - Temporal synergy similarity scores
        C_features : ndarray of shape (n_samples, n) - Complexity features [VAF, N_opt]
        y : ndarray of shape (n_samples,) - Ground truth labels (+1 Pathological, -1 Healthy)
        groups : ndarray of shape (n_samples,) - Subject IDs to prevent data leakage during splitting.
        """

        if groups is None:
            # Fallback to simple index array if no subject IDs are provided,
            # though subject-wise splitting is highly recommended for multiple trials.
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

        # ==========================================
        # Stage 1: Train Base Classifiers (40%)
        # ==========================================
        self.spatial_base.fit(X_S[idx_base], y[idx_base])
        self.temporal_base.fit(X_T[idx_base], y[idx_base])

        # ==========================================
        # Stage 2: Train Ensemble Classifiers (30%)
        # ==========================================
        # Predict base probabilities for Ensemble training subset (Eq. 3, 4)
        p_S_ens = self.spatial_base.predict_proba(X_S[idx_ens])[:, 1]
        p_T_ens = self.temporal_base.predict_proba(X_T[idx_ens])[:, 1]

        # Construct input vectors E_S and E_T (Eq. 5, 6)
        E_S_train = np.column_stack((p_S_ens, S_S[idx_ens]))
        E_T_train = np.column_stack((p_T_ens, S_T[idx_ens]))

        self.spatial_ensemble.fit(E_S_train, y[idx_ens])
        self.temporal_ensemble.fit(E_T_train, y[idx_ens])

        # ==========================================
        # Stage 3: Train Multi-Domain Fusion Classifier (30%)
        # ==========================================
        # Get base predictions for Fusion subset
        p_S_fus = self.spatial_base.predict_proba(X_S[idx_fus])[:, 1]
        p_T_fus = self.temporal_base.predict_proba(X_T[idx_fus])[:, 1]

        # Construct intermediate ensemble inputs
        E_S_fus = np.column_stack((p_S_fus, S_S[idx_fus]))
        E_T_fus = np.column_stack((p_T_fus, S_T[idx_fus]))

        # Get ensemble predictions (Eq. 7, 8)
        Y_S_fus = self.spatial_ensemble.predict_proba(E_S_fus)[:, 1]
        Y_T_fus = self.temporal_ensemble.predict_proba(E_T_fus)[:, 1]

        # Construct final fusion input M_F (Eq. 9, 10)
        M_F_train = np.column_stack((Y_S_fus, Y_T_fus, C_features[idx_fus]))

        # Final Fusion (Eq. 11)
        self.fusion_classifier.fit(M_F_train, y[idx_fus])

        return self

    def predict_proba(self, X_S, X_T, S_S, S_T, C_features):
        """
        Predict pathological probabilities for new, unseen testing data.
        """
        # Base level
        p_S = self.spatial_base.predict_proba(X_S)[:, 1]
        p_T = self.temporal_base.predict_proba(X_T)[:, 1]

        # Ensemble level
        E_S = np.column_stack((p_S, S_S))
        E_T = np.column_stack((p_T, S_T))
        Y_S = self.spatial_ensemble.predict_proba(E_S)[:, 1]
        Y_T = self.temporal_ensemble.predict_proba(E_T)[:, 1]

        # Fusion level
        M_F = np.column_stack((Y_S, Y_T, C_features))
        P_final = self.fusion_classifier.predict_proba(M_F)[:, 1]

        return P_final

    def generate_motor_function_score(self, P_final):
        """
        Map predicted probabilities to a continuous motor function score (0-100).
        (See Section 5.6)
        """
        scores = np.round(P_final * 100, 2)
        return np.clip(scores, 0, 100)

    def assign_clinical_grade(self, scores):
        """
        Classify continuous scores into clinical severity levels (See Section 5.7 & Fig 5e).
        - Cut-off: 62.0
        - Mild threshold: 53.7
        - Severe threshold: 67.6
        """
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
# Quick Start / Demo Usage
# =====================================================================
if __name__ == "__main__":
    print("Initializing CAI-FuncScoreNet Demo...")

    # 1. Generate Dummy Data (Simulating features from Figshare)
    n_samples = 200
    np.random.seed(42)

    # Simulating 5 trials per subject, so 40 subjects total
    subject_ids = np.repeat(np.arange(40), 5)
    y_dummy = np.random.choice([-1, 1], size=n_samples)  # Labels

    X_S_dummy = np.random.rand(n_samples, 3)  # 3 Spatial features
    X_T_dummy = np.random.rand(n_samples, 19)  # 19 Temporal features
    S_S_dummy = np.random.rand(n_samples)  # Spatial similarity
    S_T_dummy = np.random.rand(n_samples)  # Temporal similarity
    C_feat_dummy = np.random.rand(n_samples, 2)  # Complexity (e.g., VAF, N_opt)

    # 2. Instantiate and Train the Model
    model = CAIFuncScoreNet(random_state=42)
    print("Training cascaded hierarchical model with subject-wise 4:3:3 split...")
    model.fit(X_S_dummy, X_T_dummy, S_S_dummy, S_T_dummy, C_feat_dummy, y_dummy, groups=subject_ids)

    # 3. Inference & Scoring on 'New' Data
    # In a real scenario, this would be an independent test set (40% of subjects)
    print("Generating predictions and continuous motor function scores...")
    probabilities = model.predict_proba(X_S_dummy[:10], X_T_dummy[:10], S_S_dummy[:10], S_T_dummy[:10],
                                        C_feat_dummy[:10])
    motor_scores = model.generate_motor_function_score(probabilities)
    clinical_grades = model.assign_clinical_grade(motor_scores)

    # 4. Display Results
    results = pd.DataFrame({
        'Subject_ID': subject_ids[:10],
        'Probability': probabilities,
        'Motor_Function_Score': motor_scores,
        'Severity_Grade': clinical_grades
    })

    print("\n--- Model Output Preview ---")
    print(results.to_string(index=False))