# CAI-FuncScoreNet

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Paper DOI](https://img.shields.io/badge/Paper-10.1002%2Fadvs.77212-green.svg)](https://doi.org/10.1002/advs.77212)

Official implementation of our interpretable, data-driven, hierarchical multi-domain fusion framework for classification and motor function scoring in chronic ankle instability (CAI).

Chronic ankle instability (CAI) is a common sports-related musculoskeletal disorder characterized by recurrent sprains and neuromuscular control deficits. This study proposes an AI-enabled digital twin framework for sports health applications, offering both interpretability and clinical deployability. The framework identifies CAI and enables subtype stratification using a wearable electromyography (EMG) sensor-driven hierarchical multi-domain fusion model, generates fine-grained motor function scores through a probabilistic modeling approach, and further translates the generated scores into clinically interpretable functional stratification to support rehabilitation assessment. SHapley Additive exPlanations (SHAP)-based interpretability reveals key predictive biomarkers underlying model decisions, establishing a transparent and closed-loop framework for personalized rehabilitation. Validation on 150 participants confirms robust classification performance (Accuracy = 98.50%, AUC = 0.99), reliable discrimination of CAI subtypes, and strong concordance between the generated scores and the clinical gold-standard scale.

## Overview

![github图片](https://github.com/user-attachments/assets/2a06e0b6-da58-441a-992c-5bf0ca0e4656)

## Pipeline Workflow & Usage

This repository provides the modular, object-oriented Python pipeline for training the cascaded Random Forest models and generating the continuous motor function score. 

### 1. Data Preparation
The raw sEMG data required for this study is hosted externally. To run the classification pipeline, users must complete the feature extraction process:
*   Download the raw sEMG dataset (Excel format containing CAI and Healthy cohorts) from our Figshare repository (Link available in the published paper).
*   Implement the signal processing (filtering, muscle activation modeling) and Non-negative Matrix Factorization (NNMF) algorithms precisely as detailed in **Section 5 (Methods)** of our paper to extract the spatial, temporal, similarity, and complexity features.
*   Save the extracted features into corresponding CSV files (e.g., `spatial_features.csv`, `temporal_features.csv`) in a designated local directory.

### 2. Running the Model
Once your features are extracted and saved, install the required dependencies:

```bash
pip install -r requirements.txt
