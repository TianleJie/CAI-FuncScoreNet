# CAI_func_score_net.py
Official implementation of our interpretable, data-driven, hierarchical multi-domain fusion framework for classification and motor function scoring in chronic ankle instability (CAI).

Chronic ankle instability (CAI) is a common sports-related musculoskeletal disorder characterized by recurrent sprains and neuromuscular control deficits. It affects a wide range of individuals, from recreational to elite athletes, and poses a substantial healthcare burden. This study proposes an AI-enabled digital twin framework for sports health applications, offering both interpretability and clinical deployability. The framework identifies CAI and enables subtype stratification using a wearable electromyography (EMG) sensor-driven hierarchical multi-domain fusion model, generates fine-grained motor function scores through a probabilistic modeling approach, and further translates the generated scores into clinically interpretable functional stratification to support rehabilitation assessment. SHapley Additive exPlanations (SHAP) - based interpretability reveals key predictive biomarkers underlying model decisions, establishing a transparent and closed-loop framework for personalized rehabilitation. Validation on 150 participants, including CAI patients and healthy controls, confirms robust classification performance (Accuracy = 98.50%, AUC = 0.99), reliable discrimination of CAI subtypes (Accuracy = 87.70%, macro F1-score = 87.20%), and strong concordance between the generated scores and the clinical gold-standard scale (r = −0.908,  p  <  0.001). This non-invasive, personalized assessment framework supports long-term rehabilitation management of chronic conditions, offering an innovative and cost-effective digital health solution for sports medicine.

## Overview

![github图片](https://github.com/user-attachments/assets/2a06e0b6-da58-441a-992c-5bf0ca0e4656)

## Files

- `integrated_spatial_classifier_pipeline.py`: Spatial domain classifier pipeline.
- `integrated_temporal_classifier_pipeline.py`: Temporal domain classifier pipeline.
- `Muti-Domain_fusion_classifier_pipeline.py`: Fusion classifier pipeline integrating multiple domains.

## Usage

This repository provides modular Python scripts for training and evaluating multi-domain fusion classifiers for CAI detection and motor function scoring. 

### Prerequisites

- Python 3.7 or higher
- Required Python packages (install via pip):

```bash
pip install -r requirements.txt
