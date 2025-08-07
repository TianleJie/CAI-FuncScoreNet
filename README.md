# CAI-FuncScoreNet
Official implementation of our interpretable, data-driven, hierarchical multi-domain fusion framework for classification and motor function scoring in chronic ankle instability (CAI).

Chronic ankle instability (CAI) is a common sports-related musculoskeletal disorder, characterized by recurrent sprains and neuromuscular control deficits. This study proposes an AI-enabled digital twin framework for sports health applications, offering both interpretability and clinical deployability. The framework identifies CAI using wearable sensor data-driven hierarchical multi-domain fusion models, generates fine-grained motor function scores through probabilistic modeling, and leverages SHAP-based interpretability to reveal key risk factors. This enables a transparent, closed-loop, and individualized rehabilitation strategy. Validation on 100 participants, including CAI patients and healthy controls, demonstrated robust classification performance (accuracy = 98.50%, AUC = 0.99) and strong concordance between generated scores and clinical gold standards (r = -0.908, p < 0.001). This non-invasive, personalized assessment tool supports long-term rehabilitation management of chronic conditions, providing an innovative and cost-effective digital health solution for sports medicine.

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
