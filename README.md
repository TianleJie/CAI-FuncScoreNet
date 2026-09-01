# CAI-FuncScoreNet

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Paper DOI](https://img.shields.io/badge/Paper-10.1002%2Fadvs.77212-green.svg)](https://doi.org/10.1002/advs.77212)

Official implementation of our interpretable, data-driven, hierarchical multi-domain fusion framework for classification and motor function scoring in chronic ankle instability (CAI).

Chronic ankle instability (CAI) is a common sports-related musculoskeletal disorder characterized by recurrent sprains and neuromuscular control deficits. This study proposes an AI-enabled digital twin framework for sports health applications, offering both interpretability and clinical deployability. The framework identifies CAI and enables subtype stratification using a wearable electromyography (EMG) sensor-driven hierarchical multi-domain fusion model, generates fine-grained motor function scores through a probabilistic modeling approach, and further translates the generated scores into clinically interpretable functional stratification to support rehabilitation assessment. SHapley Additive exPlanations (SHAP)-based interpretability reveals key predictive biomarkers underlying model decisions, establishing a transparent and closed-loop framework for personalized rehabilitation. Validation on 150 participants confirms robust classification performance (Accuracy = 98.50%, AUC = 0.99), reliable discrimination of CAI subtypes, and strong concordance between the generated scores and the clinical gold-standard scale.

## Overview

![github图片](https://github.com/user-attachments/assets/2a06e0b6-da58-441a-992c-5bf0ca0e4656)

## Quick Start & Usage

This repository provides a modular, object-oriented Python script for training and evaluating the multi-domain fusion classifier and generating the continuous motor function score.

*   **`CAI_func_score_net.py`**: The core pipeline integrating spatial/temporal base classifiers, ensemble layers, and the motor function scoring algorithm.
*   **`requirements.txt`**: Specifies the required dependencies.

First, install the required dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements.txt
