# 🏦 Explainable Credit Risk Scoring Engine

![Python](https://img.shields.io/badge/Python-3.9-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)

## Executive Summary

This project demonstrates a production-style credit risk scoring system designed for regulated banking environments.
It combines a high-performance model (XGBoost) with explainability (SHAP) and fairness guardrails (Fairlearn-style auditing concepts)
to support compliant credit decisions for an existing-customer risk management use case (B-scorecard scenario).

The system outputs default probability predictions, provides human-readable explanations for individual decisions, and surfaces a fairness note
when a sensitive attribute (gender) is involved in a high-risk outcome.

---

## Key Features

### 1) High-Performance Risk Model (XGBoost + Pipeline)
- Trains a credit risk classifier using **XGBoost**.
- Uses an `sklearn`-style pipeline workflow to ensure consistent preprocessing and inference.

### 2) Explainability (SHAP)
- **Local explanations**: Generates **SHAP waterfall plots** for a single applicant, showing how features push the prediction up or down.
- **Global importance**: Supports global feature importance visualization (when `shap_summary.png` is generated).

**Why it matters:** Enables risk teams to justify model decisions during stakeholder review and model governance.

### 3) Fairness Guardrail (Compliance-Oriented Messaging)
- Includes a fairness note around sensitive attribute (gender) in high-risk outcomes.
- Uses a **<5% threshold** as a practical guardrail reference for demographic impact.

**Why it matters:** Helps reduce legal and reputational risks by explicitly monitoring sensitive attributes in decisioning.

### 4) Interactive Streamlit Dashboard
- **Single-customer mode**: Simulates a loan officer workflow with real-time scoring.
- **Actionable suggestions**: Shows practical improvement tips for high-risk predictions.
- **Decision transparency**: Displays “Why this decision?” explanations alongside the prediction output.

---

## Project Structure

```text
credit-scoring-project/
├── data/
│   └── raw/                   # Dataset file location (see "Data" section)
├── models/
│   └── xgb_baseline.joblib    # Trained model pipeline
├── src/                       # Core modules
│   ├── data_processing.py     # Data loading & train/test split
│   ├── model_training.py      # Pipeline training & evaluation
│   ├── explainability.py      # SHAP utilities / plot generation
│   └── fairness_audit.py      # Fairness metrics / checks (if applicable)
├── main.py                    # Train & save model entry
├── app.py                     # Streamlit application
├── Dockerfile                 # Container build
├── requirements.txt           # Dependencies
└── README.md
