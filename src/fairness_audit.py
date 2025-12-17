import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Fairlearn metrics
from fairlearn.metrics import (
    MetricFrame, 
    selection_rate, 
    false_positive_rate, 
    false_negative_rate, 
    count
)

# Sklearn metrics (重要！)
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Local imports
from data_processing import load_and_split_data


def run_audit(model_path='models/xgb_baseline.joblib', 
              data_path='data/raw/UCI_Credit_Card.csv'):
    """
    Run fairness audit on the trained model.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the dataset
        
    Returns:
        MetricFrame object with fairness metrics
    """
    # 1. Load data and model
    print("Loading data and model...")
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    pipeline = joblib.load(model_path)
    
    # 2. Get predictions
    print("Generating predictions...")
    y_pred = pipeline.predict(X_test)
    
    # 3. Prepare sensitive features
    print("Preparing sensitive features...")
    sensitive_feature = X_test['SEX'].map({1: 'Male', 2: 'Female'})
    
    # 4. Define metrics (using actual functions)
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "selection_rate": selection_rate,
        "count": count
    }
    
    # 5. Build MetricFrame
    print("Calculating fairness metrics...")
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    
    # 6. Print results
    print("\n" + "="*60)
    print("FAIRNESS AUDIT REPORT - GROUP BY SEX")
    print("="*60)
    print(mf.by_group)
    
    print("\n" + "="*60)
    print("MAX DIFFERENCE BETWEEN GROUPS")
    print("="*60)
    print(mf.difference())
    
    # 7. Evaluate fairness
    print("\n" + "="*60)
    print("FAIRNESS ASSESSMENT")
    print("="*60)
    
    fpr_diff = mf.difference()['false_positive_rate']
    selection_diff = mf.difference()['selection_rate']
    
    if fpr_diff > 0.10:
        print("⚠️  HIGH RISK: FPR difference > 10% - Significant bias detected!")
    elif fpr_diff > 0.05:
        print("⚠️  MEDIUM RISK: FPR difference > 5% - Model needs improvement")
    else:
        print("✅ LOW RISK: FPR difference < 5% - Acceptable fairness level")
    
    print(f"\nFalse Positive Rate Difference: {fpr_diff:.2%}")
    print(f"Selection Rate Difference: {selection_diff:.2%}")
    
    return mf


if __name__ == "__main__":
    mf = run_audit()