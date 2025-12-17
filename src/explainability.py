import joblib
import pandas as pd
import numpy as np  # 记得加这个！
import shap
import matplotlib.pyplot as plt
from data_processing import load_and_split_data

# 设置 matplotlib 风格，防止中文乱码（可选，或者用默认）
plt.style.use('ggplot')

def explain_model(model_path='models/xgb_baseline.joblib', 
                  data_path='data/raw/UCI_Credit_Card.csv',
                  sample_size=500):
    """
    Generate SHAP explanations for the trained model.
    Args:
        sample_size: Number of samples to explain (smaller = faster)
    """
    print("="*60)
    print("🤖 MODEL EXPLAINABILITY ANALYSIS (SHAP)")
    print("="*60)
    
    # [1/6] Load data and model
    print("\n[1/6] Loading model and data...")
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    pipeline = joblib.load(model_path)
    print(f"✓ Loaded {len(X_test)} test samples")
    
    # [2/6] Extract pipeline components
    print("\n[2/6] Extracting model components...")
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']
    
    # [3/6] Preprocessing data for SHAP
    print("\n[3/6] Preprocessing data for SHAP...")
    X_test_transformed = preprocessor.transform(X_test)
    
    # 限制样本数量（工程优化）
    if sample_size < len(X_test):
        print(f"   ⚡ Optimization: Using {sample_size} samples (randomly sampled) for speed")
        # 随机抽样比只取前N个更科学
        indices = np.random.choice(X_test_transformed.shape[0], sample_size, replace=False)
        X_test_transformed = X_test_transformed[indices]
        y_test_subset = y_test.iloc[indices]
    else:
        y_test_subset = y_test
    
    # [4/6] Extracting feature names
    print("\n[4/6] Extracting feature names...")
    try:
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(num_features) + list(cat_features)
    except Exception as e:
        print(f"⚠ Warning: Could not extract names, using indices. Error: {e}")
        feature_names = [f"Feature_{i}" for i in range(X_test_transformed.shape[1])]
    
    # [5/6] Calculating SHAP values
    print("\n[5/6] Calculating SHAP values...")
    print("   (This allows us to open the 'Black Box' of the model)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)
    
    # [6/6] Generating visualizations & Reports
    print("\n[6/6] Generating visualizations...")
    
    # --- 图表 1: Summary Plot (全局) ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Saved: shap_summary.png")
    
    # --- 图表 2: Waterfall Plot (高风险个案) ---
    # 找一个预测概率最高的样本（最像坏人的样本）
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    high_risk_idx = y_pred_proba.argmax()
    
    plt.figure(figsize=(10, 8))
    exp = shap.Explanation(
        values=shap_values[high_risk_idx], 
        base_values=explainer.expected_value, 
        data=X_test_transformed[high_risk_idx], 
        feature_names=feature_names
    )
    shap.waterfall_plot(exp, show=False)
    plt.title(f"Why did the model reject this client? (Prob: {y_pred_proba[high_risk_idx]:.1%})", fontsize=12)
    plt.savefig('shap_waterfall_high_risk.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved: shap_waterfall_high_risk.png (Sample Index: {high_risk_idx})")
    
    # --- 交互式报告 (HTML) ---
    # 注意：force_plot 也是 js 渲染，保存为 html 最稳妥
    try:
        html_plot = shap.force_plot(explainer.expected_value, shap_values[:100], X_test_transformed[:100], feature_names=feature_names)
        shap.save_html('shap_interactive.html', html_plot)
        print("✓ Saved: shap_interactive.html")
    except Exception as e:
        print(f"⚠ Skipped HTML generation: {e}")

    # --- 改进3: 公平性隐患检查 ---
    print("\n" + "-"*30)
    print("🕵️ FAIRNESS CHECK (SHAP Based)")
    sex_features = [f for f in feature_names if 'SEX' in f] # 查找性别相关特征
    
    if sex_features:
        # 计算全局平均绝对 SHAP 值
        global_importances = np.abs(shap_values).mean(axis=0)
        total_importance = global_importances.sum()
        
        # 计算性别特征的贡献度
        sex_importance = sum([global_importances[feature_names.index(f)] for f in sex_features])
        sex_ratio = (sex_importance / total_importance) * 100
        
        print(f"   Gender Feature Importance: {sex_ratio:.2f}%")
        if sex_ratio > 5:
            print("   ⚠️  ALERT: Model relies heavily on Gender (>5%)!")
        else:
            print("   ✅ PASS: Model relies minimaly on Gender (<5%).")
    else:
        print("   ⚠ 'SEX' feature not found in feature names.")
    print("-"*30)

    print("\nDONE! All explanation assets are ready.")

if __name__ == "__main__":
    explain_model(sample_size=500)