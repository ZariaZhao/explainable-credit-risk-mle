# 🏦 Explainable Credit Risk Scoring Engine (可解释性信用风控引擎)

![Python](https://img.shields.io/badge/Python-3.9-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)

> **项目简介**：这是一个端到端的机器学习工程项目，旨在构建一个既具备高性能（XGBoost），又符合金融监管要求（可解释性 + 公平性）的信用评分系统。该系统专注于**存量客户管理（B卡模型）**，不仅输出违约概率，还能通过 SHAP 值解释决策原因，并实时监控模型对不同性别的公平性。

## 🌟 核心功能 (Key Features)

### 1. 🧠 高性能风控模型 (The Engine)
- 基于 **XGBoost** 构建分类器，使用 `sklearn Pipeline` 进行数据预处理（缺失值填补、OneHot编码）。
- 实现了**模块化编程**，将数据处理、训练、解释逻辑分离，符合 MLE 工程标准。

### 2. 🔍 白盒可解释性 (Explainability)
- **局部解释 (Local)**: 集成 **SHAP Waterfall Plot**，针对单个客户展示具体的拒贷原因（例如：因为 `PAY_0` 逾期导致概率增加 30%）。
- **全局解释 (Global)**: 提供特征重要性总览，识别影响违约的核心因子。
- **反事实分析 (What-If)**: 提供交互式敏感度分析，模拟"如果客户按时还款，通过率会提升多少"。

### 3. ⚖️ 公平性审计 (Fairness Audit)
- 集成微软 **Fairlearn** 框架。
- 针对敏感属性（性别）进行偏差检测，确保 Demographic Parity Difference 控制在合规范围内 (< 5%)。

### 4. 💻 交互式业务中台 (Streamlit Dashboard)
- **单客模式**: 模拟信贷审批员的操作界面，实时计算风险分。
- **批量模式**: 支持上传 CSV 文件进行批量风险评估。
- **智能建议**: 针对被拒客户生成可落地的信用修复建议。

---

## 📂 项目结构 (Project Structure)

```text
credit-scoring-project/
├── .github/                   # CI/CD workflows (Optional)
├── data/
│   └── raw/                   # UCI Dataset (ignored in git)
├── models/
│   └── xgb_baseline.joblib    # Trained Model Pipeline
├── src/                       # Source Code
│   ├── data_processing.py     # ETL & Splitting
│   ├── model_training.py      # Pipeline Building & Training
│   ├── explainability.py      # SHAP Analysis Script
│   └── fairness_audit.py      # Fairness Metrics Calculation
├── app.py                     # Streamlit Application Entry
├── Dockerfile                 # Container Configuration
├── requirements.txt           # Python Dependencies
└── README.md                  # Project Documentation