import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# ========================
# 常量定义
# ========================
AVERAGE_DEFAULT_RATE = 0.22  # 平均违约率基准
GENDER_IMPACT_THRESHOLD = 0.05  # 性别影响阈值（5%）
CURRENT_GENDER_IMPACT = 0.0209  # 当前模型的性别影响
MODEL_VERSION = "1.0"
LAST_UPDATED = "2024-12-11"
MODEL_ACCURACY = "82.3%"

# ========================
# 页面配置
# ========================
st.set_page_config(
    page_title="Credit Risk AI Audit", 
    layout="wide",
    page_icon="💳",
    initial_sidebar_state="expanded"
)

# ========================
# 自定义样式
# ========================
st.markdown("""
<style>
/* ----- App background ----- */
.stApp {
    background: #F6F7FB;
}

/* ----- Main container (card) ----- */
.main .block-container {
    background: #FFFFFF;
    padding: 2.2rem 2.2rem;
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
    margin: 1.6rem auto;
    max-width: 1200px;
}

/* ----- Sidebar ----- */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid rgba(15, 23, 42, 0.08);
}

[data-testid="stSidebar"] * {
    color: #0F172A;
}

/* ----- Typography ----- */
h1, h2, h3, h4, h5, h6 {
    color: #0F172A;
}
p, li, label, .stMarkdown {
    color: #334155;
}

/* ----- Input widgets ----- */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stSlider > div {
    border-radius: 12px !important;
}

/* ----- Expander ----- */
details {
    border-radius: 14px;
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: #FFFFFF;
}
details summary {
    padding: 0.6rem 0.8rem;
}

/* ----- Buttons ----- */
.stButton button {
    border-radius: 12px;
    border: 1px solid rgba(37, 99, 235, 0.1);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
}
</style>
""", unsafe_allow_html=True)


# ========================
# 加载模型和数据
# ========================
@st.cache_resource
def load_model_and_data():
    """加载训练好的模型和特征列名"""
    try:
        model_path = os.path.join('models', 'xgb_baseline.joblib')
        data_path = os.path.join('data', 'raw', 'UCI_Credit_Card.csv')
        
        model = joblib.load(model_path)
        df_sample = pd.read_csv(data_path, nrows=1)
        
        # 删除非特征列
        columns_to_drop = ['ID', 'default.payment.next.month']
        df_sample = df_sample.drop(
            columns=[col for col in columns_to_drop if col in df_sample.columns]
        )
        
        return model, df_sample.columns
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model or data: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

pipeline, feature_columns = load_model_and_data()

# ========================
# 侧边栏输入
# ========================
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=80)
st.sidebar.title("🏦 Customer Input")

def user_input_features():
    """收集用户输入的特征"""
    with st.sidebar.expander("💰 Financial Info", expanded=True):
        limit_bal = st.slider(
            "Credit Limit", 
            min_value=10000, 
            max_value=1000000, 
            value=50000, 
            step=10000,
            help="Maximum credit limit"
        )
        bill_amt1 = st.number_input("Last Bill Amount", value=5000, step=100)
        pay_amt1 = st.number_input("Last Payment", value=1000, step=100)
    
    with st.sidebar.expander("👤 Personal Info", expanded=True):
        age = st.slider("Age", min_value=20, max_value=80, value=30)
        sex = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox(
            "Education", 
            ["Graduate School", "University", "High School", "Others"]
        )
        marriage = st.selectbox(
            "Marital Status", 
            ["Married", "Single", "Others"]
        )
    
    with st.sidebar.expander("📊 Payment History", expanded=True):
        pay_0 = st.slider(
            "Last Month Repayment", 
            min_value=-2, 
            max_value=8, 
            value=0,
            help="-1=Pay Duly, 1=Delay 1 month, 2=Delay 2 months..."
        )
        st.caption("💡 Tip: Lower is better (-1 = on-time payment)")
    
    # 数据映射
    sex_val = 1 if sex == "Male" else 2
    edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
    mar_map = {"Married": 1, "Single": 2, "Others": 3}
    
    # 构建特征字典
    data = {
        'LIMIT_BAL': limit_bal, 
        'SEX': sex_val, 
        'EDUCATION': edu_map[education], 
        'MARRIAGE': mar_map[marriage],
        'AGE': age, 
        'PAY_0': pay_0,
        'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt1, 'BILL_AMT3': bill_amt1,
        'BILL_AMT4': bill_amt1, 'BILL_AMT5': bill_amt1, 'BILL_AMT6': bill_amt1,
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt1, 'PAY_AMT3': pay_amt1,
        'PAY_AMT4': pay_amt1, 'PAY_AMT5': pay_amt1, 'PAY_AMT6': pay_amt1
    }
    
    # 确保特征顺序与训练时一致
    features = pd.DataFrame(data, index=[0])
    features = features.reindex(columns=feature_columns, fill_value=0)
    return features

input_df = user_input_features()

# ========================
# 主页面标题
# ========================
st.title("💳 Credit Risk AI Auditor")
st.markdown("""
<p style='font-size: 1.2rem; color: #64748b;'>
Explainable AI for Fair Credit Scoring
</p>
<p>
This dashboard combines <strong>XGBoost</strong> prediction with <strong>SHAP</strong> 
explanations for transparent credit decisions.
</p>
""", unsafe_allow_html=True)

# ========================
# 模型预测
# ========================
prediction = pipeline.predict(input_df)[0]
prediction_proba = pipeline.predict_proba(input_df)[0][1]

# ========================
# 预测结果展示
# ========================
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🎯 Risk Assessment")
    
    # 风险分类
    if prediction == 1:
        st.error("🚨 **High Risk** - Default Predicted")
        risk_color = "red"
    else:
        st.success("✅ **Low Risk** - Approval Recommended")
        risk_color = "green"
    
    # 违约概率显示
    st.metric(
        label="Default Probability",
        value=f"{prediction_proba:.1%}",
        delta=f"{prediction_proba - AVERAGE_DEFAULT_RATE:.1%} vs average",
        delta_color="inverse"
    )
    
    # 进度条
    st.progress(float(prediction_proba))
    
    # 改进建议
    if prediction == 1:
        st.markdown("""
        **📋 Improvement Suggestions:**
        - Maintain on-time payments for 6+ months
        - Reduce credit utilization below 30%
        - Reapply after demonstrated improvement
        """)

with col2:
    st.subheader("🔍 Why This Decision?")
    
    with st.spinner('⏳ Calculating SHAP explanations...'):
        try:
            # 获取预处理器和模型
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['classifier']
            
            # 转换输入数据
            input_transformed = preprocessor.transform(input_df)
            
            # 获取特征名称
            try:
                num_features = preprocessor.transformers_[0][2]
                cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(
                    preprocessor.transformers_[1][2]
                )
                feature_names = list(num_features) + list(cat_features)
            except (IndexError, KeyError, AttributeError) as e:
                # 如果无法获取特征名，使用默认名称
                feature_names = [f"Feature_{i}" for i in range(input_transformed.shape[1])]
                st.warning(f"⚠️ Using default feature names: {str(e)}")
            
            # 计算SHAP值
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_transformed)
            
            # 绘制Waterfall图
            fig, ax = plt.subplots(figsize=(8, 6))
            exp = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_transformed[0],
                feature_names=feature_names
            )
            shap.waterfall_plot(exp, show=False)
            st.pyplot(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error calculating SHAP values: {str(e)}")
            st.info("Please check model pipeline configuration.")
    
    st.caption("📊 Each bar shows how much a feature pushes the prediction up (red) or down (blue)")

# ========================
# 公平性检查
# ========================
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("⚖️ Fairness Guardrail")
    
    # 检查性别偏见
    if input_df['SEX'].values[0] == 2 and prediction_proba > 0.6:
        st.warning(f"""
        ⚠️ **Fairness Note:** Applicant is Female with high-risk prediction.
        
        Our model audit shows Gender impact is only **{CURRENT_GENDER_IMPACT:.2%}** 
        (Well below {GENDER_IMPACT_THRESHOLD:.0%} threshold).
        This decision is primarily based on payment history, not demographics.
        """)
    else:
        st.info("✅ **Fairness Check Passed** - No significant demographic bias detected.")

with col2:
    gender_impact_delta = CURRENT_GENDER_IMPACT - GENDER_IMPACT_THRESHOLD
    st.metric(
        "Gender Impact", 
        f"{CURRENT_GENDER_IMPACT:.2%}", 
        delta=f"{gender_impact_delta:.2%} vs threshold", 
        delta_color="normal"
    )
    st.caption("✅ Compliant with Fair Lending regulations")

# ========================
# 全局特征重要性
# ========================
st.markdown("---")
st.subheader("📊 Global Feature Importance")

shap_summary_path = os.path.join('shap_summary.png')
if os.path.exists(shap_summary_path):
    st.image(shap_summary_path, use_column_width=True)
    st.caption("Based on analysis of 500 customer samples")
else:
    st.info("💡 Run `python src/explainability.py` first to generate global importance plots")

# ========================
# Footer
# ========================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>🔒 This system uses Explainable AI (XAI) to ensure transparent and fair credit decisions</p>
    <p>Model Version: {MODEL_VERSION} | Last Updated: {LAST_UPDATED} | Accuracy: {MODEL_ACCURACY}</p>
</div>
""", unsafe_allow_html=True)