import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
try:
    font_path = "SimHei.ttf"  # 替换为您系统中可用的中文字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except:
    st.warning("中文字体设置失败，可能显示为方框")

# 模型加载
model = joblib.load('RF.pkl')

# 特征定义
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2], "desc": "性别(1:男/Male, 2:女/Female)"},
    'rural2': {"type": "categorical", "options": [1,2], "desc": "户口类型 (1:农村, 2:城市)"},
    'diabe': {"type": "categorical", "options": [0,1], "desc": "糖尿病 (0:无/No, 1:有/Yes)"},
    'lunge': {"type": "categorical", "options": [0, 1], "desc": "肺部疾病 (0:无/No, 1:有/Yes)"},
    'hearte': {"type": "categorical", "options": [0, 1], "desc": "心脏病(0:无/No, 1:有/Yes)"},
    'dyslipe': {"type": "categorical", "options": [0, 1], "desc": "退休状态 (0:未退休/Not retired, 1:已退休/Retired)"},
    'hhcperc': {"type": "numerical", "min": 0.0, "max": 100.0, "default": 8.0, "desc": "家庭年收入 (万元)","step": 0.1,"format": "%.1f"},
    'retire': {"type": "categorical", "options": [0, 1], "desc": "退休状态 (0:未退休/Not retired, 1:已退休/Retired)"},
    'sleep': {"type": "numerical", "min": 0.0, "max": 24.0, "default": 8.0, "desc": "睡眠时长(小时/hours)","step": 0.1,"format": "%.1f"},
    'edu': {"type": "categorical", "options": [1,2,3,4], "desc": "教育程度 (1:小学以下/Below Primary, 2:小学/Primary, 3:中学/Secondary, 4:中学以上/Above Secondary)"},
    'da040': {"type": "categorical", "options": [0, 1], "desc": "慢性疼痛 (0:无/No, 1:有/Yes)"},
    'pension': {"type": "categorical", "options": [0, 1], "desc": "养老保险 (0:无/No, 1:有/Yes)"},
}

# 界面布局
st.title("Depression Risk-Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 输入表单
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=properties["desc"],
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            step=properties.get("step", 1.0),
            format=properties.get("format", "%f"),
            key=f"num_{feature}"
        )
    else:
        value = st.selectbox(
            label=properties["desc"],
            options=properties["options"],
            key=f"cat_{feature}"
        )
    feature_values.append(value)

# 预测与解释
if st.button("Predict"):
    # 创建DataFrame并确保特征名称匹配
    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    # 预测
    predicted_class = model.predict(feature_df)[0]
    predicted_proba = model.predict_proba(feature_df)[0]
    probability = predicted_proba[predicted_class] * 100

    # 结果显示
    st.subheader("预测结果")
    st.write(f"预测概率: {probability:.2f}% ({'高风险' if predicted_class == 1 else '低风险'})")

    # SHAP解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    
    # 处理SHAP值(适用于二分类和多分类)
    if isinstance(shap_values, list):
        # 多分类情况
        shap_values_class = shap_values[predicted_class]
        expected_value = explainer.expected_value[predicted_class]
    else:
        # 二分类情况
        shap_values_class = shap_values
        expected_value = explainer.expected_value[predicted_class]
    
    # 方法1: 使用HTML渲染force plot
    st.subheader("SHAP解释 - 特征影响")
    force_plot = shap.force_plot(
    base_value=expected_value,
    shap_values=shap_values_class[0],  # 确保是单个样本的 SHAP 值
    features=feature_df.iloc[0].values,  # 直接使用一维数组，不 reshape
    feature_names=feature_df.columns.tolist()  # 确保是列表格式
)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(shap_html, height=400, scrolling=True)