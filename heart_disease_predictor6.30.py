import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('XGB.pkl')

feature_ranges = {
    'gender': {
        "type": "categorical",
        "options": [1, 2],
        "name": "性别",
        "options_desc": {1: "男", 2: "女"},
        "help": "请选择您的性别"
    },
    'srh': {
        "type": "categorical",
        "options": [1, 2, 3, 4, 5],
        "name": "自评健康状况",
        "options_desc": {1: "非常好", 2: "好", 3: "一般", 4: "差", 5: "非常差"},
        "help": "您如何评价自己的健康状况？"
    },
    'adlab_c': {
        "type": "categorical",
        "options": [0, 1, 2, 3, 4, 5, 6],
        "name": "慢性病数量",
        "options_desc": {0: "无", 1: "1种", 2: "2种", 3: "3种", 4: "4种", 5: "5种", 6: "6种及以上"},
        "help": "您被诊断出有多少种慢性病？"
    },
    'arthre': {
        "type": "categorical",
        "options": [0, 1],
        "name": "关节炎",
        "options_desc": {0: "否", 1: "是"},
        "help": "您是否患有关节炎？"
    },
    'digeste': {
        "type": "categorical",
        "options": [0, 1],
        "name": "消化系统疾病",
        "options_desc": {0: "否", 1: "是"},
        "help": "您是否有消化系统疾病？"
    },
    'retire': {
        "type": "categorical",
        "options": [0, 1],
        "name": "退休状态",
        "options_desc": {0: "未退休", 1: "已退休"},
        "help": "您是否已退休？"
    },
    'satlife': {
        "type": "categorical",
        "options": [1, 2, 3, 4, 5],
        "name": "生活满意度",
        "options_desc": {1: "非常不满意", 2: "不满意", 3: "一般", 4: "满意", 5: "非常满意"},
        "help": "您对当前生活的满意度如何？"
    },
    'sleep': {
        "type": "numerical",
        "min": 0.0,
        "max": 24.0,
        "default": 8.0,
        "name": "每日睡眠时间（小时）",
        "help": "您平均每天睡多少小时？"
    },
    'disability': {
        "type": "categorical",
        "options": [0, 1],
        "name": "残疾状况",
        "options_desc": {0: "无残疾", 1: "有残疾"},
        "help": "您是否有残疾？"
    },
    'internet': {
        "type": "categorical",
        "options": [0, 1],
        "name": "是否使用互联网",
        "options_desc": {0: "不使用", 1: "使用"},
        "help": "您是否使用互联网？"
    },
    'hope': {
        "type": "categorical",
        "options": [1, 2, 3, 4],
        "name": "未来希望感",
        "options_desc": {1: "非常无望", 2: "无望", 3: "一般", 4: "充满希望"},
        "help": "您对未来是否充满希望？"
    },
    'fall_down': {
        "type": "categorical",
        "options": [0, 1],
        "name": "过去一年是否跌倒",
        "options_desc": {0: "否", 1: "是"},
        "help": "过去一年您是否跌倒过？"
    },
    'eyesight_close': {
        "type": "categorical",
        "options": [1, 2, 3, 4, 5],
        "name": "近距离视力",
        "options_desc": {1: "非常好", 2: "好", 3: "一般", 4: "差", 5: "非常差"},
        "help": "您的近距离视力如何？"
    },
    'hear': {
        "type": "categorical",
        "options": [1, 2, 3, 4, 5],
        "name": "听力状况",
        "options_desc": {1: "非常好", 2: "好", 3: "一般", 4: "差", 5: "非常差"},
        "help": "您的听力如何？"
    },
    'edu': {
        "type": "categorical",
        "options": [1, 2, 3, 4],
        "name": "教育水平",
        "options_desc": {1: "小学及以下", 2: "初中", 3: "高中/中专", 4: "大专及以上"},
        "help": "您的最高教育程度是？"
    },
    'pension': {
        "type": "categorical",
        "options": [0, 1],
        "name": "是否有养老金",
        "options_desc": {0: "无", 1: "有"},
        "help": "您是否有养老金？"
    },
    'pain': {
        "type": "categorical",
        "options": [0, 1],
        "name": "是否经常疼痛",
        "options_desc": {0: "否", 1: "是"},
        "help": "您是否经常感到疼痛？"
    },
}

st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature}({properties['min']} - {properties['max']})", 
            min_value=float(properties["min"]), 
            max_value=float(properties["max"]), 
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)", 
            options=properties["options"],
        )
    feature_values.append(value)

features = np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100


    text = f"Based on feature values, predicted possibility of XGB is {probability:.2f}%" 
    fig, ax = plt.subplots(figsize=(8,1))
    ax.text(
        0.5, 0.5, text, 
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes)
    ax.axis('off')
    st.pyplot(fig)


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))
    
    if isinstance(shap_values, list):

        shap_values_class = shap_values[predicted_class][0]
        expected_value = explainer.expected_value[predicted_class]
    else:

        shap_values_class = shap_values[0]
        expected_value = explainer.expected_value

    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

    plt.figure()
    shap_plot = shap.force_plot(
        expected_value,
        shap_values_class,
        feature_df,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())