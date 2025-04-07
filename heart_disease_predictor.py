import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('XGB.pkl')

feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2]},
    'srh': {"type": "categorical", "options": [1,2,3,4,5]},
    'adlab_c': {"type": "categorical", "options": [0,1,2,3,4,5,6]},
    'arthre': {"type": "categorical", "options": [0, 1]},
    'digeste': {"type": "categorical", "options": [0, 1]},
    'retire': {"type": "categorical", "options": [0, 1]},
    'satlife': {"type": "categorical", "options": [1,2,3,4,5]},
    'sleep': {"type": "numerical", "min": 0.000, "max": 24.000, "default": 8.000},
    'disability': {"type": "categorical", "options": [0, 1]},
    'shangwang': {"type": "categorical", "options": [0, 1]},
    'hope': {"type": "categorical", "options": [1,2,3,4]},
    'fall_down': {"type": "categorical", "options": [0, 1]},
    'eyesight_close': {"type": "categorical", "options": [1,2,3,4,5]},
    'hear': {"type": "categorical", "options": [1,2,3,4,5]},
    'edu': {"type": "categorical", "options": [1,2,3,4]},
    'pension': {"type": "categorical", "options": [0, 1]},
    'tengtong': {"type": "categorical", "options": [0, 1]},
}

st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 收集特征值
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":  # 修正了拼写错误(原为"numerical")
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
    # 预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%" 
    fig, ax = plt.subplots(figsize=(8,1))
    ax.text(
        0.5, 0.5, text, 
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes)
    ax.axis('off')
    st.pyplot(fig)  # 直接使用st.pyplot()而不是保存图片

    # SHAP解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))
    
    # 处理SHAP输出的不同格式
    if isinstance(shap_values, list):
        # 多类别情况
        shap_values_class = shap_values[predicted_class][0]
        expected_value = explainer.expected_value[predicted_class]
    else:
        # 二分类情况
        shap_values_class = shap_values[0]
        expected_value = explainer.expected_value
    
    # 创建DataFrame用于显示
    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    # 创建force plot
    plt.figure()
    shap_plot = shap.force_plot(
        expected_value,
        shap_values_class,
        feature_df,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())  # 显示当前图形
    
    # 可选: 添加summary plot
    st.subheader("SHAP Summary Plot")
    plt.figure()
    shap.summary_plot(shap_values if isinstance(shap_values, list) else [shap_values],
                     feature_df,
                     plot_type="bar",
                     show=False)
    st.pyplot(plt.gcf())