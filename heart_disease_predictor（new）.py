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
    # 预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 创建自定义可视化图像
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  # 隐藏坐标轴
    
    # 添加标题
    ax.text(0.5, 0.9, "Basic value", fontsize=16, ha='center', va='center', fontweight='bold')
    
    # 添加预测概率部分 - 这里替换原来的1.69为实际预测概率
    ax.text(0.5, 0.7, "Higher = lower", fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.65, f"{probability:.2f}%",  # 显示预测概率
           fontsize=24, ha='center', va='center', color='red', fontweight='bold')
    
    # 添加类型信息
    ax.text(0.5, 0.5, "Type of type", fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.45, "- type\n- style\n- size", 
           fontsize=12, ha='center', va='center')
    
    # 添加图形代码信息
    ax.text(0.5, 0.35, "Graphic code = 1.0", fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.3, "[hope = 1.0] [version = 0.0] [str = 1.0]", 
           fontsize=10, ha='center', va='center')
    
    # 添加分割器代码信息
    ax.text(0.5, 0.2, "Splitter code = 1.0", fontsize=12, ha='center', va='center')
    ax.text(0.5, 0.15, "[step = 8.0] [gender = 1.0] [edu = 1.0] [waytime = 0.0delab < -0.0]", 
           fontsize=10, ha='center', va='center')
    
    # 显示自定义图像
    st.pyplot(fig)

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
    shap.force_plot(
        expected_value,
        shap_values_class,
        feature_df,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())  # 显示SHAP解释图