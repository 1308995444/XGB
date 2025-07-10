import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载随机森林模型
model = joblib.load('RF.pkl')

# 特征定义（中文界面）
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2], "desc": "性别 (1:男, 2:女)"},
    'rural2': {"type": "categorical", "options": [1, 2], "desc": "户口类型 (1:农村, 2:城市)"},
    'diabe': {"type": "categorical", "options": [0, 1], "desc": "糖尿病 (0:无, 1:有)"},
    'lunge': {"type": "categorical", "options": [0, 1], "desc": "肺部疾病 (0:无, 1:有)"},
    'hearte': {"type": "categorical", "options": [0, 1], "desc": "心脏病 (0:无, 1:有)"},
    'dyslipe': {"type": "categorical", "options": [0, 1], "desc": "血脂异常 (0:无, 1:有)"},
    'hhcperc': {"type": "numerical", "min": 0.0, "max": 100.0, "default": 8.0, 
               "desc": "家庭年收入 (万元)", "step": 0.1, "format": "%.1f"},
    'retire': {"type": "categorical", "options": [0, 1], "desc": "退休状态 (0:未退休, 1:已退休)"},
    'sleep': {"type": "numerical", "min": 0.0, "max": 24.0, "default": 8.0, 
             "desc": "睡眠时长 (小时)", "step": 0.1, "format": "%.1f"},
    'edu': {"type": "categorical", "options": [1, 2, 3, 4], 
           "desc": "教育程度 (1:小学以下, 2:小学, 3:中学, 4:中学以上)"},
    'da040': {"type": "categorical", "options": [0, 1], "desc": "慢性疼痛 (0:无, 1:有)"},
    'pension': {"type": "categorical", "options": [0, 1], "desc": "养老保险 (0:无, 1:有)"},
}

# 界面布局
st.title("抑郁症风险预测模型 (带SHAP可视化)")
st.header("请输入以下特征值:")

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
if st.button("预测"):
    try:
        # 准备数据（使用DataFrame保持特征名称）
        feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
        
        # 执行预测
        predicted_class = model.predict(feature_df)[0]
        predicted_proba = model.predict_proba(feature_df)[0]
        probability = predicted_proba[predicted_class] * 100

        # 显示预测结果（优化中文显示）
        risk_level = "高风险" if predicted_class == 1 else "低风险"
        st.success(f"预测结果: {risk_level}")
        st.info(f"预测概率: {probability:.2f}%")
        
        # SHAP解释可视化
        st.subheader("特征影响分析")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_df)
        
        # 处理多分类情况
        if isinstance(shap_values, list):
            expected_value = explainer.expected_value[predicted_class]
            shap_values = shap_values[predicted_class]
        else:
            expected_value = explainer.expected_value

        # 新版SHAP力力图（兼容v0.20+）
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.plots.force(
            expected_value,
            shap_values[0],  # 取第一个样本的SHAP值
            feature_df.iloc[0],
            matplotlib=True,
            show=False,
            fig=fig
        )
        st.pyplot(fig)
        plt.close()
        
        # 添加特征重要性条形图
        st.subheader("特征重要性排序")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            feature_df,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig2)
        plt.close()
        
    except Exception as e:
        st.error(f"发生错误: {str(e)}")