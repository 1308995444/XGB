import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

features = np.array([feature_values])

# 预测与解释
if st.button("Predict"):
    try:
        # 预测结果
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        probability = predicted_proba[predicted_class] * 100

        # 结果显示
        risk_text = 'High risk' if predicted_class == 1 else 'Low risk'
        st.markdown(f"## Prediction Result: **{risk_text}**")
        st.markdown(f"### Probability: **{probability:.2f}%**")

        # SHAP解释
        explainer = shap.TreeExplainer(model)
        feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
        shap_values = explainer.shap_values(feature_df)

        # 处理多分类和单分类情况
        if isinstance(shap_values, list):
            # 多分类情况
            shap_values_class = shap_values[predicted_class]
            expected_value = explainer.expected_value[predicted_class]
        else:
            # 单分类情况
            shap_values_class = shap_values
            expected_value = explainer.expected_value

        # SHAP force plot
        st.subheader("SHAP Explanation (How each feature contributes to the prediction)")
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.force_plot(
            expected_value,
            shap_values_class[0],  # 取第一个样本的解释
            feature_df.iloc[0],   # 取第一个样本的特征值
            matplotlib=True,
            show=False,
            plot_cmap=["#77dd77", "#ff6961"],  # 绿色和红色
            text_rotation=15,
            fig=fig
        )
        st.pyplot(fig, bbox_inches='tight')
        plt.close()

        # 可选：添加条形图解释
        st.subheader("Feature Importance")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values_class,
            feature_df,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig2)
        plt.close()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")