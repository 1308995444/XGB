import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

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
st.title("抑郁症风险预测模型 (含SHAP可视化)")
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
    # 准备数据
    features = np.array([feature_values])
    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    # 执行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    result_text = f"预测概率: {probability:.2f}% ({'高风险' if predicted_class == 1 else '低风险'})"
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.text(0.5, 0.7, result_text, 
            fontsize=14, ha='center', va='center', fontname='SimHei')  # 使用中文黑体
    ax.axis('off')
    st.pyplot(fig)

    # SHAP解释可视化
    st.subheader("特征影响分析")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    
    # 处理SHAP值（适应随机森林和多分类情况）
    if isinstance(shap_values, list):
        shap_values_class = shap_values[predicted_class][0]
        expected_value = explainer.expected_value[predicted_class]
    else:
        shap_values_class = shap_values[0]
        expected_value = explainer.expected_value

    # 绘制SHAP力力图
    plt.figure()
    shap.force_plot(
        expected_value,
        shap_values_class,
        feature_df.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    
    # 添加SHAP摘要图（可选）
    st.subheader("特征重要性摘要")
    shap.summary_plot(shap_values, feature_df, plot_type="bar", show=False)
    st.pyplot(plt.gcf())