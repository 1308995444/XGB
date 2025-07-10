import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
try:
    font_path = "SimHei.ttf"  # 或者您系统中可用的中文字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except:
    st.warning("中文字体设置失败，可能显示为方框")

# 模型加载
model = joblib.load('RF.pkl')

# 特征定义
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2], "desc": "性别/Gender (1:男/Male, 2:女/Female)"},
    'rural2': {"type": "categorical", "options": [1,2], "desc": "户口类型/Gender (1:农村, 2:城市)"},
    'diabe': {"type": "categorical", "options": [0,1], "desc": "糖尿病/Arthritis (0:无/No, 1:有/Yes)"},
    'lunge': {"type": "categorical", "options": [0, 1], "desc": "肺部疾病/Arthritis (0:无/No, 1:有/Yes)"},
    'hearte': {"type": "categorical", "options": [0, 1], "desc": "心脏病/Digestive issues (0:无/No, 1:有/Yes)"},
    'dyslipe': {"type": "categorical", "options": [0, 1], "desc": "退休状态/Retirement status (0:未退休/Not retired, 1:已退休/Retired)"},
    'hhcperc': {"type": "numerical", "min": 0.0, "max": 100.0, "default": 8.0, "desc": "家庭年收入/Sleep duration (万元)","step": 0.1,"format": "%.1f"},
    'retire': {"type": "categorical", "options": [0, 1], "desc": "退休状态/Retirement status (0:未退休/Not retired, 1:已退休/Retired)"},
    'sleep': {"type": "numerical", "min": 0.0, "max": 24.0, "default": 8.0, "desc": "睡眠时长/Sleep duration (小时/hours)","step": 0.1,"format": "%.1f"},
    'edu': {"type": "categorical", "options": [1,2,3,4], "desc": "教育程度/Education level (1:小学以下/Below Primary, 2:小学/Primary, 3:中学/Secondary, 4:中学以上/Above Secondary)"},
    'da040': {"type": "categorical", "options": [0, 1], "desc": "慢性疼痛/Chronic pain (0:无/No, 1:有/Yes)"},
    'pension': {"type": "categorical", "options": [0, 1], "desc": "养老保险/Pension (0:无/No, 1:有/Yes)"},
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
    # 确保特征名称与模型训练时一致
    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    predicted_class = model.predict(feature_df)[0]
    predicted_proba = model.predict_proba(feature_df)[0]
    probability = predicted_proba[predicted_class] * 100

    # 结果显示
    text_en = f"预测概率: {probability:.2f}% ({'高风险' if predicted_class == 1 else '低风险'})"
    fig, ax = plt.subplots(figsize=(10,2))
    ax.text(0.5, 0.7, text_en, 
            fontsize=14, ha='center', va='center')
    ax.axis('off')
    st.pyplot(fig)

    # SHAP解释 - 使用新API
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(feature_df)
    
    # 对于分类模型，选择正确的类别
    if len(shap_values.shape) == 3:  # 多分类
        shap_values_class = shap_values[..., predicted_class]
        expected_value = explainer.expected_value[predicted_class]
    else:  # 二分类
        shap_values_class = shap_values
        expected_value = explainer.expected_value[predicted_class]
    
    # 使用新的force_plot API
    st.subheader("SHAP解释")
    fig, ax = plt.subplots()
    shap.plots.force(expected_value, 
                    shap_values_class[0,:], 
                    feature_df.iloc[0,:],
                    matplotlib=True,
                    show=False,
                    ax=ax)
    st.pyplot(fig)