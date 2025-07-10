import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('RF.pkl')

model = load_model()

# 特征定义
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

# 输入表单
st.title("SHAP特征影响分析")
feature_values = []
for feature, props in feature_ranges.items():
    if props["type"] == "numerical":
        value = st.number_input(props["desc"], 
                              min_value=float(props["min"]),
                              max_value=float(props["max"]),
                              value=float(props["default"]),
                              step=props.get("step", 1.0))
    else:
        value = st.selectbox(props["desc"], options=props["options"])
    feature_values.append(value)

if st.button("分析特征影响"):
    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    
    # 处理多分类情况
    if isinstance(shap_values, list):
        # 如果是多分类模型，默认显示第一个类别的解释
        expected_value = explainer.expected_value[0]
        shap_values_plot = shap_values[0]
    else:
        # 二分类或回归模型
        expected_value = explainer.expected_value
        shap_values_plot = shap_values
    
    # 绘制SHAP力图
    st.subheader("特征贡献力图示")
    plt.figure(figsize=(10, 3))
    shap.plots.force(
        base_value=expected_value,
        shap_values=shap_values_plot[0],
        features=feature_df.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    plt.close()