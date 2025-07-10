import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 忽略版本警告
warnings.filterwarnings("ignore", category=UserWarning)

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

# 界面布局
st.title("抑郁症风险预测模型")
st.markdown("---")

# 输入表单
st.header("请输入特征值")
feature_values = []
cols = st.columns(2)  # 创建两列布局

for i, (feature, props) in enumerate(feature_ranges.items()):
    with cols[i % 2]:  # 交替分布在两列中
        if props["type"] == "numerical":
            value = st.number_input(
                props["desc"],
                min_value=float(props["min"]),
                max_value=float(props["max"]),
                value=float(props["default"]),
                step=props.get("step", 1.0),
                format=props.get("format", "%f")
            )
        else:
            value = st.selectbox(
                props["desc"],
                options=props["options"],
                index=0
            )
        feature_values.append(value)

# 预测按钮
if st.button("开始预测", type="primary"):
    with st.spinner("正在计算..."):
        try:
            # 准备数据
            feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
            
            # 执行预测
            predicted_class = model.predict(feature_df)[0]
            predicted_proba = model.predict_proba(feature_df)[0]
            probability = predicted_proba[predicted_class] * 100
            
            # 显示预测结果
            st.markdown("---")
            st.subheader("预测结果")
            if predicted_class == 1:
                st.error(f"高风险 (概率: {probability:.1f}%)")
            else:
                st.success(f"低风险 (概率: {100-probability:.1f}%)")
            
            # SHAP解释
            st.markdown("---")
            st.subheader("特征影响分析")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feature_df)
            
            # 处理多分类情况
            if isinstance(shap_values, list):
                expected_value = explainer.expected_value[predicted_class]
                shap_values_plot = shap_values[predicted_class]
            else:
                expected_value = explainer.expected_value
                shap_values_plot = shap_values
            
            # 力力图
            st.markdown("#### 单个特征贡献")
            plt.figure(figsize=(10, 3))
            shap.plots.force(
                expected_value,
                shap_values_plot[0],  # 第一个样本
                feature_df.iloc[0],
                matplotlib=True,
                show=False
            )
            st.pyplot(plt.gcf(), bbox_inches='tight')
            plt.close()
            
            # 特征重要性
            st.markdown("#### 特征重要性排序")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values_plot,
                feature_df,
                plot_type="bar",
                show=False
            )
            st.pyplot(plt.gcf(), bbox_inches='tight')
            plt.close()
            
            # 详细SHAP值表格
            st.markdown("#### 详细特征贡献值")
            contrib_df = pd.DataFrame({
                "特征": feature_ranges.keys(),
                "特征值": feature_values,
                "SHAP值": shap_values_plot[0]
            }).sort_values("SHAP值", ascending=False)
            st.dataframe(contrib_df.style.background_gradient(cmap="RdBu", subset=["SHAP值"]))
            
        except Exception as e:
            st.error(f"预测过程中发生错误: {str(e)}")

# 添加说明
st.markdown("---")
st.info("""
**使用说明：**
1. 填写/选择所有特征值
2. 点击"开始预测"按钮
3. 查看预测结果和特征影响分析
""")