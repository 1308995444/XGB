import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as pyp

model= joblib.load('XGB.pkl')

feature_ranges={
    'gender':{"type": "categorical", "options": [1, 2]},
    'srh':{"type": "categorical", "options": [1,2,3,4,5]},
    'adlab_c':{"type": "categorical", "options": [0,1,2,3,4,5,6]},
    'arthre':{"type": "categorical", "options": [0, 1]},
    'digeste':{"type": "categorical", "options": [0, 1]},
    'retire':{"type": "categorical", "options": [0, 1]},
    'satlife':{"type": "categorical", "options": [1,2,3,4,5]},
    'sleep':{"type":"numerical","min":0.000,"max":24.000,"default":8.000},
    'disability':{"type": "categorical", "options": [0, 1]},
    'shangwang':{"type": "categorical", "options": [0, 1]},
    'hope':{"type": "categorical", "options": [1,2,3,4]},
    'fall_down':{"type": "categorical", "options": [0, 1]},
    'eyesight_close':{"type": "categorical", "options": [1,2,3,4,5]},
    'hear':{"type": "categorical", "options": [1,2,3,4,5]},
    'edu':{"type": "categorical", "options": [1,2,3,4]},
    'pension':{"type": "categorical", "options": [0, 1]},
    'tengtong':{"type": "categorical", "options": [0, 1]},
}
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")
feature_values=[]
for feature, properties in feature_ranges.items():
    if properties["type"]== "numerical": 
        value = st.number_input(
            label=f"{feature}({properties['min']} -{properties['max']})", 
            min_value=float(properties["min"]), 
            max_value=float(properties["max"]), 
            value=float(properties["default"]),
        )
    elif properties["type"]== "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)", 
            options=properties["options"],
        )
    feature_values.append(value)
feature= np.array([feature_values])

if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_c1ass]*100


text = f"Based on feature values, predicted possibility of AKI is {probabi1ity:.2f}%" 
fig,ax = pit.subplots(figsize=(8,1))
ax.text(
    0.5,0.5,text, 
    fontsize=16,
    ha='center', va='center',	
    fontname='Times New Roman',
    transform=ax.transAxes
)

ax.axis('off')
p1t.savefig("prediction_text.png",bbox_inches='tight', dpi=300)
st.image("prediction_text.png")

explainer = shap.TreeExp1ainer(mode1)
shap_va1ues = exp1ainer.shap_va1ues(pd.DataFrame([feature_values],columns=feature_ranges.keys()))

class_index = predicted_class
shap_fig = shap,force_p1ot(
    exp1ainer.expected_va1ue[c1ass_index], 
    shap_values[:,:,c1ass_index],
    pd.DataFrame([feature_values],columns=feature_ranges.keys()),
    matp1ot1ib=True,
    )
p1t.savefig("shap_force_p1ot.png",bbox_inches='tight', dpi=1200)
st.image("shap_force_p1ot.png")