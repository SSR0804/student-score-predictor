
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Score Predictor", page_icon="📈")
st.title("📈 Student Score Predictor")
st.write("Predict exam score from study hours using a simple linear regression model.")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "linear_model.joblib"

# Fallback training data if model not found
data = {
    "hours": [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0],
    "score": [37,42,45,49,52,56,57,61,65,66,69,72,75,78,82,85,88,91,94]
}

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    df = pd.DataFrame(data)
    X = df[["hours"]].values
    y = df["score"].values
    model = LinearRegression().fit(X, y)

hours = st.slider("Enter study hours:", min_value=0.0, max_value=12.0, value=5.0, step=0.5)
pred = float(model.predict(np.array([[hours]]))[0])
st.metric(label="Predicted Score", value=f"{pred:.1f}")

st.caption("Model: Scikit-learn LinearRegression. If trained model exists in artifacts, it is used; otherwise a fallback model is built.")
