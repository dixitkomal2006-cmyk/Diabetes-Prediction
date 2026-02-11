import streamlit as st
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

st.title("🩺 Diabetes Progression Prediction")

# ---------------- Load Dataset ----------------
diabetes = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data,
    diabetes.target,
    test_size=0.2,
    random_state=42
)

# ---------------- Model ----------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- Metrics ----------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"### Mean Squared Error: **{mse:.2f}**")
st.write(f"### R² Score: **{r2:.2f}**")

# ---------------- Create DataFrame for Graph ----------------
results_df = pd.DataFrame({
    "Actual Values": y_test,
    "Predicted Values": y_pred
})

st.subheader("Actual vs Predicted Values")

st.line_chart(results_df)

# ---------------- BMI Feature Analysis ----------------
st.subheader("BMI Feature vs Target")

bmi_index = list(diabetes.feature_names).index("bmi")

bmi_df = pd.DataFrame({
    "BMI": X_test[:, bmi_index],
    "Predicted Progression": y_pred
})

st.scatter_chart(bmi_df)
