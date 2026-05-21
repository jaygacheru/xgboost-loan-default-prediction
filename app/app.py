# Import libraries
import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("models/final_xgboost_credit_model.pkl")

# Load selected features
selected_features = joblib.load("models/selected_features.pkl")

# Load best threshold
best_threshold = joblib.load("models/best_threshold.pkl")

# App title
st.title("Loan Default Prediction App")

# App description
st.write("Enter applicant details to predict default risk.")

# User inputs
income = st.number_input("Total Income", min_value=0.0, value=100000.0)
credit = st.number_input("Loan Amount", min_value=0.0, value=500000.0)
annuity = st.number_input("Loan Annuity", min_value=0.0, value=25000.0)
goods_price = st.number_input("Goods Price", min_value=0.0, value=450000.0)

age_years = st.number_input("Age in Years", min_value=18, max_value=80, value=30)
employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=5)

ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.5)
ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5)
ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5)

late_payment_rate = st.slider("Late Payment Rate", 0.0, 1.0, 0.0)
payment_delay_mean = st.number_input("Average Payment Delay", value=0.0)

# Create blank input with all selected features
input_data = pd.DataFrame(
    columns=selected_features
)

# Add one row filled with zeros
input_data.loc[0] = 0

# Fill known features if they exist
if "AMT_INCOME_TOTAL" in input_data.columns:
    input_data["AMT_INCOME_TOTAL"] = income

if "AMT_CREDIT" in input_data.columns:
    input_data["AMT_CREDIT"] = credit

if "AMT_ANNUITY" in input_data.columns:
    input_data["AMT_ANNUITY"] = annuity

if "AMT_GOODS_PRICE" in input_data.columns:
    input_data["AMT_GOODS_PRICE"] = goods_price

if "DAYS_BIRTH" in input_data.columns:
    input_data["DAYS_BIRTH"] = -age_years * 365

if "DAYS_EMPLOYED" in input_data.columns:
    input_data["DAYS_EMPLOYED"] = -employment_years * 365

if "EXT_SOURCE_1" in input_data.columns:
    input_data["EXT_SOURCE_1"] = ext_source_1

if "EXT_SOURCE_2" in input_data.columns:
    input_data["EXT_SOURCE_2"] = ext_source_2

if "EXT_SOURCE_3" in input_data.columns:
    input_data["EXT_SOURCE_3"] = ext_source_3

if "REPAY_LATE_PAYMENT_MEAN" in input_data.columns:
    input_data["REPAY_LATE_PAYMENT_MEAN"] = late_payment_rate

if "REPAY_PAYMENT_DELAY_MEAN" in input_data.columns:
    input_data["REPAY_PAYMENT_DELAY_MEAN"] = payment_delay_mean

# Predict button
if st.button("Predict Default Risk"):

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]

    # Apply best threshold
    prediction = int(probability >= best_threshold)

    # Show probability
    st.write(f"Default Probability: {probability:.2%}")

    # Show result
    if prediction == 1:
        st.error("High Risk: Applicant is likely to default.")
    else:
        st.success("Low Risk: Applicant is not likely to default.")
