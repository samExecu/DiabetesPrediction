import streamlit as st

# Page Config
st.set_page_config(
    page_title="Diabetes Prediction | Home",
    layout="centered"
)

# Title and Team Info
st.title("Diabetes Risk Prediction System")
st.subheader("DATA 200 Project")
st.write("**Team:** The Three Musketeers")

st.divider()

# Project Description
st.markdown("""
### Project Overview
Diabetes is a chronic health condition that affects how your body turns food into energy.
Early detection is key to managing the disease and preventing serious complications.

This application uses **Machine Learning (Logistic Regression & Random Forest)**
to predict the likelihood of diabetes based on diagnostic measures.

### How to use this app:
1. **Navigate to the 'Prediction' page** via the sidebar.
2. Enter your health details (Age, BMI, HbA1c, Glucose).
3. Get an instant risk assessment.

### Data Source
We utilized the **Diabetes Prediction Dataset** from Kaggle, consisting of comprehensive
clinical records including age, body mass index (BMI), blood glucose levels, and HbA1c levels.
""")

st.info(" **Disclaimer:** This tool is for educational purposes only and does not substitute professional medical advice.")
