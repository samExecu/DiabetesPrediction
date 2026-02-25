import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Dashboard")

st.title("Exploratory Data Analysis")
st.markdown("""
This section visualizes the structure of the dataset and identifies the key relationships
between health factors (like Glucose and Age) and Diabetes.
""")

# Loading the data
@st.cache_data
def load_data():
    try:
        # Try main folder first, then data folder
        return pd.read_csv('diabetes_prediction_dataset.csv')
    except:
        return pd.read_csv('data/diabetes_prediction_dataset.csv')

df = load_data()

if df is not None:
    # Show Raw Data Toggle
    if st.checkbox("Show Raw Data Preview"):
        st.write(df.head())
        st.caption(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    st.divider()

    # --- 1. Target Distribution ---
    st.subheader("1. Distribution of Diabetes Cases")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.countplot(x='diabetes', data=df, palette='viridis', ax=ax1)
    ax1.set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'])
    st.pyplot(fig1)

    st.info("""
    **Observation:** The dataset is imbalanced. There are significantly more healthy patients
    (Class 0) than diabetic patients (Class 1). This is expected in real-world medical data,
    but it means our model must be careful not to just guess "Healthy" for everyone.
    """)

    st.divider()

    # --- 2. Correlation Matrix ---
    st.subheader("2. Correlation Heatmap")

    # Encode for correlation just for this view
    df_encoded = df.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Handle columns if they exist
    if 'gender' in df_encoded.columns:
        df_encoded['gender'] = le.fit_transform(df_encoded['gender'])
    if 'smoking_history' in df_encoded.columns:
        df_encoded['smoking_history'] = le.fit_transform(df_encoded['smoking_history'])

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.markdown("""
    **Key Insights:**
    *   **Strongest Predictors:** `blood_glucose_level` (0.42) and `HbA1c_level` (0.40) show the highest positive correlation with Diabetes.
    *   **Secondary Predictors:** `age` (0.26) and `bmi` (0.21) are also significant factors.
    *   **Low Impact:** `gender` and `smoking_history` have very weak correlations in this specific dataset.
    """)

    st.divider()

    # --- 3. Boxplots (Glucose) ---
    st.subheader("3. Blood Glucose Levels by Diagnosis")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='diabetes', y='blood_glucose_level', data=df, palette='Set2', ax=ax3)
    ax3.set_xticklabels(['No Diabetes', 'Diabetes'])
    st.pyplot(fig3)

    st.info("""
    **Observation:** There is a distinct separation between the groups.
    Diabetic patients consistently have Blood Glucose levels above **140 mg/dL**.
    This confirms that Glucose is a critical feature for our machine learning model.
    """)

else:
    st.error("File 'diabetes_prediction_dataset.csv' not found. Please check your folder structure.")
