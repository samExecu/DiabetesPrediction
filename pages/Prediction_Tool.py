import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Prediction Tool")

st.title("Diabetes Risk Calculator")

# --- 1. Load and Train Model (Cached for Speed) ---
@st.cache_resource
def load_and_train_model():
    try:
        # Load Data
        df = pd.read_csv('data/diabetes_prediction_dataset.csv')

        # Simple Preprocessing (Encoding gender/smoking not strictly needed for the 4 key features)
        # We focus on the 4 key features identified in Week 4: Age, BMI, HbA1c, Glucose
        X = df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
        y = df['diabetes']

        # Train Model (Random Forest is usually best for accuracy)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Calculate accuracy for display
        acc = accuracy_score(y_test, model.predict(X_test))

        return model, acc
    except FileNotFoundError:
        return None, None

model, accuracy = load_and_train_model()

if model is None:
    st.error(" Error: 'diabetes_prediction_dataset.csv' not found. Please put the CSV file in the main folder.")
else:
    st.success(f" Model Trained Successfully! (Accuracy: {accuracy*100:.2f}%)")

    st.divider()

    # --- 2. User Input Form ---
    st.sidebar.header("Patient Input Features")

    def user_input_features():
        age = st.sidebar.slider('Age', 0, 100, 30)
        bmi = st.sidebar.number_input('BMI (Body Mass Index)', 10.0, 60.0, 27.3)

        st.sidebar.markdown("---")
        st.sidebar.write("**Clinical Levels:**")

        # HbA1c Input
        hba1c = st.sidebar.slider('HbA1c Level', 3.0, 9.0, 5.5)
        st.sidebar.caption("Normal: < 5.7 | Pre-diabetes: 5.7–6.4 | Diabetes: > 6.5")

        # Glucose Input
        glucose = st.sidebar.number_input('Blood Glucose Level', 50, 300, 100)
        st.sidebar.caption("Normal: < 140 mg/dL")

        data = {'age': age,
                'bmi': bmi,
                'HbA1c_level': hba1c,
                'blood_glucose_level': glucose}
        return pd.DataFrame(data, index=[0])

    # Get inputs
    input_df = user_input_features()

    # Display User Inputs
    st.subheader("1. Patient Data")
    st.write(input_df)

    # --- 3. Prediction ---
    if st.button("Predict Diabetes Risk"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("2. Prediction Result")

        # Logic for result display
        is_diabetic = prediction[0] == 1
        probability = prediction_proba[0][1] * 100

        if is_diabetic:
            st.error(f"**Result: POSITIVE (High Risk)**")
            st.write(f"The model predicts a **{probability:.1f}%** probability of diabetes.")
            st.warning("Recommendation: Please consult a healthcare professional immediately.")
        else:
            st.success(f"**Result: NEGATIVE (Low Risk)**")
            st.write(f"The model predicts a **{probability:.1f}%** probability of diabetes.")
            st.balloons()
