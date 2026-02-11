
import streamlit as st
import pandas as pd
import pickle
import sklearn

# -------------------------------------------------
# App title
# -------------------------------------------------
st.title("🚢 Titanic Survival Prediction (Logistic Regression)")

# -------------------------------------------------
# Show sklearn version (debug safety)
# -------------------------------------------------
st.write("Sklearn version:", sklearn.__version__)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
st.header("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# -------------------------------------------------
# Create input dataframe (MUST MATCH TRAINING DATA)
# -------------------------------------------------
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"🎉 Passenger is likely to SURVIVE (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Passenger is NOT likely to survive (Probability: {probability:.2f})")
