import streamlit as st
import pandas as pd
import pickle
import sklearn

st.title("🚢 Titanic Survival Prediction")
st.write("Sklearn version:", sklearn.__version__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# User inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.number_input("Age", min_value=0.0, value=30.0)
SibSp = st.number_input("Siblings/Spouses", min_value=0, value=0)
Parch = st.number_input("Parents/Children", min_value=0, value=0)
Fare = st.number_input("Fare", min_value=0.0, value=32.0)
Embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# EXACT same columns as training
input_df = pd.DataFrame([{
    "Pclass": Pclass,
    "Sex": Sex,
    "Age": Age,
    "SibSp": SibSp,
    "Parch": Parch,
    "Fare": Fare,
    "Embarked": Embarked
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")