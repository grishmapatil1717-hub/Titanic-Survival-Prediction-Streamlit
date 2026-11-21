import streamlit as st
import joblib
import pandas as pd

st.title("Titanic Survival Prediction")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

st.write("Welcome to Titanic Survival Prediction App!")
st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider("Age", 1, 80, 25)

fare = st.number_input(
    "Ticket Fare",
    min_value=0.0,
    value=30.0
)
if st.button("Predict Survival"):
    sex_value = 1 if sex == "Female" else 0

    input_data = pd.DataFrame(
        [[pclass, sex_value, age, fare]],
        columns=["pclass", "sex", "age", "fare"]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("Passenger is likely to survive!")
    else:
        st.metric("Survival Probability", f"{probability:.2%}")
        st.error("Passenger is unlikely to survive!")
