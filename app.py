
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("final_random_forest_model.pkl")

st.title("Vehicle CO2 Emissions Prediction")
st.write("Predict vehicle CO2 emissions using machine learning.")

vehicle_class = st.selectbox("Vehicle Class", [
    "Compact", "Mid-size", "Full-size", "SUV - Small",
    "SUV - Standard", "Pickup Truck - Small",
    "Pickup Truck - Standard", "Minivan", "Van - Cargo",
    "Van - Passenger", "Two-seater",
    "Station wagon - Small", "Station wagon - Mid-size"
])

engine_size = st.number_input("Engine Size (L)", 0.5, 8.0, 2.0, 0.1)

cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12, 16])

transmission = st.selectbox("Transmission", [
    "A4", "A5", "A6", "A7", "A8", "A9", "A10",
    "AM5", "AM6", "AM7", "AM8", "AS4", "AS5",
    "AS6", "AS7", "AS8", "AS9", "AS10", "AV",
    "AV6", "AV7", "AV8", "AV10", "M5", "M6", "M7"
])

fuel_type = st.selectbox("Fuel Type", ["X", "Z", "E", "D", "N"])

if st.button("Predict CO2 Emissions"):
    input_data = pd.DataFrame([{
        "Vehicle Class": vehicle_class,
        "Engine Size(L)": engine_size,
        "Cylinders": cylinders,
        "Transmission": transmission,
        "Fuel Type": fuel_type
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted CO2 Emissions: {prediction:.2f} g/km")
