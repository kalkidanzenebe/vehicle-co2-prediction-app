
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Vehicle CO2 Predictor",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# SESSION STATE
# ==========================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
    color: #F4F7FB;
}

[data-testid="stHeader"] {
    background-color: #0B1220;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.hero {
    background: linear-gradient(120deg, #111D33, #172D4A);
    border: 1px solid #00D4FF;
    border-radius: 18px;
    padding: 23px 15px;
    text-align: center;
    margin-bottom: 20px;
}

.hero h1 {
    color: #00D4FF;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero p {
    color: #DCE7F5;
    font-size: 15px;
    margin-bottom: 0;
}

h2, h3 {
    color: #00D4FF !important;
}

p, label, span {
    color: #F4F7FB;
}

[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    color: #00D4FF !important;
    font-weight: 600;
}

[data-testid="stNumberInput"] input {
    background-color: #18263A;
    color: #FFFFFF !important;
    border: 1px solid #35516F;
    border-radius: 8px;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #18263A;
    color: #FFFFFF;
    border: 1px solid #35516F;
    border-radius: 8px;
}

.panel {
    background-color: #14253B;
    border: 1px solid #35516F;
    border-radius: 15px;
    padding: 18px;
    min-height: 145px;
}

.result-card {
    background: linear-gradient(135deg, #14253B, #1B3852);
    border: 1px solid #00D4FF;
    border-radius: 15px;
    padding: 18px 10px;
    text-align: center;
    min-height: 145px;
}

.result-label {
    color: #AFC5DC;
    font-size: 13px;
    font-weight: 700;
}

.result-value {
    color: #00D4FF;
    font-size: 38px;
    font-weight: 800;
    margin: 10px 0;
}

.result-unit {
    color: #FFFFFF;
    font-size: 14px;
}

.waiting-card {
    background-color: #18263A;
    border: 1px dashed #54708C;
    border-radius: 15px;
    padding: 18px 10px;
    text-align: center;
    min-height: 145px;
}

.waiting-card p {
    color: #AFC5DC;
    font-size: 14px;
}

div.stButton > button {
    background-color: #00D4FF;
    color: #081321;
    border: none;
    border-radius: 10px;
    min-height: 52px;
    font-size: 15px;
    font-weight: 800;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #75E8FF;
    color: #081321;
}

.footer {
    text-align: center;
    color: #8FAAC5;
    padding: 15px;
    font-size: 12px;
}

hr {
    border-color: #263D59;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL SAFELY
# ==========================================
current_folder = Path(__file__).resolve().parent

possible_model_paths = [
    current_folder / "final_random_forest_model.pkl",
    current_folder / "models" / "final_random_forest_model.pkl",
    current_folder.parent / "models" / "final_random_forest_model.pkl",
    Path.cwd() / "final_random_forest_model.pkl",
    Path.cwd() / "models" / "final_random_forest_model.pkl"
]

model_path = None

for path in possible_model_paths:
    if path.is_file():
        model_path = path
        break

if model_path is None:
    st.error("Trained model file was not found.")

    st.markdown("""
    **Expected model filename:**

    `final_random_forest_model.pkl`

    Upload the trained model into your GitHub repository,
    or place it in the `models` folder.
    """)
    st.stop()


# Check whether model file is empty
file_size = os.path.getsize(model_path)

if file_size == 0:
    st.error("Your model file is empty (0 bytes).")
    st.warning(
        "Upload a fresh copy of final_random_forest_model.pkl "
        "from your Google Drive models folder."
    )
    st.stop()


# Load the model
try:
    model = joblib.load(model_path)

except EOFError:
    st.error("The model file is incomplete or corrupted.")
    st.warning(
        "Replace the existing model file with a fresh trained "
        "model file from Google Drive."
    )
    st.write("Model file:", str(model_path))
    st.write("File size:", f"{file_size:,} bytes")
    st.stop()

except Exception as error:
    st.error("Model loading failed.")
    st.code(repr(error))
    st.write("Model file:", str(model_path))
    st.write("File size:", f"{file_size:,} bytes")
    st.stop()


# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="hero">
    <h1>🚗 VEHICLE CO₂ PREDICTOR</h1>
    <p>
        Intelligent Vehicle Emissions Prediction
        | Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# VEHICLE INPUTS
# ==========================================
st.markdown("### Vehicle Specifications")

st.write(
    "Select your vehicle details below to estimate "
    "its carbon dioxide emissions."
)

col1, col2 = st.columns(2)

with col1:

    vehicle_class = st.selectbox(
        "Vehicle Class",
        [
            "COMPACT",
            "SUV - SMALL",
            "SUV - STANDARD",
            "MID-SIZE",
            "FULL-SIZE",
            "TWO-SEATER",
            "MINICOMPACT",
            "SUBCOMPACT",
            "PICKUP TRUCK - SMALL",
            "PICKUP TRUCK - STANDARD",
            "MINIVAN",
            "VAN - CARGO",
            "VAN - PASSENGER",
            "STATION WAGON - SMALL",
            "STATION WAGON - MID-SIZE",
            "SPECIAL PURPOSE VEHICLE"
        ]
    )

    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=0.0,
        value=2.0,
        step=0.1,
        format="%.1f"
    )

    cylinders = st.selectbox(
        "Number of Cylinders",
        [3, 4, 5, 6, 8, 10, 12, 16]
    )

with col2:

    transmission = st.selectbox(
        "Transmission",
        [
            "A4", "A5", "A6", "A7", "A8", "A9", "A10",
            "AM5", "AM6", "AM7", "AM8", "AM9",
            "AS4", "AS5", "AS6", "AS7", "AS8", "AS9",
            "AS10", "AV", "AV6", "AV7", "AV8", "AV10",
            "M5", "M6", "M7"
        ]
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["X", "Z", "D", "E", "N"]
    )


# ==========================================
# BUTTON AND RESULT SIDE BY SIDE
# ==========================================
st.markdown("---")
st.markdown("### CO₂ Emissions Estimation")

button_col, result_col = st.columns(
    [1, 1.4],
    gap="medium"
)

# LEFT PANEL: REAL BUTTON
with button_col:

    st.markdown("""
    <div class="panel">
        <p style="color:#00D4FF;font-weight:700;font-size:16px;">
            VEHICLE EMISSIONS ANALYSIS
        </p>
        <p style="color:#DCE7F5;font-size:13px;">
            Click below to calculate the estimated CO₂ emissions
            for your selected vehicle.
        </p>
    </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button(
        "🚗 PREDICT CO₂ EMISSIONS",
        use_container_width=True
    )


# PREDICTION
if predict_clicked:

    input_data = pd.DataFrame([{
        "Vehicle Class": vehicle_class,
        "Engine Size(L)": engine_size,
        "Cylinders": cylinders,
        "Transmission": transmission,
        "Fuel Type": fuel_type
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.session_state.prediction_result = float(prediction)
        st.session_state.prediction_error = None

    except Exception as error:
        st.session_state.prediction_result = None
        st.session_state.prediction_error = repr(error)


# RIGHT PANEL: RESULT
with result_col:

    if st.session_state.prediction_error is not None:

        st.error("Prediction failed.")
        st.code(st.session_state.prediction_error)

    elif st.session_state.prediction_result is not None:

        prediction = st.session_state.prediction_result

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">
                ESTIMATED CO₂ EMISSIONS
            </div>

            <div class="result-value">
                {prediction:.2f}
            </div>

            <div class="result-unit">
                grams of CO₂ per kilometre (g/km)
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="waiting-card">
            <p>ESTIMATED CO₂ EMISSIONS</p>
            <p style="
                color:#00D4FF;
                font-size:32px;
                font-weight:800;
                margin:10px 0;
            ">
                -- g/km
            </p>
            <p>Your prediction will appear here.</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<hr>
<div class="footer">
    VEHICLE CO₂ PREDICTOR | MACHINE LEARNING PROJECT
</div>
""", unsafe_allow_html=True)
