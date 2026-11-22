
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Vehicle CO₂ Predictor",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# AUTOMOTIVE DASHBOARD DESIGN
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
        padding: 25px 18px;
        text-align: center;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #00D4FF;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        color: #DCE7F5;
        font-size: 16px;
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

    [data-testid="stForm"] {
        background-color: #111D30;
        padding: 18px;
        border: 1px solid #263D59;
        border-radius: 15px;
    }

    div[data-testid="stFormSubmitButton"] > button {
        background-color: #00D4FF;
        color: #081321;
        border: none;
        border-radius: 10px;
        min-height: 48px;
        font-size: 16px;
        font-weight: 800;
        width: 100%;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #75E8FF;
        color: #081321;
    }

    .result-card {
        background: linear-gradient(135deg, #14253B, #1B3852);
        border: 1px solid #00D4FF;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        margin-top: 10px;
    }

    .result-label {
        color: #AFC5DC;
        font-size: 16px;
        font-weight: 600;
    }

    .result-value {
        color: #00D4FF;
        font-size: 44px;
        font-weight: 800;
        margin: 10px 0;
    }

    .result-unit {
        color: #FFFFFF;
        font-size: 17px;
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
# LOAD TRAINED RANDOM FOREST MODEL
# ==========================================
current_folder = Path(__file__).resolve().parent

possible_model_paths = [
    current_folder / "final_random_forest_model.pkl",
    current_folder / "../models/final_random_forest_model.pkl",
    current_folder / "models/final_random_forest_model.pkl",
    Path.cwd() / "final_random_forest_model.pkl"
]

model_path = next(
    (path for path in possible_model_paths if path.exists()),
    None
)

if model_path is None:
    st.error(
        "Model file not found. Please check that "
        "final_random_forest_model.pkl is included in your repository."
    )
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
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
# VEHICLE INPUT SECTION
# ==========================================
st.markdown("### Vehicle Specifications")

st.write(
    "Enter your vehicle information below "
    "to estimate its carbon dioxide emissions."
)


# ==========================================
# INPUT FORM
# ==========================================
with st.form("vehicle_prediction_form"):

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

    submitted = st.form_submit_button(
        "🚗 PREDICT CO₂ EMISSIONS",
        use_container_width=True
    )


# ==========================================
# CO₂ EMISSIONS PREDICTION
# ==========================================
if submitted:

    input_data = pd.DataFrame([{
        "Vehicle Class": vehicle_class,
        "Engine Size(L)": engine_size,
        "Cylinders": cylinders,
        "Transmission": transmission,
        "Fuel Type": fuel_type
    }])

    try:

        # Original trained model prediction
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.markdown("### Prediction Result")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">
                PREDICTED CO₂ EMISSIONS
            </div>

            <div class="result-value">
                {prediction:.2f}
            </div>

            <div class="result-unit">
                grams of CO₂ per kilometre (g/km)
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<hr>
<div class="footer">
    VEHICLE CO₂ PREDICTOR | MACHINE LEARNING PROJECT
</div>
""", unsafe_allow_html=True)
