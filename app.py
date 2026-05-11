import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ======================
# LOAD MODEL
# ======================
model = joblib.load("model/random_forest.pkl")
encoder = joblib.load("model/location_encoder.pkl")

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="R-Forest YogyaHouse",
    page_icon="🏠",
    layout="wide"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.big-font {
    font-size:28px !important;
    font-weight:bold;
    color:#1e293b;
}

.card {
    background-color:white;
    padding:20px;
    border-radius:15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
}

.metric-card {
    background: white;
    padding:15px;
    border-radius:12px;
    text-align:center;
    box-shadow: 0 3px 6px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=120
)

st.sidebar.title("R-Forest YogyaHouse")
st.sidebar.markdown("---")

st.sidebar.info("""
**Prediksi Harga Rumah Yogyakarta**

Model:
Random Forest Regressor

Dataset:
Yogyakarta Housing Price
""")

# ======================
# HEADER
# ======================
st.markdown('<p class="big-font">🏠 R-Forest YogyaHouse Dashboard</p>', unsafe_allow_html=True)

st.write("Prediksi harga rumah berbasis Machine Learning di wilayah Yogyakarta")

st.markdown("---")

# ======================
# METRIC CARDS
# ======================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algoritma", "Random Forest")

with col2:
    st.metric("Wilayah", "Yogyakarta")

with col3:
    st.metric("Status", "Active")

st.markdown("---")

# ======================
# MAIN LAYOUT
# ======================
left, right = st.columns([1, 1])

# ======================
# INPUT PANEL
# ======================
with left:
    st.markdown("### Input Data Rumah")

    location = st.selectbox(
        "📍 Lokasi",
        encoder.classes_
    )

    surface_area = st.number_input("Luas Tanah (m²)", 20, 1000, 100)
    building_area = st.number_input("Luas Bangunan (m²)", 20, 1000, 80)
    bed = st.number_input("Kamar Tidur", 1, 20, 3)
    bath = st.number_input("Kamar Mandi", 1, 20, 2)
    carport = st.number_input("Carport", 0, 10, 1)

    predict_button = st.button("🔍 Prediksi Harga")

# ======================
# OUTPUT PANEL
# ======================
with right:
    st.markdown("### Hasil Prediksi")

    if predict_button:

        location_encoded = encoder.transform([location])[0]

        data = np.array([[
            surface_area,
            building_area,
            bed,
            bath,
            carport,
            location_encoded
        ]])

        prediction = model.predict(data)

        st.success(f"Estimasi Harga Rumah")
        st.markdown(f"## Rp {prediction[0]:,.0f}")

        st.markdown("---")

        st.markdown("### Analisis Model")

        features = [
            "Luas Tanah",
            "Luas Bangunan",
            "Kamar Tidur",
            "Kamar Mandi",
            "Carport",
            "Lokasi"
        ]

        importance = model.feature_importances_

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(features, importance)
        plt.xticks(rotation=20)
        ax.set_ylabel("Importance")

        st.pyplot(fig)

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("Developed using Streamlit + Random Forest Regressor")