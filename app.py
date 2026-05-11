import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
import folium
from streamlit_folium import st_folium

# ======================
# LOAD MODEL & ENCODER
# ======================
@st.cache_resource
def load_model():
    return joblib.load("model/random_forest.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("model/location_encoder.pkl")

model = load_model()
encoder = load_encoder()

# ======================
# KOORDINAT LOKASI (Yogyakarta & sekitarnya)
# ======================
location_coords = {
    "Kotagede": [-7.8266, 110.3975],
    "Gondokusuman": [-7.7825, 110.3686],
    "Danurejan": [-7.7905, 110.3701],
    "Gondomanan": [-7.8005, 110.3688],
    "Pakualaman": [-7.7990, 110.3710],
    "Mantrijeron": [-7.8183, 110.3700],
    "Wirobrajan": [-7.7965, 110.3625],
    "Mergangsan": [-7.8126, 110.3678],
    "Umbulharjo": [-7.7945, 110.3927],
    "Tegalrejo": [-7.7725, 110.3605],
    "Jetis": [-7.7805, 110.3638],
    "Ngampilan": [-7.7963, 110.3605],
    "Kraton": [-7.8080, 110.3690],
    "Gedongtengen": [-7.7912, 110.3630],
    "Sleman": [-7.6855, 110.3595],
    "Depok": [-7.7650, 110.4110],
    "Banguntapan": [-7.8100, 110.3910],
    "Kasihan": [-7.8230, 110.3410],
    "Sewon": [-7.8760, 110.3330],
    "Bantul": [-7.8860, 110.3310],
    "Godean": [-7.7680, 110.2930],
    "Mlati": [-7.7380, 110.3570],
    "Ngaglik": [-7.7080, 110.3880],
    "Ngemplak": [-7.6890, 110.4040],
    "Kalasan": [-7.7280, 110.4720],
}
default_coord = [-7.7956, 110.3695]  # Pusat Yogyakarta

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="R-Forest YogyaHouse | AI Property Predictor",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM CSS (modern glassmorphism)
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;400;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

html, body, .stApp {
    background: radial-gradient(circle at 10% 20%, rgba(240,248,255,0.95) 0%, rgba(230,242,255,0.98) 90%);
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: radial-gradient(#3b82f6 0.8px, transparent 0.8px);
    background-size: 28px 28px;
    opacity: 0.08;
    pointer-events: none;
    z-index: 0;
}

.glass-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    border-radius: 32px;
    padding: 1.8rem;
    box-shadow: 0 20px 35px -12px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.5);
    transition: all 0.3s cubic-bezier(0.2,0.9,0.4,1.1);
    animation: fadeInUp 0.6s ease-out;
    margin-bottom: 1.5rem;
}
.glass-card:hover {
    transform: translateY(-6px);
    background: rgba(255,255,255,0.85);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.metric-super {
    background: rgba(255,255,240,0.6);
    backdrop-filter: blur(8px);
    border-radius: 28px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.7);
    transition: transform 0.2s;
}
.metric-super:hover { transform: scale(1.02); }

.price-big {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
    letter-spacing: -0.02em;
    animation: pulse 1.2s ease-out;
}
@keyframes pulse {
    0% { opacity: 0; transform: scale(0.95);}
    100% { opacity: 1; transform: scale(1);}
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.6);
}

.glass-footer {
    background: rgba(0,0,0,0.2);
    backdrop-filter: blur(5px);
    border-radius: 40px;
    padding: 0.8rem;
    text-align: center;
    color: #1e293b;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ======================
# SESSION STATE (untuk navigasi & prediksi)
# ======================
if "show_main" not in st.session_state:
    st.session_state.show_main = False
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
    st.session_state.last_prediction = None
    st.session_state.last_location = None

# ======================
# LANDING PAGE
# ======================
def landing_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=130)
        st.markdown('<h1 style="font-size:3.5rem; font-weight:800; background:linear-gradient(120deg,#0f172a,#2563eb); -webkit-background-clip:text; color:transparent;">R-Forest Yogya</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:1.2rem; color:#334155;">Prediksi harga rumah berbasis AI untuk Yogyakarta<br>Akurasi tinggi & peta interaktif</p>', unsafe_allow_html=True)
        if st.button("✨ MASUK KE DASHBOARD ✨", use_container_width=True):
            st.session_state.show_main = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown('<div class="metric-super"><span style="font-size:2rem;">🌲</span><h4>Random Forest</h4><p>Model ensemble stabil & akurat</p></div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="metric-super"><span style="font-size:2rem;">📈</span><h4>Feature Importance</h4><p>Ketahui faktor paling berpengaruh</p></div>', unsafe_allow_html=True)
    with colC:
        st.markdown('<div class="metric-super"><span style="font-size:2rem;">📍</span><h4>Lokasi Yogyakarta</h4><p>Data real properti di berbagai kecamatan</p></div>', unsafe_allow_html=True)

    with st.expander("📘 Pelajari Lebih Lanjut", expanded=False):
        st.markdown("""
        **R-Forest Yogya** menggunakan algoritma **Random Forest Regressor** yang dilatih dengan data properti di Daerah Istimewa Yogyakarta.
        - Input luas tanah, bangunan, jumlah kamar tidur, kamar mandi, carport, dan pilih lokasi.
        - Dapatkan estimasi harga + peta lokasi + analisis faktor penting.
        """)
    st.markdown("---")
    st.markdown('<div class="glass-footer">🏯 R-Forest Yogya — Prediksi properti cerdas</div>', unsafe_allow_html=True)

# ======================
# MAIN DASHBOARD (dengan map & session state)
# ======================
def main_dashboard():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=90)
        st.markdown("### 🏮 R-Forest Yogya")
        st.markdown("---")
        if st.button("🏠 Kembali ke Beranda", use_container_width=True):
            st.session_state.show_main = False
            st.session_state.prediction_done = False  # reset prediksi
            st.rerun()
        st.markdown("---")
        st.markdown("#### 🧠 Informasi Model")
        st.info("Algoritma: Random Forest\nEstimator: 100 trees\nFitur: 6 dimensi")
        st.markdown("#### 📌 Tips")
        st.success("Semakin luas tanah & bangunan, semakin tinggi estimasi harga.")
        st.markdown("---")
        st.caption("v3.0 — UI/UX + Peta Interaktif")

    st.markdown('<h1 style="font-size:2.5rem; font-weight:700;">🏡 Dashboard Prediksi Harga</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569; margin-bottom:2rem;">Masukkan detail properti di bawah, sistem akan memprediksi harga wajar di Yogyakarta.</p>', unsafe_allow_html=True)

    # Metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-super"><span>🎯</span><h4>Random Forest</h4></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-super"><span>📍</span><h4>Yogyakarta</h4></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-super"><span>⚡</span><h4>Real-time</h4></div>', unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1, 1], gap="large")

    # Kolom input
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Spesifikasi Properti")

        location = st.selectbox("📍 Lokasi / Kecamatan", encoder.classes_)
        surface_area = st.number_input("📐 Luas Tanah (m²)", 20, 1000, 100, step=10)
        building_area = st.number_input("🏠 Luas Bangunan (m²)", 20, 1000, 80, step=10)
        bed = st.number_input("🛏️ Jumlah Kamar Tidur", 1, 20, 3)
        bath = st.number_input("🚿 Jumlah Kamar Mandi", 1, 20, 2)
        carport = st.number_input("🚗 Kapasitas Carport", 0, 10, 1)

        predict_btn = st.button("🔮 PREDIKSI HARGA SEKARANG", use_container_width=True)

        if predict_btn:
            with st.spinner("🧠 AI sedang menganalisis..."):
                time.sleep(0.8)
                location_encoded = encoder.transform([location])[0]
                data = np.array([[
                    surface_area,
                    building_area,
                    bed,
                    bath,
                    carport,
                    location_encoded
                ]])
                prediction = model.predict(data)[0]

            # Simpan ke session state
            st.session_state.prediction_done = True
            st.session_state.last_prediction = prediction
            st.session_state.last_location = location

        st.markdown('</div>', unsafe_allow_html=True)

    # Kolom output (hasil + peta)
    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💰 Estimasi Harga Properti")

        if st.session_state.prediction_done:
            pred = st.session_state.last_prediction
            loc = st.session_state.last_location

            st.markdown(f'<p class="price-big">Rp {pred:,.0f}</p>', unsafe_allow_html=True)
            st.success("✅ Prediksi berhasil! Harga di atas adalah estimasi berdasarkan model AI.")
            st.markdown("---")

            # TAMPILKAN PETA
            st.markdown("### 🗺️ Peta Lokasi Properti")
            coords = location_coords.get(loc, default_coord)
            m = folium.Map(location=coords, zoom_start=14, control_scale=True)
            folium.Marker(
                location=coords,
                popup=f"<b>{loc}</b><br>Estimasi harga: Rp {pred:,.0f}",
                icon=folium.Icon(color="red", icon="home", prefix="fa"),
                tooltip=loc
            ).add_to(m)
            folium.Circle(
                radius=300,
                location=coords,
                color="crimson",
                fill=True,
                fill_opacity=0.1
            ).add_to(m)

            st_folium(m, width=550, height=400, key="prediction_map")
            st.caption(f"📍 Lokasi yang dipilih: {loc} (perkiraan pusat kecamatan)")
            st.markdown("---")

            # FEATURE IMPORTANCE
            st.markdown("### 📊 Faktor Pengaruh Terbesar")
            features = ["Luas Tanah", "Luas Bangunan", "Kamar Tidur", "Kamar Mandi", "Carport", "Lokasi"]
            importance = model.feature_importances_

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#3b82f6' if i < 2 else '#94a3b8' for i in range(len(importance))]
            bars = ax.barh(features, importance, color=colors, edgecolor='white', linewidth=1.5)
            ax.set_xlabel("Tingkat Kepentingan", fontsize=11)
            ax.set_title("Feature Importance - Random Forest", fontsize=13, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("💡 **Insight:** Luas tanah dan bangunan mendominasi prediksi. Lokasi juga cukup berpengaruh.")

            # Tombol reset prediksi
            if st.button("🔄 Prediksi Baru", key="reset_btn"):
                st.session_state.prediction_done = False
                st.rerun()
        else:
            st.markdown('<div style="text-align:center; padding:2rem 0;"><span style="font-size:2rem;">👈</span><br>Silakan masukkan data properti di samping kiri, lalu tekan tombol prediksi.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="glass-footer">🚀 R-Forest Yogya | Prediksi harga bersifat indikatif • Peta menunjukkan perkiraan lokasi berdasarkan kecamatan</div>', unsafe_allow_html=True)

# ======================
# RENDER PAGE
# ======================
if st.session_state.show_main:
    main_dashboard()
else:
    landing_page()