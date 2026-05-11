import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import time

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
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="R-Forest YogyaHouse | AI Property Predictor",
    page_icon="🏯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM CSS (ULTRA MODERN)
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,600;14..32,700;14..32,800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

html, body, .stApp {
    background: radial-gradient(circle at 10% 20%, rgba(240, 248, 255, 0.95) 0%, rgba(230, 242, 255, 0.98) 90%);
}

/* Animated subtle pattern */
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

/* Card glassmorphism + animation */
.glass-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(12px);
    border-radius: 32px;
    padding: 1.8rem;
    box-shadow: 0 20px 35px -12px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.5);
    transition: all 0.3s cubic-bezier(0.2, 0.9, 0.4, 1.1);
    animation: fadeInUp 0.6s ease-out;
    margin-bottom: 1.5rem;
}
.glass-card:hover {
    transform: translateY(-6px);
    background: rgba(255, 255, 255, 0.85);
    box-shadow: 0 28px 40px -16px rgba(0,0,0,0.2);
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Gradient button */
.gradient-btn {
    background: linear-gradient(105deg, #1e3a8a 0%, #3b82f6 100%);
    border: none;
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.8rem;
    border-radius: 60px;
    transition: all 0.25s;
    box-shadow: 0 6px 14px rgba(59,130,246,0.3);
    width: 100%;
}
.gradient-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 12px 22px rgba(59,130,246,0.4);
    background: linear-gradient(105deg, #2563eb, #1d4ed8);
}

/* Metric supercard */
.metric-super {
    background: rgba(255,255,240,0.6);
    backdrop-filter: blur(8px);
    border-radius: 28px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.7);
    transition: transform 0.2s;
}
.metric-super:hover {
    transform: scale(1.02);
}

/* Big price display */
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

/* Sidebar style */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.6);
}
section[data-testid="stSidebar"] .css-1d391kg {
    background: transparent;
}

/* Custom select, number input */
.stSelectbox, .stNumberInput {
    background: rgba(255,255,255,0.5);
    border-radius: 20px;
}

/* Loading spinner custom */
.loader {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3b82f6;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Footer glass */
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
# SESSION STATE
# ======================
if "show_main" not in st.session_state:
    st.session_state.show_main = False
if "loading" not in st.session_state:
    st.session_state.loading = False

# ======================
# LANDING PAGE (enhanced)
# ======================
def landing_page():
    # Hero section dengan efek glass
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=130)
        st.markdown('<h1 style="font-size:3.5rem; font-weight:800; background:linear-gradient(120deg,#0f172a,#2563eb); -webkit-background-clip:text; color:transparent;">R-Forest Yogya</h1>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:1.2rem; color:#334155;">Prediksi harga rumah berbasis AI untuk Yogyakarta<br>Keakuratan tinggi & visualisasi real-time</p>', unsafe_allow_html=True)
        
        if st.button("✨ MASUK KE DASHBOARD ✨", key="enter_main", use_container_width=True):
            st.session_state.show_main = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 3 fitur unggulan
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown('<div class="metric-super"><span style="font-size:2rem;">🌲</span><h4>Random Forest</h4><p>Model ensemble yang stabil & akurat</p></div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="metric-super"><span style="font-size:2rem;">📈</span><h4>Feature Importance</h4><p>Ketahui faktor paling berpengaruh</p></div>', unsafe_allow_html=True)
    with colC:
        st.markdown('<div class="metric-super"><span style="font-size:2rem;">📍</span><h4>Lokasi Yogyakarta</h4><p>Data real properti di berbagai kecamatan</p></div>', unsafe_allow_html=True)
    
    # Penjelasan dengan expander
    with st.expander("📘 Pelajari Lebih Lanjut tentang R-Forest Yogya", expanded=False):
        st.markdown("""
        **R-Forest Yogya** menggunakan algoritma **Random Forest Regressor** yang dilatih dengan ratusan data properti di Daerah Istimewa Yogyakarta.  
        Model ini mempertimbangkan:
        - Luas tanah & bangunan
        - Jumlah kamar tidur, mandi, dan carport
        - Kode lokasi (6+ kecamatan)
        
        Keunggulan: tahan terhadap overfitting, mampu menangkap hubungan non-linear, dan memberikan interpretasi feature importance.
        
        **Cara pakai:**  
        1. Klik tombol "Masuk ke Dashboard"  
        2. Isi spesifikasi properti Anda  
        3. Klik prediksi dan dapatkan estimasi harga + analisis
        """)
    
    st.markdown("---")
    st.markdown('<div class="glass-footer">🏯 R-Forest Yogya — Memberdayakan keputusan properti Anda dengan Machine Learning</div>', unsafe_allow_html=True)

# ======================
# MAIN DASHBOARD (enhanced)
# ======================
def main_dashboard():
    # Sidebar super stylish
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=90)
        st.markdown("### 🏮 R-Forest Yogya")
        st.markdown("---")
        if st.button("🏠 Kembali ke Beranda", use_container_width=True):
            st.session_state.show_main = False
            st.rerun()
        st.markdown("---")
        st.markdown("#### 🧠 Informasi Model")
        st.info("""
        **Algoritma** : Random Forest  
        **Estimator** : 100 trees  
        **Fitur** : 6 dimensi  
        **Akurasi** : R² ~0.85
        """)
        st.markdown("#### 📌 Tips")
        st.success("Semakin luas tanah & bangunan, semakin tinggi estimasi harga. Lokasi premium juga menambah nilai.")
        st.markdown("---")
        st.caption("v3.0 — UI/UX Premium")
    
    # Header dashboard
    st.markdown('<h1 style="font-size:2.5rem; font-weight:700;">🏡 Dashboard Prediksi Harga</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569; margin-bottom:2rem;">Masukkan detail properti di bawah, sistem akan memprediksi harga wajar di Yogyakarta.</p>', unsafe_allow_html=True)
    
    # Metric row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-super"><span>🎯</span><h4>Random Forest</h4><p>Regressor</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-super"><span>📍</span><h4>Yogyakarta</h4><p>+ Sleman, Bantul</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-super"><span>⚡</span><h4>Real-time</h4><p>Prediksi instan</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Kolom input & output dengan glass card
    left, right = st.columns([1, 1], gap="large")
    
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💰 Estimasi Harga Properti")
        
        if predict_btn:
            # Simulasi loading
            with st.spinner("🧠 AI sedang menganalisis..."):
                time.sleep(0.8)  # efek loading halus
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
            
            # Animasi angka besar
            st.markdown(f'<p class="price-big">Rp {prediction:,.0f}</p>', unsafe_allow_html=True)
            st.success("✅ Prediksi berhasil! Harga di atas adalah estimasi berdasarkan model AI.")
            st.markdown("---")
            st.markdown("### 📊 Faktor Pengaruh Terbesar")
            
            # Feature importance plot dengan styling lebih baik
            features = ["Luas Tanah", "Luas Bangunan", "Kamar Tidur", "Kamar Mandi", "Carport", "Lokasi"]
            importance = model.feature_importances_
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#3b82f6' if i<2 else '#94a3b8' for i in range(len(importance))]
            bars = ax.barh(features, importance, color=colors, edgecolor='white', linewidth=1.5)
            ax.set_xlabel("Tingkat Kepentingan", fontsize=11)
            ax.set_title("Feature Importance - Random Forest", fontsize=13, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Insight tambahan
            st.markdown("💡 **Insight:** Luas tanah dan bangunan mendominasi prediksi. Lokasi juga cukup berpengaruh.")
        else:
            st.markdown('<div style="text-align:center; padding:2rem 0;"><span style="font-size:2rem;">👈</span><br>Silakan masukkan data properti di samping kiri, lalu tekan tombol prediksi.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer premium
    st.markdown("---")
    st.markdown('<div class="glass-footer">🚀 R-Forest Yogya | Prediksi harga bersifat indikatif • Selalu lakukan verifikasi pasar secara mandiri</div>', unsafe_allow_html=True)

# ======================
# RENDER PAGE
# ======================
if st.session_state.show_main:
    main_dashboard()
else:
    landing_page()