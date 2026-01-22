import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Sampah Plastik AI",
    page_icon="♻️",
    layout="centered"
)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file '{model_path}' ada! Error: {e}")
        return None

model = load_model('best.pt')

# --- TAMPILAN UTAMA (UI) ---
st.title("♻️ Deteksi Sampah Plastik Cerdas")
st.caption("By: Rendy Elang Lesmana")
st.write("Silakan upload gambar atau ambil foto langsung untuk mendeteksi sampah plastik.")
st.markdown("---")

# --- BAGIAN INPUT (DISATUKAN) ---
st.subheader("Pilih Input Gambar")

# Opsi 1: Upload File
uploaded_file = st.file_uploader("📤 Upload foto sampah (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

# Spasi sedikit agar tidak terlalu rapat
st.write("--- atau ---")

# Opsi 2: Kamera Langsung (Berada di bawah upload)
camera_file = st.camera_input("📸 Ambil foto menggunakan kamera")

# Menentukan sumber gambar mana yang akan dipakai
# Prioritas: Jika ada upload file, pakai itu. Jika tidak, pakai kamera.
image_source = None
if uploaded_file is not None:
    image_source = uploaded_file
elif camera_file is not None:
    image_source = camera_file

# --- LOGIKA UTAMA ---
if image_source is not None and model is not None:
    # Membaca gambar
    image_pil = Image.open(image_source)
    
    # Menyiapkan kolom hasil
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image_pil, use_container_width=True)
        
    # Tombol Deteksi
    if st.button("🔍 Mulai Deteksi Plastik", type="primary"):
        with st.spinner('Sedang menganalisis... 🤖'):
            # Prediksi (conf=0.25)
            results = model.predict(image_pil, conf=0.25)
            
            # Plot hasil
            result_array_bgr = results[0].plot()
            result_array_rgb = result_array_bgr[..., ::-1]
            jumlah_plastik = len(results[0].boxes)

        with col2:
            st.subheader("Hasil Deteksi")
            st.image(result_array_rgb, use_container_width=True)
            
        if jumlah_plastik > 0:
            st.success(f"Ditemukan **{jumlah_plastik}** objek plastik.")
        else:
            st.warning("Tidak ditemukan objek plastik yang terdeteksi.")

else:
    st.info("💡 Menunggu input gambar... (Upload file atau gunakan kamera di atas)")

# Footer
st.markdown("---")
st.caption("Dibuat untuk Project Mata Kuliah Digital Image Processing")