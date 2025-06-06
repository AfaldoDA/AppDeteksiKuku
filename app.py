import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
#css
st.markdown("""
<style>
    .title-text {
        font-size: 'Arial', sans-serif;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f4f8;
    }
    .stProgress .st-bo {
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model.keras: {e}")
        return None

model = load_model()
if model is None:
    st.stop()


img_size = (224, 224)

# Label 
label_map = {
    "Healthy": "Kuku Anda sehat, menandakan tidak ada tanda penyakit tidak menular. Tetap jaga kebersihan kuku!",
    "Koilonychia": "Kuku Anda terdeteksi sebagai **koilonychia**! Ini bisa jadi tanda anemia. Segera konsultasi dengan dokter untuk pemeriksaan lebih lanjut.",
    "Onychomycosis": "Kuku Anda terdeteksi sebagai **onychomycosis**! Ini mungkin terkait diabetes atau infeksi jamur. Segera hubungi dokter untuk penanganan."
}
index_to_label = {0: "Healthy", 1: "Koilonychia", 2: "Onychomycosis"}

# preprocessing
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(img_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

# Sidebar
st.sidebar.title("Tentang Aplikasi")
st.sidebar.markdown("""
Aplikasi ini menggunakan model **VGG16** untuk mendeteksi kondisi kuku berdasarkan citra. Dibuat untuk membantu skrining awal penyakit tidak menular seperti anemia (koilonychia) dan diabetes (onychomycosis).  
**Akurasi Model**: 90.56%  
**Dibuat oleh**: Alfiana Hidayati.  
""")
st.sidebar.image("image.jpg", caption="Contoh Upload Gambar ", use_container_width=True)

# Main UI
st.markdown('<h1 class="title-text">Deteksi Penyakit Kuku Berbasis AI</h1>', unsafe_allow_html=True)
st.markdown("Unggah gambar kuku untuk mengetahui apakah kuku Anda sehat, terkena koilonychia, atau onychomycosis. Dapatkan hasil cepat dengan akurasi tinggi!")

# File uploader
uploaded_file = st.file_uploader("Pilih Gambar Kuku (JPG/PNG)", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded_file is not None:
    # Preview gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Kuku yang Diunggah", width=300)

    # Tombol prediksi
    if st.button("Prediksi Sekarang"):
        with st.spinner("Memproses gambar..."):
            # Preprocess dan prediksi
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100

            label = index_to_label[predicted_class]
            description = label_map[label]

            # Tampilkan hasil di box
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### Hasil Prediksi: **{label}**")
            st.progress(int(confidence))
            st.markdown(f"**Tingkat Keyakinan**: {confidence:.2f}%")
            st.info(description)
            st.markdown('</div>', unsafe_allow_html=True)


# Info tambahan
with st.expander("Informasi Penting"):
    st.markdown("""
    - **Kuku Sehat**: Kuku tampak normal, tanpa perubahan bentuk atau warna.
    - **Koilonychia**: Kuku cekung seperti sendok, sering terkait anemia.
    - **Onychomycosis**: Kuku menebal, berubah warna, atau rapuh, bisa tanda diabetes atau infeksi jamur.
    - **Catatan**: Hasil prediksi bukan diagnosis medis. Konsultasikan dengan dokter untuk kepastian.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Alfiana Hidayati.")
