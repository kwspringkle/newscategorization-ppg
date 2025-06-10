import streamlit as st
import joblib
from pyvi import ViTokenizer

# Load stopwords
@st.cache_data
def load_stopwords():
    with open("vietnamese_stopwords.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

# Tiền xử lý văn bản
def preprocess(text, stopwords):
    tokens = ViTokenizer.tokenize(text).split()
    filtered = [t for t in tokens if t not in stopwords]
    return ' '.join(filtered)

# Load model, vectorizer, label encoder
@st.cache_resource
def load_components():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_components()
stopwords = load_stopwords()

# Giao diện người dùng
st.title("📄 Vietnamese News Classification")
st.write("Hãy dán nội dung bài báo vào ô bên dưới. Hệ thống sẽ phân loại thể loại của bài viết.")
st.write("© Created by Dinh Ngoc Khanh Huyen, Truong Ngoc Mai, Tran Khanh Quynh")
st.image("images/download.jfif")
# Text input
input_text = st.text_area("📝 Nhập nội dung bài báo tại đây")

# Khi người dùng nhấn nút "Phân loại"
if st.button("Phân loại"):
    if not input_text.strip():
        st.warning("⚠️ Vui lòng nhập nội dung bài báo.")
    else:
        # Tiền xử lý và vector hóa văn bản
        processed_text = preprocess(input_text, stopwords)
        X = vectorizer.transform([processed_text])
        
        # Dự đoán
        predicted_index = model.predict(X)[0]
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        st.success(f"📚 Dự đoán thể loại: **{predicted_label}**")
