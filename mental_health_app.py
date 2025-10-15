import streamlit as st
from huggingface_hub import InferenceClient

# Load model using Hugging Face Inference API
@st.cache_resource(show_spinner=False)
def load_client():
    return InferenceClient(
        model="SamLowe/roberta-base-go_emotions",
        token=st.secrets["HF_TOKEN"]
    )

client = load_client()

def analyze_text(text):
    """Analyze text using the Hugging Face Inference API."""
    try:
        response = client.text_classification(text)
        return [{"label": r["label"], "score": r["score"]} for r in response]
    except Exception as e:
        st.error(f"⚠️ API error: {e}")
        return []


# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Mental Health Analyzer", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f7ff 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    div[data-testid="stMarkdownContainer"] h1 {
        color: #1f77b4;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Mental Health / Sentiment Analyzer")
st.markdown("Type how you’re feeling below, and let AI detect your mood instantly!")

text = st.text_area("💬 How are you feeling today?", height=150, placeholder="e.g. I feel calm and hopeful today...")

if st.button("🔍 Analyze"):
    if not text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            results = analyze_text(text)
            if results:
                label = results[0]["label"]
                confidence = round(results[0]["score"] * 100, 2)
                emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😔"}
                st.success(f"**Emotion:** {label} {emoji_map.get(label, '🤖')}  \n**Confidence:** {confidence}%")

st.markdown("<hr><center><small>Powered by 🤗 Hugging Face Inference API & Streamlit Cloud</small></center>", unsafe_allow_html=True)