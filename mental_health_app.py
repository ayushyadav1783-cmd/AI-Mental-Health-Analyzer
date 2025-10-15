import streamlit as st
from transformers import pipeline

@st.cache_resource(show_spinner=False)
def load_model():
    # Use Hugging Face API with authentication token
    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        use_auth_token=st.secrets["HF_TOKEN"]
    )

clf = load_model()

def analyze_text(text):
    """Analyze text using a public model with authentication."""
    try:
        result = clf(text)
        return [{"label": r["label"], "score": r["score"]} for r in result]
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        return []


# ---------- UI ----------
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠", layout="centered")

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

st.markdown("<hr><center><small>Powered by 🤗 Transformers & Streamlit Cloud</small></center>", unsafe_allow_html=True)