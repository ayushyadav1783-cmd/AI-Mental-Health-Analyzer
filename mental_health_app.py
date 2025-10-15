import os
import warnings
warnings.filterwarnings("ignore")
os.environ["STREAMLIT_SUPPRESS_BARE_WARNING"] = "1"

import streamlit as st
from huggingface_hub import InferenceClient

# ==========================
# Page setup
# ==========================
st.set_page_config(
    page_title="AI Mental Health Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================
# Modern Gradient UI (CSS)
# ==========================
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg,#c2e9fb 0%,#a1c4fd 100%);
            font-family:'Poppins',sans-serif;
        }
        .stApp {
            background:linear-gradient(120deg,#a1c4fd,#c2e9fb);
        }
        .title {
            text-align:center;
            font-size:2.3rem;
            color:white;
            text-shadow:2px 2px 12px rgba(0,0,0,0.2);
            margin-bottom:10px;
        }
        .emotion-card {
            background:rgba(255,255,255,0.15);
            border-radius:20px;
            padding:25px;
            margin-top:25px;
            box-shadow:0 10px 25px rgba(0,0,0,0.15);
            backdrop-filter:blur(15px);
            color:#fff;
            text-align:center;
            transition:transform 0.3s ease;
        }
        .emotion-card:hover { transform:scale(1.03); }
        .stButton>button {
            background:linear-gradient(90deg,#36d1dc,#5b86e5);
            color:white;border:none;border-radius:10px;
            padding:0.6em 1.2em;font-size:1rem;font-weight:500;
            transition:all 0.3s ease;
        }
        .stButton>button:hover {
            background:linear-gradient(90deg,#5b86e5,#36d1dc);
            transform:scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Title
# ==========================
st.markdown("<h1 class='title'>🧠 AI Mental Health Emotion Analyzer</h1>", unsafe_allow_html=True)

# ==========================
# Emoji & Explanations
# ==========================
EMOJI_MAP = {
    "sadness":"😔","joy":"😊","love":"❤️","anger":"😡","fear":"😨","surprise":"😲",
    "optimism":"🌟","pessimism":"☁️","trust":"🤝","disgust":"🤢","anticipation":"⏳",
    "admiration":"👏","amusement":"😄","annoyance":"😒","approval":"👍","caring":"🤗",
    "confusion":"😵","curiosity":"🤔","desire":"🔥","disappointment":"😞","disapproval":"👎",
    "embarrassment":"😳","excitement":"🤩","gratitude":"🙏","grief":"🖤","nervousness":"😬",
    "pride":"🏅","realization":"💡","relief":"😮‍💨","remorse":"😣","neutral":"😐"
}

EXPLANATIONS = {
    "joy":"You sound happy and positive!",
    "love":"Your message shows affection and warmth.",
    "gratitude":"You’re expressing thanks and appreciation.",
    "excitement":"You sound energized and thrilled!",
    "optimism":"You’re hopeful and looking forward to good things.",
    "anticipation":"You’re eagerly awaiting something good.",
    "trust":"You express confidence and reliability.",
    "sadness":"You might be feeling down or reflective.",
    "fear":"Your words express worry or concern.",
    "anger":"You seem frustrated or irritated.",
    "disgust":"You’re showing strong disapproval or aversion.",
    "surprise":"You sound astonished or caught off guard.",
    "neutral":"You seem calm and balanced."
}

# ==========================
# Public Hugging Face API (no token)
# ==========================
client = InferenceClient(model="SamLowe/roberta-base-go_emotions")

def analyze_text(text):
    """Use free public inference endpoint (no token needed)."""
    result = client.text_classification(text)
    return [{"label":r["label"],"score":r["score"]} for r in result]

# ==========================
# Input + Analysis
# ==========================
st.markdown("### ✍️ How are you feeling today?")
text = st.text_area("", height=150, placeholder="Example: I feel so happy and proud today!")

if st.button("🔍 Analyze Emotion"):
    if not text.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing emotions with AI... 💫"):
            outputs = analyze_text(text)
            sorted_outputs = sorted(outputs,key=lambda x:x["score"],reverse=True)
            pred_label = sorted_outputs[0]["label"].lower()
            confidence = round(sorted_outputs[0]["score"]*100,2)

        pred_emoji = EMOJI_MAP.get(pred_label,"❓")
        expl = EXPLANATIONS.get(pred_label,"")

        st.markdown(f"""
            <div class='emotion-card'>
                <div class='emotion-label'>Primary Emotion: <b>{pred_label.upper()} {pred_emoji}</b></div>
                <div class='confidence'>Confidence: {confidence}%</div>
                <p style='margin-top:10px;font-size:1rem;color:#f0f0f0;'>{expl}</p>
            </div>
        """,unsafe_allow_html=True)

        st.markdown("#### 🎭 Top Emotions:")
        for item in sorted_outputs[:5]:
            lbl = item["label"].lower()
            conf = round(item["score"]*100,2)
            emoji = EMOJI_MAP.get(lbl,"")
            st.progress(conf/100)
            st.write(f"**{lbl.upper()}** {emoji} — {conf}%")

st.markdown(
    "<hr><p style='text-align:center;font-size:0.9rem;color:#333;'>💡 Powered by Hugging Face Inference API · Model: SamLowe/roberta-base-go_emotions</p>",
    unsafe_allow_html=True,
)

if __name__ == '__main__':
    print('✅ App ready. Run with: streamlit run mental_health_app.py')