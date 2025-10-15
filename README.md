# 🧠 AI Mental Health Text Analyzer  

A **Streamlit web app** that uses **Hugging Face Transformers** to detect emotions and mental health tones from user text input in real-time.  

✨ **Live Demo:** [AI Mental Health Analyzer on Streamlit](https://ai-mental-health-analyzer-gpd4fnlecxpcfanl7hcpig.streamlit.app/)  

---

## 🚀 Features

- 🎯 Emotion detection (Joy, Sadness, Fear, Anger, Surprise, etc.)  
- 🧩 Powered by Transformer-based NLP models from Hugging Face  
- 🎨 Modern UI with smooth animations and emoji-based feedback  
- 🧱 Streamlit-powered — runs on web and local easily  
- ☁️ Hugging Face API integration for secure inference  

---

## 🧰 Tech Stack

| Technology | Description |
|-------------|-------------|
| **Python 3.10+** | Programming language |
| **Streamlit** | Web app framework |
| **Transformers** | NLP models |
| **Hugging Face Inference API** | Emotion model backend |
| **Torch (PyTorch)** | Model runtime |
| **HTML + CSS (Custom)** | For UI Styling |

---

## ⚙️ Setup Instructions (Local)

```bash
# 1️⃣ Clone the repository
git clone https://github.com/ayushyadav1783-cmd/AI-Mental-Health-Analyzer.git
cd AI-Mental-Health-Analyzer

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Add your Hugging Face token in secrets.toml
mkdir -p ~/.streamlit
echo '[general]\nemail="your_email@domain.com"\nHF_TOKEN="your_huggingface_token"' > ~/.streamlit/secrets.toml

# 4️⃣ Run the app
streamlit run mental_health_app.py