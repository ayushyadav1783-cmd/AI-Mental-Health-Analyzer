## 🧠 AI Mental Health Emotion Analyzer

A Streamlit-powered web app that analyzes the **emotional tone of text** using a fine-tuned **RoBERTa emotion classification model** from Hugging Face.  
Simply type how you’re feeling — the app identifies emotions like *joy, sadness, fear, love,* and more, with confidence scores and emojis for an intuitive user experience.  

---

### 🚀 Live Demo  
🔗 **[Open the App on Streamlit Cloud](https://ai-mental-health-analyzer.streamlit.app)** *(replace with your actual app link once it’s deployed)*  

---

### 🧩 Features
✅ Real-time emotion prediction  
✅ Friendly UI with emojis & confidence levels  
✅ Powered by Hugging Face Transformers  
✅ Lightweight & fast — runs entirely in the cloud  
✅ Secure Hugging Face token management with `st.secrets`  

---

### 🖥️ Tech Stack
- **Frontend/UI:** Streamlit  
- **Model:** `SamLowe/roberta-base-go_emotions` (Hugging Face)  
- **Backend:** Python  
- **Libraries:**  
  - `transformers`  
  - `torch` *(optional / CPU mode supported)*  
  - `streamlit`  
  - `protobuf`, `accelerate`, `sentencepiece`  

---

### ⚙️ Local Setup
You can run this app locally in 3 simple steps:

```bash
# 1️⃣ Clone the repo
git clone https://github.com/ayushyadav1783-cmd/AI-Mental-Health-Analyzer.git
cd AI-Mental-Health-Analyzer

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the app
streamlit run mental_health_app.py
```

---

### 🔐 Hugging Face Token Setup
Create a `.streamlit/secrets.toml` file in your project directory:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

You can generate your token from [Hugging Face settings → Access Tokens](https://huggingface.co/settings/tokens).  

---

### 📸 Preview
*(Add a screenshot of your running app here once deployed)*  

---

### 🧑‍💻 Author
**Ayush Yadav**  
📍 Machine Learning Engineer | Data Science Enthusiast  
🔗 [GitHub](https://github.com/ayushyadav1783-cmd)
