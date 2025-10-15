#!/bin/bash
echo "=== Setting up a clean Streamlit environment ==="

# Step 1: Create new venv (in your home folder)
python3 -m venv ~/streamlit_env

# Step 2: Activate venv
source ~/streamlit_env/bin/activate

# Step 3: Upgrade pip and install Streamlit (pinned stable)
pip install --upgrade pip
pip install streamlit==1.38.0

# Step 4: Run the app
echo "=== Launching Streamlit app ==="
streamlit run /Users/ayush_home/Downloads/PYTHON/mental_health_app.py
