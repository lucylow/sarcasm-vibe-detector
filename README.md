# Sarcasm Vibe Detector (TinyBERT + Dendrites)

This is the foundation code for a fast, on-device Hinglish sarcasm detector.

## 🚀 Features
- **TinyBERT Base**: Lightweight transformer model for low-latency inference.
- **Dendritic Optimization**: Custom `DendriticLayer` to capture Hinglish nuances without heavy parameter overhead.
- **FastAPI Backend**: High-performance API for real-time predictions.
- **React Frontend**: Modern UI for vibe analysis.

## 🛠️ Setup

### Backend
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   python backend/main.py
   ```

### Frontend
The frontend is a React component. You can integrate it into a Next.js or Create React App project.

## 🧠 Model Architecture
The model uses `huawei-noah/TinyBERT_General_4L_312D` as the backbone, augmented with a custom dendritic layer that adds learned branch-like sub-computations. This is specifically designed to handle the noisy nature of code-mixed Hinglish text.
