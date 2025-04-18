# 🩺 AI Doctor Chatbot — Multimodal Diagnosis App

This project is a **Gradio-based AI doctor chatbot** capable of analyzing both **voice inputs** and **facial images** to assist in diagnosing common visible diseases and conditions. It uses **state-of-the-art language models, vision models, speech-to-text, and text-to-speech tools**.

---

## 📌 Features

- 🔍 **Facial Image Analysis** — Detect visible symptoms like acne, jaundice, conjunctivitis, and more using multimodal AI.
- 🗣️ **Voice Transcription** — Convert voice input into text using **Whisper**.
- 🖥️ **Multimodal LLM (LLaMA 3 Vision)** — Combine image and text inputs for comprehensive health assessment.
- 🗣️ **ElevenLabs Text-to-Speech** — Converts AI responses into realistic, human-like voice.
- 📖 **Medical Document Q&A** — Upload PDFs and ask medical questions based on embedded documents.
- 🧠 **FAISS-powered Vector Search** — For fast, intelligent document retrieval.
- 💾 **Customizable Interface with Gradio** — Clean and responsive user interface.

---

## 🛠️ Tech Stack

- **Python**
- **Gradio**
- **LangChain**
- **FAISS**
- **HuggingFace Transformers**
- **Whisper (Groq API)**
- **LLaMA 3 Vision**
- **ElevenLabs**
- **Sentence Transformers (MiniLM-L6-v2)**

---

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-doctor-chatbot.git
   cd ai-doctor-chatbot
