# 🧠 GPT-2 Style Small Language Model + RAG Chat System

An end-to-end AI project that combines a **GPT-2 style transformer-based Small Language Model (SLM)** with a **Retrieval-Augmented Generation (RAG) system** to generate stories and answer questions from documents.

---

##  Project Overview

This project demonstrates how modern AI systems work by building:

- A **transformer-based language model from scratch**
- A **multi-document RAG pipeline**
- A **FastAPI backend**
- A **Streamlit interactive UI**


##  Features

### 🧠 Story Generator (SLM)
- GPT-2 style transformer model (implemented from scratch)
- Autoregressive text generation
- Supports temperature and top-k sampling
- Trained on TinyStories dataset (~4.4M tokens)

### 📚 RAG Chat System
- Upload multiple documents (.txt / .pdf)
- Ask questions from documents
- Uses embeddings + FAISS for retrieval
- Gemini API for accurate answer generation
- Chat-style interface with memory

### 💻 UI (Streamlit)
- Dual dashboard:
  - Story Generator
  - RAG Chat
- Clean chat interface (like ChatGPT)
- Real-time streaming responses

---

# 🧠 GPT-2 Style Small Language Model + RAG Chat System

An end-to-end AI project that combines a **GPT-2 style transformer-based Small Language Model (SLM)** with a **Retrieval-Augmented Generation (RAG) system** to generate stories and answer questions from documents.

---

## 🎯 Project Overview

This project demonstrates how modern AI systems work by building:

- A **transformer-based language model from scratch**
- A **multi-document RAG pipeline**
- A **FastAPI backend**
- A **Streamlit interactive UI**

👉 It is a complete ML system:  
**Model → API → UI → User Interaction**

---

## ✨ Features

### 🧠 Story Generator (SLM)
- GPT-2 style transformer model (implemented from scratch)
- Autoregressive text generation
- Supports temperature and top-k sampling
- Trained on TinyStories dataset (~4.4M tokens)

### 📚 RAG Chat System
- Upload multiple documents (.txt / .pdf)
- Ask questions from documents
- Uses embeddings + FAISS for retrieval
- Gemini API for accurate answer generation
- Chat-style interface with memory

### 💻 UI (Streamlit)
- Dual dashboard:
  - Story Generator
  - RAG Chat
- Clean chat interface (like ChatGPT)
- Real-time streaming responses

---

##  Tech Stack

- **Language:** Python  
- **Deep Learning:** PyTorch  
- **NLP:** Transformers, Tokenization (tiktoken)  
- **RAG:** FAISS, SentenceTransformers  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **API:** Gemini API  

---

---

## ⚙️ Installation

### 1️⃣ Clone Repository


git clone https://github.com/your-username/slm-transformers.git
cd slm-transformers
---

## Creating Virtual Environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

## Create Virtual file (.env) name 
GEMINI_API_KEY= paste your api

## Backend
uvicorn main:app --reload

## Frotend
streamlit run streamlit_ui.py


