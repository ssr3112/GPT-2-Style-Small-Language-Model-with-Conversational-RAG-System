import streamlit as st
import requests
import time
import os
from rag_single import rag_query

API_URL = "http://127.0.0.1:8000/generate"

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="SLM Story Pro",
    layout="wide"
)

# ==========================
# SESSION STATE
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "rag_files" not in st.session_state:
    st.session_state.rag_files = []

# ==========================
# CSS
# ==========================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #E6EDF3;
}
.chat-user {
    background: #2F81F7;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 8px 0;
    max-width: 70%;
    margin-left: auto;
}
.chat-bot {
    background: #21262D;
    color: #E6EDF3;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
    max-width: 70%;
    border: 1px solid #30363D;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# TAB SWITCH (FIX SIDEBAR ISSUE)
# ==========================
active_tab = st.radio(
    "",
    ["🧠 Story Generator", "📚 RAG Chat"],
    horizontal=True
)

# ==========================
# SIDEBAR 

with st.sidebar:

    if active_tab == "🧠 Story Generator":

        st.header("⚙️ Story Settings")

        max_tokens = st.slider("Max Tokens", 20, 100, 70, key="story_max")
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, key="story_temp")
        top_k = st.slider("Top-K", 10, 100, 40, key="story_topk")

        st.divider()

        if st.button("🧹 Clear Story Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    else:
        if st.session_state.rag_messages:
            if st.button("🧹 Clear RAG Chat", use_container_width=True):
                st.session_state.rag_messages = []
                st.rerun()

# ==========================
#  STORY TAB

if active_tab == "🧠 Story Generator":

    st.title("🧠 SLM Story Generator")
    st.markdown("Adjust the parameters in the sidebar to see how it affects the output.")

    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("✍️ Enter your story prompt...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        payload = {
            "prompt": user_input,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k
        }

        with st.spinner("Generating..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            reply = response.json()["generated_text"]
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()
        else:
            st.error("API Error")



# ==========================
# 📚 RAG TAB

elif active_tab == "📚 RAG Chat":

    st.title("📚 RAG Chat")

    # ==========================
    # FILE UPLOAD (MULTI)
    # ==========================
    uploaded_files = st.file_uploader(
        "Upload documents (.txt or .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        os.makedirs("data", exist_ok=True)
        st.session_state.rag_files = []

        for file in uploaded_files:
            path = os.path.join("data", file.name)

            with open(path, "wb") as f:
                f.write(file.getbuffer())

            st.session_state.rag_files.append(path)

        st.success(f"{len(uploaded_files)} documents ready")

    # ==========================
    # CHAT DISPLAY
    # ==========================
    for msg in st.session_state.rag_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

    # ==========================
    # CHAT INPUT
    # ==========================
    if st.session_state.rag_files:

        question = st.chat_input("Ask about your documents...")

        if question:
            st.session_state.rag_messages.append({"role": "user", "content": question})

            with st.spinner("Thinking..."):

                answers = []
                for path in st.session_state.rag_files:
                    ans = rag_query(path, question)
                    answers.append(ans)

                final_answer = "\n\n".join(answers)

            st.session_state.rag_messages.append({"role": "assistant", "content": final_answer})

            st.rerun()

    else:
        st.info("Upload documents to start chatting.")