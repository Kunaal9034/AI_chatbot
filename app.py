import os
import faiss
import pandas as pd
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# =========================
# 0. BASIC CONFIG + STYLING
# =========================

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Hide Streamlit default chrome (menu, footer, header)
HIDE_ST_STYLE = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

body {
    background-color: #050816;
}
</style>
"""
st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

# Extra CSS for chat + model bar
CUSTOM_CSS = """
<style>
.chat-container {
    max-width: 900px;
    margin: 0 auto;
}

.model-bar {
    background: #090f1f;
    border-radius: 999px;
    padding: 6px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    border: 1px solid #222b3d;
}

.model-label {
    font-size: 13px;
    color: #9fa6b2;
}

.model-selectbox > div[data-baseweb="select"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.model-selectbox span {
    font-size: 13px !important;
}

.chat-bubble-user {
    background: #1f2933;
    color: #ffffff;
    padding: 10px 14px;
    border-radius: 18px;
    border-bottom-right-radius: 4px;
    max-width: 80%;
    margin-left: auto;
    margin-bottom: 6px;
    font-size: 14px;
}

.chat-bubble-bot {
    background: #0f172a;
    color: #e5e7eb;
    padding: 10px 14px;
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    max-width: 80%;
    margin-right: auto;
    margin-bottom: 6px;
    font-size: 14px;
}

.chat-role {
    font-size: 11px;
    color: #9ca3af;
    margin-bottom: 2px;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========
# 1. GROQ KEY
# ==========

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
if not GROQ_API_KEY and "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Set it in .env (local) or Streamlit secrets (cloud).")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ==========================
# 2. LOAD DATA + BUILD INDEX
# ==========================

DATA_PATH = "Data/galgotias_faq_3000.csv"

@st.cache_data
def load_faq_data(path: str):
    df = pd.read_csv(path)
    return df

faq_df = load_faq_data(DATA_PATH)

@st.cache_resource
def build_rag_index(df: pd.DataFrame):
    docs = []
    meta = []

    for _, row in df.iterrows():
        q = str(row["question"])
        a = str(row["answer"])
        text = f"Q: {q}\nA: {a}"
        docs.append(text)
        meta.append({"id": int(row["id"]), "category": row["category"]})

    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = embed_model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return docs, meta, embed_model, index

docs, metadata, embed_model, index = build_rag_index(faq_df)

# ================
# 3. RAG UTILITIES
# ================

def retrieve_context(query: str, k: int = 5):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, k)
    return [docs[i] for i in indices[0]]

def generate_rag_answer(query: str, profile: str, k: int = 5) -> str:
    context_chunks = retrieve_context(query, k=k)
    context = "\n\n".join(context_chunks)

    # Different behaviour based on profile
    if profile == "Smart (RAG)":
        style = "Be concise but clear."
    elif profile == "Search (strict RAG)":
        style = ("Use ONLY the context. "
                 "If answer is not present, clearly say you don't know.")
    else:
        style = "Be helpful and slightly detailed."

    prompt = f"""
You are an enquiry bot for Galgotias University.

Conversation style: {style}

CONTEXT:
{context}

QUESTION:
{query}

If the answer is not clearly in the context, say you don't know
and suggest visiting the official Galgotias University website.

Answer:
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content

def generate_general_answer(history, profile: str) -> str:
    # Build system prompt based on profile
    if profile == "Quick response":
        style = "Be very brief and to the point."
        temp = 0.3
    elif profile == "Think deeper":
        style = "Give deeper reasoning and more detailed explanations."
        temp = 0.6
    elif profile == "Study and learn":
        style = "Act like a friendly tutor. Explain step by step with examples."
        temp = 0.4
    else:
        style = "Be a helpful AI assistant."
        temp = 0.4

    messages = [
                   {
                       "role": "system",
                       "content": style,
                   }
               ] + history

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temp,
    )
    return resp.choices[0].message.content

# =======================
# 4. STREAMLIT CHAT UI
# =======================

st.markdown(
    "<div class='chat-container'>"
    "<h2 style='color:#e5e7eb; font-weight:600;'>Hey Kunaal, what's on your mind today?</h2>",
    unsafe_allow_html=True,
)

# ---- Model / profile selector bar ----
model_profiles = [
    "Smart (RAG)",
    "Quick response",
    "Think deeper",
    "Study and learn",
    "Search (strict RAG)",
]

col_bar = st.container()
with col_bar:
    c1, c2 = st.columns([1, 5])
    with c1:
        st.markdown(
            "<div class='model-bar'><span class='model-label'>Mode</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("<div class='model-bar'>", unsafe_allow_html=True)
        selected_profile = st.selectbox(
            "",
            model_profiles,
            index=0,
            key="model_profile",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ---- Session state for chat history ----
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display history in bubble style
for msg in st.session_state["messages"]:
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown("<div class='chat-role'>You</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-user'>{content}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='chat-role'>Assistant</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-bot'>{content}</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Chat input at bottom
user_input = st.chat_input("Message")

if user_input:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Decide which engine to use based on selected_profile
    if selected_profile in ["Smart (RAG)", "Search (strict RAG)"]:
        answer = generate_rag_answer(user_input, profile=selected_profile, k=5)
    else:
        # General chat uses full history
        answer = generate_general_answer(st.session_state["messages"], profile=selected_profile)

    # Add assistant message
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Rerun to display updated chat
    st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)
