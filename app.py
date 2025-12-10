import os

import faiss
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# PAGE CONFIG + GLOBAL CSS
# ------------------------------------------------
st.set_page_config(
    page_title="AI-Powered College Enquiry Chatbot",
    page_icon="🤖",
    layout="wide",
)

custom_css = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #020617 0, #020617 60%, #000000 100%);
    color: #e5e7eb;
}

/* Main centered wrapper (title + mode + chat) */
.main-chat {
    max-width: 900px;
    margin: 0 auto;
    padding-top: 2.5rem;
}

/* Title styling */
.main-chat h1 {
    text-align: left;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: 0.02em;
}

/* Mode label */
.mode-label {
    margin-top: 2rem;
    margin-bottom: 0.35rem;
    font-size: 0.95rem;
}

/* Select box styling (mode dropdown) */
.mode-selectbox div[data-baseweb="select"] {
    background: #020617;
    border-radius: 999px;
    border: 1px solid #1f2937;
    padding: 0.15rem 0.25rem;
    color: #e5e7eb;
}
.mode-selectbox div[data-baseweb="select"] > div {
    min-height: 3rem;
}
/* Selected value inside select */
.mode-selectbox span {
    font-size: 0.95rem;
}

/* Thin separator line under mode */
.chat-separator {
    margin-top: 1.3rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #111827;
}

/* Chat bubbles */
.chat-row {
    margin: 0.75rem 0;
}
.chat-label {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-bottom: 0.2rem;
}
.chat-user {
    background: #020617;
    border-radius: 18px;
    padding: 0.9rem 1rem;
}
.chat-assistant {
    background: #020617;
    border-radius: 18px;
    padding: 0.9rem 1rem;
    border: 1px solid #1f2937;
}

/* Chat input: pill shape + a bit lifted */
div[data-baseweb="textarea"] > textarea {
    border-radius: 999px !important;
}
.block-container {
    padding-bottom: 6rem; /* space above bottom input */
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------------------------------------
# 1. Env + Groq client (works locally + Streamlit Cloud)
# ------------------------------------------------
# Try Streamlit secrets first (for cloud)
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Fallback: .env for local dev
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Set in .env or Streamlit secrets.")

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------------------------
# 2. Load FAQ dataset
# ------------------------------------------------
DATA_PATH = "Data/galgotias_faq_3000.csv"   # keep this as your 3000-row CSV

@st.cache_data
def load_faq_data(path: str):
    return pd.read_csv(path)

faq_df = load_faq_data(DATA_PATH)

# ------------------------------------------------
# 3. Build docs + embeddings + FAISS (for RAG mode)
# ------------------------------------------------
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
    embeddings = embed_model.encode(
        docs, show_progress_bar=False, convert_to_numpy=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return docs, meta, embed_model, index

docs, metadata, embed_model, index = build_rag_index(faq_df)

# ------------------------------------------------
# 4. RAG helper (Colleges FAQ mode)
# ------------------------------------------------
def retrieve_context(query: str, k: int = 5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    return [docs[i] for i in indices[0]]

def generate_rag_answer(query: str, k: int = 5) -> str:
    context_chunks = retrieve_context(query, k=k)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an enquiry bot for Galgotias University.
Primarily use ONLY the information in the context to answer the question.
If the answer is clearly not in the context, then you may answer using your own knowledge,
but mention that the information is approximate.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# ------------------------------------------------
# 5. General chat helper (ChatGPT-like mode)
# ------------------------------------------------
def generate_general_answer(chat_history: list[dict]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant like ChatGPT. "
                "You can answer general questions about programming, maths, reasoning, etc. "
                "If user asks specifically about Galgotias University, answer as best as you can "
                "even without the FAQ context."
            ),
        }
    ]
    messages += chat_history

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
    )
    return resp.choices[0].message.content

# ------------------------------------------------
# 6. Streamlit Chat UI (to match your first screenshot)
# ------------------------------------------------

# Start centered wrapper
st.markdown('<div class="main-chat">', unsafe_allow_html=True)

st.markdown("<h1>🤖 AI-Powered College Enquiry Chatbot</h1>", unsafe_allow_html=True)

# Mode dropdown
st.markdown('<div class="mode-label">Select mode:</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="mode-selectbox">', unsafe_allow_html=True)
    mode = st.selectbox(
        "",
        ["Colleges FAQ (RAG)", "General Chat (ChatGPT-like)"],
        key="mode_select",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="chat-separator"></div>', unsafe_allow_html=True)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display existing messages as chat bubbles
for msg in st.session_state["messages"]:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="chat-label">user</div>
                <div class="chat-user">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row">
                <div class="chat-label">assistant</div>
                <div class="chat-assistant">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# End wrapper (content part)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# 7. Bottom chat input (full width)
# ------------------------------------------------
user_input = st.chat_input("Ask any question...")

if user_input:
    # 1. Add user message
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )

    # 2. Generate answer based on selected mode
    if mode == "Colleges FAQ (RAG)":
        answer_text = generate_rag_answer(user_input, k=5)
    else:
        answer_text = generate_general_answer(st.session_state["messages"])

    # 3. Add assistant message
    st.session_state["messages"].append(
        {"role": "assistant", "content": answer_text}
    )

    # 4. Show latest assistant bubble (inside centered wrapper)
    st.markdown(
        f"""
        <div class="main-chat">
            <div class="chat-row">
                <div class="chat-label">assistant</div>
                <div class="chat-assistant">{answer_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
