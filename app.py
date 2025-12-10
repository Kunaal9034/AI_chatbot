import os

import faiss
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# 0. Page config  + global CSS (dark ChatGPT style)
# ------------------------------------------------
st.set_page_config(
    page_title="AI-Powered College Enquiry Chatbot",
    page_icon="🤖",
    layout="wide",
)

custom_css = """
<style>
/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* App background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0f172a 0, #020617 55%);
    color: #e5e7eb;
}

/* Centered big title */
h1 {
    text-align: center;
    font-weight: 700;
}

/* Mode selector container */
div[data-testid="stRadio"] > div {
    background: #020617;
    border-radius: 999px;
    border: 1px solid #1f2937;
    padding: 0.25rem 0.5rem;
}

/* Radio labels look like pills */
div[data-testid="stRadio"] label {
    border-radius: 999px;
    padding: 0.6rem 1.6rem !important;
    cursor: pointer;
}

/* Hide small radio dot */
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}

/* Active pill */
div[data-testid="stRadio"] label:has(input:checked) {
    background: #0f172a;
    border: 1px solid #38bdf8;
    color: #e5e7eb;
}

/* Inactive pill */
div[data-testid="stRadio"] label:not(:has(input:checked)) {
    color: #9ca3af;
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
    background: #0b1120;
    border-radius: 16px;
    padding: 0.9rem 1rem;
}
.chat-assistant {
    background: #020617;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    border: 1px solid #1f2937;
}

/* Chat input a bit rounded */
div[data-baseweb="textarea"] > textarea {
    border-radius: 999px !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------------------------------------
# 1. Env + Groq client
# ------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------------------------
# 2. Load FAQ dataset
# ------------------------------------------------
# yahan apna CSV path rakho (abhi Galgotias 3000)
DATA_PATH = "Data/galgotias_faq_3000.csv"

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
# 6. Streamlit Chat UI
# ------------------------------------------------
st.title("🤖 AI-Powered College Enquiry Chatbot")

# Mode selector (pills)
st.write("")
mode = st.radio(
    "Select mode:",
    ["Colleges FAQ (RAG)", "General Chat (ChatGPT-like)"],
    horizontal=True,
    index=0,
)
st.markdown("---")

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show chat history with custom bubbles
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

# Chat input
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

    # 4. Show last assistant message bubble immediately
    st.markdown(
        f"""
        <div class="chat-row">
            <div class="chat-label">assistant</div>
            <div class="chat-assistant">{answer_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
