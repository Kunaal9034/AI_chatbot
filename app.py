import os

import faiss
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -----------------------------
# 1. Env + Groq client
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# 2. Load FAQ dataset (3000 rows)
# -----------------------------
DATA_PATH = "Data/galgotias_faq_3000.csv"

@st.cache_data
def load_faq_data(path: str):
    return pd.read_csv(path)

faq_df = load_faq_data(DATA_PATH)

# -----------------------------
# 3. Build docs + embeddings + FAISS (for RAG mode)
# -----------------------------
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

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return docs, meta, embed_model, index

docs, metadata, embed_model, index = build_rag_index(faq_df)

# -----------------------------
# 4. RAG helper (Galgotias mode)
# -----------------------------
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

# -----------------------------
# 5. General chat helper (ChatGPT-like mode)
# -----------------------------
def generate_general_answer(chat_history: list[dict]) -> str:
    """
    chat_history: list of {"role": "user"/"assistant", "content": "..."}
    """
    # System instruction for general chatbot
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

# -----------------------------
# 6. Streamlit Chat UI
# -----------------------------
st.set_page_config(page_title="Galgotias + ChatGPT-like Bot", page_icon="🤖", layout="wide")

st.title("🤖 AI-Powered College Enquiry Chatbot ")

mode = st.radio(
    "Select mode:",
    ["Galgotias FAQ (RAG)", "General Chat (ChatGPT-like)"],
    horizontal=True,
)

st.markdown("---")

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask any question...")

if user_input:
    # 1. Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # 2. Generate answer based on selected mode
    if mode == "Galgotias FAQ (RAG)":
        answer_text = generate_rag_answer(user_input, k=5)
    else:
        # General chat: pass full history
        answer_text = generate_general_answer(st.session_state["messages"])

    # 3. Add assistant message to history
    st.session_state["messages"].append({"role": "assistant", "content": answer_text})

    # 4. Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer_text)
