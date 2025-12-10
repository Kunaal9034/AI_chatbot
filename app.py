import os

import faiss
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# 0. Page config + Global CSS (Sidebar Commander Theme)
# ------------------------------------------------
st.set_page_config(
    page_title="AI-Powered College Enquiry Chatbot",
    page_icon="🤖",
    layout="wide",
)

# New CSS for Sidebar layout and Avatars
custom_css = """
<style>
/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Main content area background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #0f172a, #020617);
    color: #e5e7eb;
}

/* Sidebar background & border */
[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #1f2937;
}

/* Sidebar Title */
[data-testid="stSidebar"] h1 {
    text-align: center;
    font-weight: 700;
    font-size: 1.4rem;
    margin-bottom: 1.5rem;
    color: #ffffff;
}

/* --- Radio Buttons in Sidebar --- */
/* Container for radio options stack */
div[data-testid="stRadio"] > div {
    background: transparent;
    border: none;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.75rem; /* Space between pills */
}

/* Radio labels styled as stacked pills */
div[data-testid="stRadio"] label {
    background: #0b1120;
    border-radius: 8px;
    border: 1px solid #1f2937;
    padding: 0.75rem 1rem !important;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s ease-in-out;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* Hide default radio dot */
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}

/* Active pill state */
div[data-testid="stRadio"] label:has(input:checked) {
    background: #1e293b;
    border-color: #38bdf8;
    color: #ffffff;
    font-weight: 600;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

/* Inactive pill state */
div[data-testid="stRadio"] label:not(:has(input:checked)) {
    color: #9ca3af;
}
div[data-testid="stRadio"] label:not(:has(input:checked)):hover {
    border-color: #4b5563;
    background: #111827;
}


/* --- Chat Components with Avatars --- */

/* Flex container for a chat row (Avatar + Bubble) */
.chat-container {
    display: flex;
    align-items: flex-start; /* Align items to the top */
    margin: 1.2rem 0;
    gap: 0.75rem;
}

/* User message: reverse order to put avatar on right */
.chat-container.user {
    flex-direction: row-reverse;
}

/* Assistant message: normal order (avatar on left) */
.chat-container.assistant {
    flex-direction: row;
}

/* Avatar circle styling */
.avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0; /* Prevent avatar from squishing */
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Specific avatar colors */
.avatar.user { background-color: #3b82f6; } /* Blue for user */
.avatar.assistant { background-color: #10b981; } /* Green for bot */

/* Chat bubble styling */
.chat-bubble {
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    max-width: 80%; /* Do not span full width */
    line-height: 1.5;
    position: relative;
}

/* User bubble style */
.chat-bubble.user {
    background: #0b1120;
    border: 1px solid #1f2937;
    border-bottom-right-radius: 4px; /* Subtle corner tweak */
}

/* Assistant bubble style */
.chat-bubble.assistant {
    background: #020617;
    border: 1px solid #374151;
    border-bottom-left-radius: 4px; /* Subtle corner tweak */
}

/* --- Chat Input Styling --- */
/* Make input text area rounded and dark */
div[data-baseweb="textarea"] > textarea {
    border-radius: 24px !important;
    background-color: #0b1120 !important;
    border: 1px solid #374151 !important;
    color: #e5e7eb !important;
    padding: 0.8rem 1.2rem !important;
}
/* Focus state for input */
div[data-baseweb="textarea"] > textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
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
    st.error("GROQ_API_KEY not found in .env file. Please set it.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------------------------
# 2. Load FAQ dataset
# ------------------------------------------------
# Replace with your actual CSV path
DATA_PATH = "Data/galgotias_faq_3000.csv"

@st.cache_data
def load_faq_data(path: str):
    if not os.path.exists(path):
        st.warning(f"Data file not found at {path}. RAG mode will not work.")
        return pd.DataFrame(columns=["id", "question", "answer", "category"])
    return pd.read_csv(path)

faq_df = load_faq_data(DATA_PATH)

# ------------------------------------------------
# 3. Build docs + embeddings + FAISS (for RAG mode)
# ------------------------------------------------
@st.cache_resource
def build_rag_index(df: pd.DataFrame):
    if df.empty:
        return [], [], None, None

    docs = []
    meta = []

    for _, row in df.iterrows():
        q = str(row["question"])
        a = str(row["answer"])
        text = f"Q: {q}\nA: {a}"
        docs.append(text)
        meta.append({"id": row.get("id"), "category": row.get("category")})

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
    if embed_model is None or index is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    return [docs[i] for i in indices[0]]

def generate_rag_answer(query: str, k: int = 5) -> str:
    context_chunks = retrieve_context(query, k=k)
    if not context_chunks:
        return "I'm sorry, I couldn't find any relevant information in the FAQ database to answer your question."

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
    # Filter out tool calls or other non-text content if necessary for your API
    text_history = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history if isinstance(msg["content"], str)]
    messages += text_history

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
    )
    return resp.choices[0].message.content

# ------------------------------------------------
# 6. Streamlit Chat UI
# ------------------------------------------------

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🤖 College Enquiry Bot")
    st.markdown("---")
    st.write("Select Conversation Mode:")
    # Vertical radio buttons for sidebar
    mode = st.radio(
        "Mode Selection", # Hidden label due to CSS
        ["Colleges FAQ (RAG)", "General Chat"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #9ca3af; font-size: 0.8rem;'>
            Powered by Llama 3 & Groq
        </div>
        """, unsafe_allow_html=True
    )


# --- Main Chat Area ---

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Helper to render chat messages with avatars
def render_chat_message(role, content):
    # Choose avatar icon and CSS classes based on role
    if role == "user":
        avatar_icon = "👤"
        container_class = "user"
        avatar_class = "user"
        bubble_class = "user"
    else:
        avatar_icon = "🤖"
        container_class = "assistant"
        avatar_class = "assistant"
        bubble_class = "assistant"

    # Build the HTML structure
    html = f"""
    <div class="chat-container {container_class}">
        <div class="avatar {avatar_class}">{avatar_icon}</div>
        <div class="chat-bubble {bubble_class}">{content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Show chat history using the new render function
for msg in st.session_state["messages"]:
    render_chat_message(msg["role"], msg["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # 1. Add and render user message immediately
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_chat_message("user", user_input)

    # 2. Generate answer based on selected mode
    # Use a spinner to show activity while waiting for response
    with st.spinner("Thinking..."):
        if mode == "Colleges FAQ (RAG)":
            answer_text = generate_rag_answer(user_input, k=5)
        else:
            answer_text = generate_general_answer(st.session_state["messages"])

    # 3. Add and render assistant message
    st.session_state["messages"].append({"role": "assistant", "content": answer_text})
    render_chat_message("assistant", answer_text)

    # Rerun to update state correctly if needed, though st.chat_input usually handles this.
    # st.rerun()