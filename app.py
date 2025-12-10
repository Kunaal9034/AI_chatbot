import os
import faiss
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# 0. Page config + CYBERPUNK CSS
# ------------------------------------------------
st.set_page_config(
    page_title="AI-Powered College Enquiry Chatbot",
    page_icon="🤖",
    layout="wide",
)

cyberpunk_css = """
<style>
/* Import Monospace Font */
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

/* Reset & Global Styles */
:root {
    --neon-green: #39ff14;
    --neon-cyan: #00ffff;
    --bg-black: #000000;
}

/* Main container background and font */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-black) !important;
    font-family: 'Roboto Mono', monospace !important;
    color: var(--neon-green);
}
[data-testid="stHeader"] {
    background-color: transparent;
}

/* Hide default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* --- Centered, Glowing Title --- */
h1 {
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--neon-green);
    text-shadow: 0 0 10px var(--neon-green), 0 0 20px var(--neon-green);
    border: 3px solid var(--neon-green);
    box-shadow: 0 0 15px var(--neon-green), inset 0 0 10px var(--neon-green);
    padding: 15px 30px;
    display: inline-block;
    margin-top: 1rem;
    /* Angular cut corners via clip-path */
    clip-path: polygon(5% 0, 95% 0, 100% 20%, 100% 80%, 95% 100%, 5% 100%, 0 80%, 0 20%);
}
/* Center the title container */
.stMainBlockContainer > div:first-child {
    display: flex;
    justify-content: center;
    margin-bottom: 2rem;
}

/* --- Glowing Radio Buttons (Mode Selection) --- */
div[data-testid="stRadio"] > div {
    gap: 1.5rem;
    justify-content: center;
}

/* Default radio label style (inactive) */
div[data-testid="stRadio"] label {
    background: #000;
    border: 2px solid var(--neon-green);
    color: var(--neon-green);
    font-family: 'Roboto Mono', monospace !important;
    text-transform: uppercase;
    padding: 0.8rem 2rem !important;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 0 5px var(--neon-green);
    /* Squarish angular shape */
    clip-path: polygon(10% 0, 90% 0, 100% 30%, 100% 70%, 90% 100%, 10% 100%, 0 70%, 0 30%);
}

/* Hide default dot */
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}

/* Active/Checked state */
div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(57, 255, 20, 0.1);
    box-shadow: 0 0 20px var(--neon-green), inset 0 0 10px var(--neon-green);
    text-shadow: 0 0 8px var(--neon-green);
    font-weight: 700;
}
/* Hover state */
div[data-testid="stRadio"] label:hover {
    box-shadow: 0 0 25px var(--neon-green);
}

/* --- Chat Layout & Avatars --- */
.chat-row {
    display: flex;
    align-items: flex-start; /* Align to top */
    margin: 2rem auto;
    gap: 1.5rem;
    max-width: 900px; /* Keep chat centered on wide screens */
}
.chat-row.user { flex-direction: row-reverse; }
.chat-row.assistant { flex-direction: row; }

/* Angular Avatar Container */
.avatar-box {
    flex-shrink: 0;
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid;
    background: #000;
    font-size: 2rem;
    /* Hexagonal/Angular shape */
    clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
}
.avatar-box.user {
    border-color: var(--neon-green);
    box-shadow: 0 0 15px var(--neon-green);
    color: var(--neon-green);
    text-shadow: 0 0 5px var(--neon-green);
}
.avatar-box.assistant {
    border-color: var(--neon-cyan);
    box-shadow: 0 0 15px var(--neon-cyan);
    color: var(--neon-cyan);
    text-shadow: 0 0 5px var(--neon-cyan);
}

/* --- Neon Chat Bubbles --- */
.neon-bubble {
    padding: 1.2rem 1.5rem;
    border: 2px solid;
    background: #000;
    max-width: 80%;
    line-height: 1.4;
    position: relative;
    /* Angular speech bubble shape */
    clip-path: polygon(0 0, 100% 0, 100% 85%, 95% 100%, 5% 100%, 0 85%);
}

.neon-bubble.user {
    border-color: var(--neon-green);
    color: var(--neon-green);
    /* Green glow inset and outset */
    box-shadow: inset 0 0 10px rgba(57, 255, 20, 0.3), 0 0 15px rgba(57, 255, 20, 0.4);
    text-shadow: 0 0 2px rgba(57, 255, 20, 0.5);
}

.neon-bubble.assistant {
    border-color: var(--neon-cyan);
    color: var(--neon-cyan);
    /* Cyan glow inset and outset */
    box-shadow: inset 0 0 10px rgba(0, 255, 255, 0.3), 0 0 15px rgba(0, 255, 255, 0.4);
    text-shadow: 0 0 2px rgba(0, 255, 255, 0.5);
}

/* --- Glowing Chat Input --- */
/* Target the fixed container at bottom */
.stChatInput {
    bottom: 20px !important; /* Lift it slightly */
}
/* The actual input box container */
[data-testid="stChatInput"] {
    border: 3px solid var(--neon-green) !important;
    background: #000 !important;
    border-radius: 0px !important; /* Square corners */
    /* Intense green glow */
    box-shadow: 0 0 20px var(--neon-green), inset 0 0 10px var(--neon-green) !important;
    /* Angular cut corners */
    clip-path: polygon(2% 0, 98% 0, 100% 20%, 100% 80%, 98% 100%, 2% 100%, 0 80%, 0 20%);
}

/* The textarea inside */
div[data-baseweb="textarea"] > textarea {
    background: transparent !important;
    color: var(--neon-green) !important;
    font-family: 'Roboto Mono', monospace !important;
    caret-color: var(--neon-green); /* The typing cursor */
}
/* Placeholder text color */
div[data-baseweb="textarea"] > textarea::placeholder {
    color: rgba(57, 255, 20, 0.5) !important;
}
/* The send button */
button[data-testid="stChatInputSubmitButton"] {
    color: var(--neon-green) !important;
}
button[data-testid="stChatInputSubmitButton"]:hover {
    text-shadow: 0 0 15px var(--neon-green);
}
</style>
"""
st.markdown(cyberpunk_css, unsafe_allow_html=True)

# ------------------------------------------------
# 1. Env + Groq client
# ------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------------------------
# 2. Load FAQ dataset
# ------------------------------------------------
DATA_PATH = "Data/galgotias_faq_3000.csv"

@st.cache_data
def load_faq_data(path: str):
    if not os.path.exists(path):
        st.warning(f"Data file not found: {path}")
        return pd.DataFrame(columns=["id", "question", "answer", "category"])
    return pd.read_csv(path)

faq_df = load_faq_data(DATA_PATH)

# ------------------------------------------------
# 3. Build RAG index
# ------------------------------------------------
@st.cache_resource
def build_rag_index(df: pd.DataFrame):
    if df.empty:
        return [], [], None, None
    docs = []
    meta = []
    for _, row in df.iterrows():
        text = f"Q: {str(row['question'])}\nA: {str(row['answer'])}"
        docs.append(text)
        meta.append({"id": row.get("id")})

    embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = embed_model.encode(
        docs, show_progress_bar=False, convert_to_numpy=True
    ).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return docs, meta, embed_model, index

docs, metadata, embed_model, index = build_rag_index(faq_df)

# ------------------------------------------------
# 4. RAG & Chat Helpers
# ------------------------------------------------
def generate_rag_answer(query: str, k: int = 5) -> str:
    if embed_model is None or index is None:
        return "RAG system not initialized."
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    context_chunks = [docs[i] for i in indices[0]]
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a futuristic enquiry bot for Galgotias University.
Use ONLY the provided CONTEXT to answer. If the answer is not present, state that data is unavailable in the archives.

CONTEXT ARCHIVE:
{context}

USER QUERY:
{query}

TERMINAL OUTPUT:
"""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

def generate_general_answer(chat_history: list[dict]) -> str:
    messages = [{"role": "system", "content": "You are a helpful AI assistant with a futuristic, terminal-like persona."}]
    # Clean history for API
    clean_history = [{"role": m["role"], "content": m["content"]} for m in chat_history]
    messages += clean_history
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
    )
    return resp.choices[0].message.content

# ------------------------------------------------
# 5. Streamlit Chat UI
# ------------------------------------------------

# Title Area
st.title("AI-Powered College Enquiry Chatbot")

# Mode Selector Pills
st.write("") # Spacer
mode = st.radio(
    "Select Mode",
    ["Colleges FAQ (RAG)", "General Chat"],
    horizontal=True,
    label_visibility="collapsed"
)
st.write("") # Spacer

# Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Custom Render Function for Neon Cyberpunk style ---
def render_neon_message(role, content):
    if role == "user":
        row_class = "user"
        avatar_class = "user"
        bubble_class = "user"
        # Using emojis that fit the aesthetic
        avatar_icon = "🧑‍💻"
    else:
        row_class = "assistant"
        avatar_class = "assistant"
        bubble_class = "assistant"
        avatar_icon = "🤖"

    html = f"""
    <div class="chat-row {row_class}">
        <div class="avatar-box {avatar_class}">
            <div>{avatar_icon}</div>
        </div>
        <div class="neon-bubble {bubble_class}">
            {content}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# Display Chat History
for msg in st.session_state["messages"]:
    render_neon_message(msg["role"], msg["content"])

# Chat Input
user_input = st.chat_input("ENTER COMMAND...")

if user_input:
    # Render user msg instantly
    st.session_state["messages"].append({"role": "user", "content": user_input})
    render_neon_message("user", user_input)

    # Generate Response
    with st.spinner("PROCESSING..."):
        if mode == "Colleges FAQ (RAG)":
            answer_text = generate_rag_answer(user_input)
        else:
            answer_text = generate_general_answer(st.session_state["messages"])

    # Render assistant msg
    st.session_state["messages"].append({"role": "assistant", "content": answer_text})
    render_neon_message("assistant", answer_text)