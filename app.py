
import time
import json
import uuid
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntentRAG AI | Intelligent Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM UI STYLING
# ============================================================

st.markdown("""
<style>

/* Main application background */

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(79, 70, 229, 0.13),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(168, 85, 247, 0.10),
            transparent 28%
        ),
        #0b1020;
    color: #e5e7eb;
}

/* Main content width */

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #10172a;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc;
}

/* Hero section */

.hero-banner {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        120deg,
        #312e81,
        #6d28d9 55%,
        #9333ea
    );
    padding: 34px;
    border-radius: 22px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 15px 45px rgba(79,70,229,0.22);
}

.hero-banner h1 {
    color: white;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.hero-banner p {
    color: #e9d5ff;
    font-size: 0.98rem;
    margin-top: 12px;
    margin-bottom: 0;
    line-height: 1.7;
}

.hero-label {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.18);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 16px;
}

/* Section headings */

.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 14px;
}

/* Dashboard cards */

.dashboard-card {
    background: linear-gradient(
        145deg,
        rgba(30, 41, 59, 0.92),
        rgba(15, 23, 42, 0.95)
    );
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    padding: 18px;
    min-height: 110px;
}

.card-label {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-bottom: 8px;
}

.card-value {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 750;
}

.card-description {
    color: #94a3b8;
    font-size: 0.76rem;
    margin-top: 5px;
}

/* Status badges */

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 12px;
    border-radius: 20px;
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(16,185,129,0.25);
    color: #6ee7b7;
    font-size: 0.78rem;
    font-weight: 600;
}

.intent-badge {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 20px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(129,140,248,0.25);
    color: #c4b5fd;
    font-size: 0.76rem;
    font-weight: 600;
}

.time-badge {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 20px;
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(16,185,129,0.22);
    color: #6ee7b7;
    font-size: 0.76rem;
    font-weight: 600;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background: rgba(30, 41, 59, 0.58);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 15px;
    margin-bottom: 12px;
}

/* Chat input */

[data-testid="stChatInput"] {
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 16px;
}

/* Buttons */

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,0.2);
    background: #1e293b;
    color: #e2e8f0;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #a78bfa;
    color: white;
    background: #312e81;
    transform: translateY(-2px);
}

/* Download button */

.stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(139,92,246,0.35);
    background: rgba(109,40,217,0.18);
    color: #ddd6fe;
}

/* Dividers */

hr {
    border-color: rgba(148,163,184,0.15);
}

/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    padding-top: 30px;
    padding-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Welcome to **IntentRAG AI**!\n\n"
                "I'm your intelligent document assistant. "
                "Ask questions about your company policy or "
                "start a normal conversation.\n\n"
                "I'll automatically route your request through "
                "the appropriate processing chain."
            ),
            "latency": None
        }
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_response(question):

    answer = rag_chain.invoke({
        "input": question,
        "chat_history": st.session_state.chat_history[-10:]
    })

    if isinstance(answer, AIMessage):
        answer = answer.content

    return str(answer)


def reset_conversation():

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Welcome back to **IntentRAG AI**!\n\n"
                "Your conversation has been reset. "
                "What would you like to explore?"
            ),
            "latency": None
        }
    ]

    st.session_state.chat_history = []
    st.session_state.query_count = 0
    st.session_state.session_id = str(uuid.uuid4())


def export_chat():

    return json.dumps(
        st.session_state.messages,
        indent=4,
        ensure_ascii=False
    )


def word_stream(response):

    for word in response.split(" "):
        yield word + " "
        time.sleep(0.015)


def display_metadata(latency):

    st.markdown(
        f"""
        <div class="meta-row">
            <span class="intent-badge">
                🎯 Intent-based routing
            </span>
            <span class="time-badge">
                ⚡ {latency}s
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 5. SIDEBAR DASHBOARD
# ============================================================

with st.sidebar:

    st.markdown("## ✨ IntentRAG AI")

    st.caption("Advanced Intent-Routed RAG Platform")

    st.markdown(
        '<span class="status-badge">● System Ready</span>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 📊 Session Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Questions",
            st.session_state.query_count
        )

    with col2:
        st.metric(
            "Pipeline",
            "Active"
        )

    st.divider()

    st.markdown("### 🎨 Preferences")

    show_metadata = st.toggle(
        "Show response time",
        value=True
    )

    typing_animation = st.toggle(
        "Typing animation",
        value=True
    )

    st.divider()

    st.markdown("### ⚙️ System Architecture")

    with st.expander("AI Configuration", expanded=True):

        st.markdown("""
        **Language Model**

        Groq — `openai/gpt-oss-20b`

        **Embedding Model**

        `all-MiniLM-L6-v2`

        **Vector Database**

        Supabase Vector

        **Framework**

        LangChain

        **Retrieval**

        Top 3 similar chunks
        """)

    with st.expander("Intent Routing"):

        st.markdown("""
        - 📄 **DOCUMENT** — Knowledge-base questions
        - 💬 **HISTORY** — Conversation context
        - 🌐 **GENERAL** — General conversation
        """)

    st.divider()

    st.markdown("### 🛠️ Session Actions")

    st.download_button(
        label="📥 Export Conversation",
        data=export_chat(),
        file_name="intent_rag_chat.json",
        mime="application/json",
        use_container_width=True
    )

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        reset_conversation()
        st.rerun()

    st.divider()

    st.caption("Powered by LangChain · Groq · Supabase")


# ============================================================
# 6. HERO HEADER
# ============================================================

st.markdown("""
<div class="hero-banner">

    <div class="hero-label">
        ✨ INTELLIGENT DOCUMENT ASSISTANT
    </div>

    <h1>Advanced Intent-Based RAG</h1>

    <p>
        Ask questions. Retrieve knowledge. Get intelligent answers.
        <br>
        Semantic intent classification · Vector retrieval ·
        Conversational memory · Context-aware generation
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 7. DASHBOARD OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-heading">System Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-label">VECTOR DATABASE</div>
        <div class="card-value">Supabase</div>
        <div class="card-description">Semantic vector search</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-label">AI MODEL</div>
        <div class="card-value">Groq LLM</div>
        <div class="card-description">Intelligent response generation</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-label">RETRIEVAL ENGINE</div>
        <div class="card-value">LangChain</div>
        <div class="card-description">Intent-based RAG pipeline</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 8. QUICK PROMPTS
# ============================================================

if len(st.session_state.messages) == 1:

    st.markdown("")
    st.markdown(
        '<div class="section-heading">💡 Try asking</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    starters = [
        (
            "📄 Document Summary",
            "Give me a high-level summary of the company policy."
        ),
        (
            "🔍 Policy Details",
            "Explain the important policies mentioned in the document."
        ),
        (
            "⚡ Key Insights",
            "What are the most important takeaways from the document?"
        )
    ]

    for col, (label, query) in zip(
        [c1, c2, c3],
        starters
    ):

        with col:
            if st.button(
                label,
                use_container_width=True
            ):
                st.session_state.pending_prompt = query
                st.rerun()


# ============================================================
# 9. CHAT HISTORY
# ============================================================

st.markdown("")
st.markdown(
    '<div class="section-heading">💬 Conversation</div>',
    unsafe_allow_html=True
)

for msg in st.session_state.messages:

    avatar = "🧑‍💻" if msg["role"] == "user" else "✨"

    with st.chat_message(
        msg["role"],
        avatar=avatar
    ):

        if (
            msg["role"] == "assistant"
            and show_metadata
            and msg.get("latency") is not None
        ):
            display_metadata(msg["latency"])

        st.markdown(msg["content"])


# ============================================================
# 10. USER INPUT
# ============================================================

prompt = st.chat_input(
    "Ask a question about your documents..."
)

if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")


# ============================================================
# 11. RAG PIPELINE EXECUTION
# ============================================================

if prompt:

    prompt = prompt.strip()

    if not prompt:

        st.warning("Please enter a question.")

    else:

        st.session_state.query_count += 1

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message(
            "user",
            avatar="🧑‍💻"
        ):
            st.markdown(prompt)

        with st.chat_message(
            "assistant",
            avatar="✨"
        ):

            start_time = time.time()

            try:

                with st.status(
                    "Processing your request...",
                    expanded=True
                ) as status:

                    st.write("🔎 Classifying user intent...")
                    st.write("🔀 Selecting the appropriate processing chain...")
                    st.write("🗄️ Retrieving relevant knowledge from Supabase...")
                    st.write("🤖 Generating an answer using Groq...")

                    response_text = get_response(prompt)

                    elapsed = round(
                        time.time() - start_time,
                        2
                    )

                    status.update(
                        label="Response generated successfully",
                        state="complete",
                        expanded=False
                    )

                if show_metadata:
                    display_metadata(elapsed)

                if typing_animation:
                    st.write_stream(word_stream(response_text))
                else:
                    st.markdown(response_text)

                # Update conversation history

                st.session_state.chat_history.append(
                    HumanMessage(content=prompt)
                )

                st.session_state.chat_history.append(
                    AIMessage(content=response_text)
                )

                # Save assistant message

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "latency": elapsed
                })

            except Exception as e:

                st.error(
                    "Unable to process your request."
                )

                with st.expander("Technical Error Details"):
                    st.code(str(e))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        "I encountered an error while processing "
                        "your request. Please try again."
                    ),
                    "latency": None
                })


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    IntentRAG AI · Advanced Retrieval-Augmented Generation
    <br>
    Built with LangChain, Groq, Hugging Face and Supabase
</div>
""", unsafe_allow_html=True)
