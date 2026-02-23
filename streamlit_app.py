# ============================================================
#  streamlit_app.py
#
#   Streamlit UI for Multi-Agent System
#
#  Run BOTH of these in separate terminals:
#    Terminal 1: uvicorn Fast_api:app --reload
#    Terminal 2: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import requests

# ─────────────────────────────────────────────────────
#  FASTAPI BASE URL
# ─────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

AGENT_ICONS = {
    "ipc":         "⚖️  IPC Legal",
    "healthcare":  "🏥 Healthcare",
    "software":    "💻 Software",
    "out_of_scope":" !!! Out of Scope",
}


# ─────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Multi-Agent System",
    page_icon  = "🤖",
    layout     = "wide",
)


# ─────────────────────────────────────────────────────
#  SESSION STATE
#  Streamlit reruns entire script on every interaction.
#  st.session_state keeps data alive between reruns.
# ─────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None      # MongoDB session ID

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []     # [{role, content, agents}]


# ─────────────────────────────────────────────────────
#  API HELPER FUNCTIONS
# ─────────────────────────────────────────────────────
def api_chat(question: str, session_id: str):
    """
    POST /chat
    Sends question to FastAPI and returns the response dict.
    Returns None if API call fails.
    """
    try:
        res = requests.post(
            f"{API_URL}/chat",
            json={
                "question"  : question,
                "session_id": session_id or "",
            },
            timeout=120,    # LLM calls can take time
        )
        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"API Error {res.status_code}: {res.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI. Run: uvicorn Fast_api:app --reload")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Try again.")
        return None


def api_history(session_id: str):
    """
    GET /history/{session_id}
    Returns list of past Q&A records for this session.
    """
    try:
        res = requests.get(f"{API_URL}/history/{session_id}", timeout=10)
        if res.status_code == 200:
            return res.json().get("history", [])
        return []
    except Exception:
        return []


def api_health():
    """
    GET /health
    Returns True if FastAPI is running, False otherwise.
    """
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:

    st.title("🤖 Multi-Agent")
    st.caption("Powered by LangGraph + Azure GPT-4")
    st.divider()

    # ── API Status ──
    st.subheader("API Status")
    if api_health():
        st.success("FastAPI is running ✅")
    else:
        st.error("FastAPI is offline ❌")
        st.code("uvicorn Fast_api:app --reload", language="bash")

    st.divider()

    # ── Agents Info ──
    st.subheader("Available Agents")
    st.markdown("⚖️  **IPC Legal** — Penal Code, sections, crimes")
    st.markdown("🏥 **Healthcare** — Diseases, symptoms, treatments")
    st.markdown("💻 **Software** — Code, frameworks, tech")

    st.divider()

    # ── Session Info ──
    st.subheader("Session")
    if st.session_state.session_id:
        st.info(f"Session ID: `{st.session_state.session_id}`")
    else:
        st.caption("Session starts on first question")

    # ── New Session Button ──
    if st.button("🔄 Start New Session", use_container_width=True):
        st.session_state.session_id    = None
        st.session_state.chat_messages = []
        st.rerun()


# ─────────────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────────────
st.title("Multi-Agent System")
st.caption("Ask questions about IPC Law, Healthcare, or Software & Technology")
st.divider()

# ── TAB LAYOUT — Chat | History ──
tab_chat, tab_history = st.tabs(["💬 Chat", "📋 History"])


# ══════════════════════════════════════════════════════
#  TAB 1 — CHAT
# ══════════════════════════════════════════════════════
with tab_chat:

    # ── Display all chat messages ──
    if not st.session_state.chat_messages:
        st.info("Ask a question below to start the conversation!")

    for msg in st.session_state.chat_messages:

        if msg["role"] == "user":
            # User message
            with st.chat_message("user"):
                st.write(msg["content"])

        else:
            # Agent message
            with st.chat_message("assistant"):
                # Show which agents were used
                agents = msg.get("agents", [])
                if agents:
                    badges = "  ".join([
                        AGENT_ICONS.get(a, f"🤖 {a}") for a in agents
                    ])
                    st.caption(f"Answered by: {badges}")

                st.write(msg["content"])

    st.divider()

    # ── Input Box ──
    with st.form(key="chat_form", clear_on_submit=True):
        question = st.text_input(
            label       = "Your Question",
            placeholder = "e.g. What is IPC Section 302? / Symptoms of diabetes? / How to use FastAPI?",
        )
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    # ── Handle Submit ──
    if submitted and question.strip():

        # Add user message to chat
        st.session_state.chat_messages.append({
            "role"   : "user",
            "content": question.strip(),
        })

        # Call FastAPI
        with st.spinner("Thinking... (this may take a few seconds)"):
            result = api_chat(
                question   = question.strip(),
                session_id = st.session_state.session_id or "",
            )

        if result:
            # Save session_id returned from API
            if not st.session_state.session_id:
                st.session_state.session_id = result.get("session_id")

            # Add agent answer to chat
            st.session_state.chat_messages.append({
                "role"   : "assistant",
                "content": result.get("answer", "No answer received."),
                "agents" : result.get("agents_used", []),
            })

        # Rerun to refresh UI with new messages
        st.rerun()


# ══════════════════════════════════════════════════════
#  TAB 2 — HISTORY
# ══════════════════════════════════════════════════════
with tab_history:

    st.subheader("📋 Conversation History")

    if not st.session_state.session_id:
        st.info("No session started yet. Ask a question in the Chat tab first.")

    else:
        st.caption(f"Session ID: `{st.session_state.session_id}`")

        # Fetch history from FastAPI → MongoDB
        if st.button("🔃 Refresh History"):
            st.rerun()

        history = api_history(st.session_state.session_id)

        if not history:
            st.warning("No history found for this session yet.")
        else:
            st.success(f"Found {len(history)} conversation(s)")
            st.divider()

            # Display each record
            for i, record in enumerate(history, start=1):

                agents = record.get("agents_used", [])
                agent_labels = "  ".join([
                    AGENT_ICONS.get(a, f"🤖 {a}") for a in agents
                ])

                with st.expander(f"#{i} — {record.get('question', '')[:60]}..."):
                    st.markdown(f"**Question:**")
                    st.write(record.get("question", ""))

                    st.markdown(f"**Answer:**")
                    st.write(record.get("answer", ""))

                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"🤖 Agents: {agent_labels}")
                    with col2:
                        st.caption(f"🕐 Time: {record.get('timestamp', '')}")

                