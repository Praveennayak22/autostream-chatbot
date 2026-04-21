"""
AutoStream AI Agent — Streamlit Chat UI
Run: streamlit run app.py
"""
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoStream AI Agent",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0e1a; color: #e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #0b0e1a 100%);
    border-right: 1px solid #1e2a45;
}

/* ── Header ── */
.hero-header {
    background: linear-gradient(135deg, #6c63ff 0%, #a855f7 50%, #ec4899 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(108,99,255,0.35);
}
.hero-header h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0; }
.hero-header p  { color: rgba(255,255,255,0.85); font-size: 0.95rem; margin: 6px 0 0; }

/* ── Chat messages ── */
.chat-container { display: flex; flex-direction: column; gap: 14px; margin-bottom: 20px; }

.msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 75%;
    box-shadow: 0 4px 16px rgba(108,99,255,0.3);
    font-size: 0.95rem;
    line-height: 1.55;
}

.msg-agent {
    align-self: flex-start;
    background: #161d33;
    color: #e2e8f0;
    border: 1px solid #1e2a45;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    max-width: 80%;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ── Lead success banner ── */
.lead-success {
    background: linear-gradient(135deg, #059669, #10b981);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(16,185,129,0.4);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 4px 20px rgba(16,185,129,0.4); }
    50%       { box-shadow: 0 4px 32px rgba(16,185,129,0.7); }
}

/* ── Intent badge ── */
.intent-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.intent-greeting       { background: #1e3a5f; color: #60a5fa; }
.intent-product_inquiry { background: #1e2a45; color: #a78bfa; }
.intent-high_intent_lead{ background: #1a3a2a; color: #34d399; }

/* ── Lead progress card ── */
.lead-card {
    background: #161d33;
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 10px;
}
.lead-card h4 { color: #a78bfa; font-size: 0.8rem; text-transform: uppercase;
                letter-spacing: 0.08em; margin: 0 0 10px; }
.lead-field { display: flex; align-items: center; gap: 8px;
              font-size: 0.85rem; margin: 4px 0; color: #94a3b8; }
.lead-field.filled { color: #34d399; }
.lead-dot { width: 8px; height: 8px; border-radius: 50%; background: #374151; flex-shrink: 0; }
.lead-dot.filled { background: #34d399; }

/* ── Input area ── */
.stChatInputContainer { background: #161d33 !important; border-top: 1px solid #1e2a45; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Lazy graph import (so Streamlit doesn't crash during CSS render)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔧 Initialising AI agent…")
def load_graph():
    from agent.graph import graph
    return graph


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session():
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {
            "messages": [],
            "intent": "",
            "lead_info": {},
            "lead_captured": False,
            "collecting_lead": False,
        }
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []  # list of {"role", "content"}
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = ""


init_session()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 AutoStream Agent")
    st.markdown("*AI-powered video editing for content creators.*")
    st.divider()

    st.markdown("### 🤖 Agent Status")
    state = st.session_state.agent_state
    turns = sum(1 for m in state["messages"] if m.type == "human")
    st.metric("Conversation Turns", turns)

    intent = st.session_state.last_intent
    if intent:
        badge_map = {
            "greeting": ("👋 Greeting", "#60a5fa"),
            "product_inquiry": ("🔍 Product Inquiry", "#a78bfa"),
            "high_intent_lead": ("🔥 High-Intent Lead!", "#34d399"),
        }
        label, color = badge_map.get(intent, ("—", "#94a3b8"))
        st.markdown(f"**Last Intent:** <span style='color:{color}'>{label}</span>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📋 Lead Progress")
    info = state.get("lead_info") or {}
    fields = [("Name", "name"), ("Email", "email"), ("Platform", "platform")]
    for label_f, key in fields:
        val = info.get(key)
        if val:
            st.markdown(f"✅ **{label_f}:** {val}")
        else:
            st.markdown(f"⬜ **{label_f}:** —")

    if state.get("lead_captured"):
        st.success("🎯 Lead captured!")

    st.divider()
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.agent_state = {
            "messages": [], "intent": "", "lead_info": {},
            "lead_captured": False, "collecting_lead": False,
        }
        st.session_state.display_messages = []
        st.session_state.last_intent = ""
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Gemini 1.5 Flash · LangGraph · FAISS")


# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
  <h1>🎬 AutoStream AI Agent</h1>
  <p>Ask about plans, features, or get started with your free trial</p>
</div>
""", unsafe_allow_html=True)

# Lead captured banner (if applicable)
if st.session_state.agent_state.get("lead_captured"):
    st.markdown('<div class="lead-success">🎯 Lead Captured Successfully! Our team will be in touch soon.</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.display_messages:
    st.markdown("""
    <div class="msg-agent">
        👋 Hi there! I'm the <strong>AutoStream AI Assistant</strong>.<br><br>
        I can help you with pricing, features, and getting started with your
        <strong>7-day free Pro trial</strong>. What would you like to know? 🎬
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.display_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Render markdown inside agent bubble using st.markdown in a container
            with st.container():
                st.markdown(f'<div class="msg-agent">', unsafe_allow_html=True)
                st.markdown(msg["content"])
                st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if user_input := st.chat_input("Type your message…"):
    # 1. Add user message to display & state
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    current_state = st.session_state.agent_state
    current_state["messages"] = current_state["messages"] + [HumanMessage(content=user_input)]

    # 2. Run the LangGraph agent
    try:
        with st.spinner("AutoStream is thinking…"):
            agent = load_graph()
            new_state = agent.invoke(current_state)

        # 3. Persist updated state
        st.session_state.agent_state = new_state
        st.session_state.last_intent = new_state.get("intent", "")

        # 4. Extract the last AI message and add to display
        ai_messages = [m for m in new_state["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            last_ai = ai_messages[-1].content
            st.session_state.display_messages.append({"role": "agent", "content": last_ai})

    except Exception as e:
        err_str = str(e)
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            friendly = (
                "⚠️ **API Rate Limit Reached**\n\n"
                "The Gemini free-tier daily quota has been exhausted for today. "
                "Please try again after midnight (Pacific Time) or upgrade to a paid API plan at "
                "[ai.google.dev](https://ai.google.dev).\n\n"
                "*The agent logic is fully functional — this is only a quota issue.*"
            )
        else:
            friendly = f"⚠️ **Agent Error:** {err_str[:300]}"
        st.session_state.display_messages.append({"role": "agent", "content": friendly})
        # Remove the user message we just added from internal state since we couldn't process it
        current_state["messages"] = current_state["messages"][:-1]
        st.session_state.agent_state = current_state

    st.rerun()
