"""
LangGraph node functions.
Each node receives the full AgentState and returns a dict of state updates.
"""
import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.state import AgentState
from agent.intent_classifier import classify_intent
from agent.tools import mock_lead_capture

load_dotenv()

# ---------------------------------------------------------------------------
# Shared LLM instance
# ---------------------------------------------------------------------------
_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            convert_system_message_to_human=True,
        )
    return _llm


# ---------------------------------------------------------------------------
# FAISS vectorstore (built lazily)
# ---------------------------------------------------------------------------
_vectorstore = None


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_or_load_vectorstore()
    return _vectorstore


def _build_or_load_vectorstore() -> FAISS:
    vs_path = Path("vectorstore")
    kb_path = Path("knowledge_base/autostream_kb.md")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    if vs_path.exists() and (vs_path / "index.faiss").exists():
        try:
            return FAISS.load_local(
                str(vs_path), embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            pass  # Rebuild if loading fails

    text = kb_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    docs = splitter.create_documents([text])
    vs = FAISS.from_documents(docs, embeddings)
    vs_path.mkdir(exist_ok=True)
    vs.save_local(str(vs_path))
    return vs


# ---------------------------------------------------------------------------
# Helper: build a short conversation history string
# ---------------------------------------------------------------------------
def _history_str(messages: list, n: int = 6) -> str:
    recent = messages[-n:] if len(messages) > n else messages
    lines = []
    for m in recent:
        role = "User" if m.type == "human" else "AutoStream Agent"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: extract lead fields from a single user message
# ---------------------------------------------------------------------------
def _extract_lead_fields(user_msg: str, existing: dict, llm) -> dict:
    prompt = f"""\
Extract any of the following from the user's message (return null if absent):
- name  : the person's name
- email : email address
- platform : content platform (YouTube, Instagram, TikTok, Twitter, LinkedIn, Facebook, etc.)

User message: "{user_msg}"

Return ONLY a valid JSON object. Example:
{{"name": "Jane", "email": null, "platform": "YouTube"}}
JSON:"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip().strip("```json").strip("```").strip()
        extracted = json.loads(raw)
        updated = dict(existing)
        for key in ("name", "email", "platform"):
            val = extracted.get(key)
            if val and not updated.get(key):
                updated[key] = val
        return updated
    except Exception:
        # Regex fallback for email
        updated = dict(existing)
        m = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", user_msg)
        if m and not updated.get("email"):
            updated["email"] = m.group()
        return updated


# ===========================================================================
# NODE 1 — classify_intent_node
# ===========================================================================
def classify_intent_node(state: AgentState) -> dict:
    """Classify the latest user message intent (no output message)."""
    intent = classify_intent(state["messages"], get_llm())
    return {"intent": intent}


# ===========================================================================
# NODE 2 — greeting_node
# ===========================================================================
def greeting_node(state: AgentState) -> dict:
    """Handle casual greetings with a warm, branded response."""
    last_user = state["messages"][-1].content if state["messages"] else "Hello"
    resp = get_llm().invoke([
        SystemMessage(content=(
            "You are AutoStream's friendly AI assistant for content creators. "
            "AutoStream uses AI to automate video editing — saving creators hours every week. "
            "Greet the user warmly, introduce yourself in one sentence, and invite them to "
            "ask about plans, features, or to get started. Keep it to 2–3 sentences."
        )),
        HumanMessage(content=last_user),
    ])
    return {"messages": [AIMessage(content=resp.content)]}


# ===========================================================================
# NODE 3 — rag_response_node
# ===========================================================================
def rag_response_node(state: AgentState) -> dict:
    """Answer product/pricing questions using RAG over the knowledge base."""
    last_user = state["messages"][-1].content
    vs = get_vectorstore()
    docs = vs.similarity_search(last_user, k=3)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    history = _history_str(state["messages"][:-1])

    resp = get_llm().invoke([
        SystemMessage(content=(
            f"You are AutoStream's knowledgeable AI assistant.\n"
            f"Answer questions using ONLY the context below. Be accurate and conversational.\n"
            f"If the answer is not in the context, say so honestly.\n\n"
            f"=== AutoStream Knowledge Base ===\n{context}\n\n"
            f"=== Recent Conversation ===\n{history}"
        )),
        HumanMessage(content=last_user),
    ])
    return {"messages": [AIMessage(content=resp.content)]}


# ===========================================================================
# NODE 4 — lead_collection_node
# ===========================================================================
def lead_collection_node(state: AgentState) -> dict:
    """
    Incrementally collect name → email → platform.
    Tries to extract info from the latest user message before asking for more.
    Does NOT call mock_lead_capture (that is lead_capture_node's job).
    """
    existing = state.get("lead_info") or {}
    last_user = state["messages"][-1].content if state["messages"] else ""

    # Extract whatever the user just shared
    updated = _extract_lead_fields(last_user, existing, get_llm())

    # Determine what is still missing
    missing = [f for f in ("name", "email", "platform") if not updated.get(f)]

    if not missing:
        # All fields collected — just update state; graph will route to lead_capture
        return {"lead_info": updated, "collecting_lead": True}

    # Ask for the next missing field
    next_field = missing[0]
    name_so_far = updated.get("name", "")

    questions = {
        "name": (
            "✨ **Great choice!** I'd love to help you get started with AutoStream.\n\n"
            "First, what's your name?"
        ),
        "email": (
            f"Nice to meet you, **{name_so_far}**! 😊\n\n"
            "What's your email address so we can set up your account?"
        ),
        "platform": (
            "Almost there! 🎉\n\n"
            "Which platform do you primarily create content for? "
            "(e.g., YouTube, Instagram, TikTok, LinkedIn…)"
        ),
    }
    reply = questions[next_field]

    return {
        "messages": [AIMessage(content=reply)],
        "lead_info": updated,
        "collecting_lead": True,
    }


# ===========================================================================
# NODE 5 — lead_capture_node
# ===========================================================================
def lead_capture_node(state: AgentState) -> dict:
    """Fire mock_lead_capture() and confirm to the user."""
    info = state.get("lead_info") or {}
    name = info.get("name", "")
    email = info.get("email", "")
    platform = info.get("platform", "")

    # ✅ Tool execution — called ONLY when all three fields are present
    mock_lead_capture(name, email, platform)

    reply = (
        f"🎯 **You're all set, {name}!**\n\n"
        f"Here's what we've captured:\n"
        f"- **Name:** {name}\n"
        f"- **Email:** {email}\n"
        f"- **Platform:** {platform}\n\n"
        f"Our team will reach out to **{email}** within 24 hours to activate your "
        f"**7-day free Pro trial**. 🚀\n\n"
        f"Feel free to ask me anything else about AutoStream!"
    )
    return {
        "messages": [AIMessage(content=reply)],
        "lead_captured": True,
        "collecting_lead": False,
    }
