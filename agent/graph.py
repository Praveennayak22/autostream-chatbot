"""
LangGraph StateGraph definition for the Social-to-Lead Agentic Workflow.

Flow:
  START
    └─> classify_intent_node
          ├─ (collecting_lead=True) ──> lead_collection_node
          ├─ greeting               ──> greeting_node          ──> END
          ├─ product_inquiry        ──> rag_response_node       ──> END
          └─ high_intent_lead       ──> lead_collection_node
                                          ├─ (fields missing)  ──> END
                                          └─ (all collected)   ──> lead_capture_node ──> END
"""
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (
    classify_intent_node,
    greeting_node,
    rag_response_node,
    lead_collection_node,
    lead_capture_node,
)


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    """Choose the next node based on intent and collection status."""
    # Mid-collection: keep collecting regardless of latest intent
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return "lead_collection"
    intent = state.get("intent", "product_inquiry")
    if intent == "greeting":
        return "greeting"
    if intent == "high_intent_lead":
        return "lead_collection"
    return "rag_response"  # product_inquiry or fallback


def route_after_collection(state: AgentState) -> str:
    """Decide whether all lead fields are ready for capture."""
    info = state.get("lead_info") or {}
    if info.get("name") and info.get("email") and info.get("platform"):
        return "lead_capture"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """Compile and return the LangGraph agent."""
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("classify_intent", classify_intent_node)
    g.add_node("greeting", greeting_node)
    g.add_node("rag_response", rag_response_node)
    g.add_node("lead_collection", lead_collection_node)
    g.add_node("lead_capture", lead_capture_node)

    # Entry
    g.add_edge(START, "classify_intent")

    # After intent classification → conditional branch
    g.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "greeting": "greeting",
            "rag_response": "rag_response",
            "lead_collection": "lead_collection",
        },
    )

    # Terminal nodes
    g.add_edge("greeting", END)
    g.add_edge("rag_response", END)

    # After lead_collection → either ask more or fire the tool
    g.add_conditional_edges(
        "lead_collection",
        route_after_collection,
        {
            "lead_capture": "lead_capture",
            END: END,
        },
    )

    g.add_edge("lead_capture", END)

    return g.compile()


# Singleton compiled graph
graph = build_graph()
