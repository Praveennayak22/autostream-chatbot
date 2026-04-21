"""
Intent classifier — uses Gemini to label each user turn as one of:
  greeting | product_inquiry | high_intent_lead
"""
from langchain_core.messages import HumanMessage, SystemMessage

INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for AutoStream, an AI-powered video-editing SaaS.

Classify the user's LATEST message into exactly ONE of these three labels:

1. greeting        – Casual hello, how-are-you, small talk, off-topic messages.
2. product_inquiry – Questions about features, pricing, plans, policies, comparisons,
                     how the product works, or any informational request.
3. high_intent_lead – The user explicitly wants to try, sign up, subscribe, purchase,
                      get started, or shows a clear buying signal.

Rules
- Buying keywords (try / sign up / subscribe / get started / I want / I'd like to / purchase / buy) → high_intent_lead
- Any question about pricing, features, or policies → product_inquiry
- Everything else → greeting
- When uncertain between product_inquiry and high_intent_lead, prefer high_intent_lead if ANY buying signal exists.

Respond with ONLY one of these exact strings (lowercase, no punctuation):
greeting
product_inquiry
high_intent_lead
"""


def classify_intent(messages: list, llm) -> str:
    """Return the intent label for the latest user turn."""
    # Use the last 4 messages for context
    recent = messages[-4:] if len(messages) >= 4 else messages
    conversation = "\n".join(
        f"{'User' if m.type == 'human' else 'Agent'}: {m.content}"
        for m in recent
    )
    resp = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"Conversation:\n{conversation}\n\nLabel:"),
    ])
    label = resp.content.strip().lower()
    if label not in {"greeting", "product_inquiry", "high_intent_lead"}:
        label = "product_inquiry"
    return label
