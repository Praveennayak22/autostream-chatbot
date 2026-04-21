# 🎬 AutoStream AI Agent — Social-to-Lead Agentic Workflow

> **Machine Learning Intern Assignment** | ServiceHive × Inflx  
> Built with **LangGraph** · **gemini-2.0-flash** · **FAISS RAG** · **Streamlit**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A Google Gemini API key (free at [ai.google.dev](https://ai.google.dev))

### 1. Clone the repository
```bash
git clone https://github.com/Praveennayak22/autostream-chatbot.git
cd autostream-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the agent
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**. The FAISS vectorstore is built automatically on first run.

---

## 🎯 Demo Conversation Flow

Below is the exact conversation from the recorded demo:

| Turn | User Says | Agent Does |
|------|-----------|------------|
| 1 | `Hi` | Warm greeting, introduces AutoStream as an AI-powered video editing tool |
| 2 | `What is this?` | Explains AutoStream's product — features, and free trial offer |
| 3 | `What plans do you offer?` | RAG lookup → lists Free, Pro, and Enterprise plans with pricing |
| 4 | `Tell me more about the plam` | Understands intent despite typo → RAG retrieves full Pro plan feature details |
| 5 | `How can i get started?` | Detects **high-intent lead** 🔥 → starts collecting contact info, asks for name |
| 6 | `myself Praveen Kumar` | Extracts and saves name → asks for email |
| 7 | `email id : dazzlingdacchu16@gmail.com` | Extracts and saves email → asks for platform |
| 8 | `Youtube` | Saves platform → fires `mock_lead_capture()` ✅ → 🎯 Lead Captured! banner shown |
| 9 | `You havent tell about the plans` | Answers follow-up product question even after lead capture |
| 10 | `what about pricing` | RAG retrieves pricing details again |
| 11 | `okay then i will start with the free trail` | Acknowledges interest, confirms free trial details |
| 12 | `okay Thank you` | Warm closing response |

---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit state machines** with typed shared state (`AgentState`), conditional edges, and deterministic routing — exactly what a production lead-capture workflow needs. AutoGen is better suited for multi-agent debates; LangGraph excels at single-agent, multi-step workflows with strict guardrails.

### How State Is Managed

The `AgentState` TypedDict carries five fields across every graph node:

- **`messages`** — Full conversation history (LangChain `BaseMessage` list, appended via `operator.add` reducer so previous turns are never lost).
- **`intent`** — The classified label for the latest user turn (`greeting | product_inquiry | high_intent_lead`).
- **`lead_info`** — Dict incrementally updated as name, email, and platform are extracted from user replies.
- **`collecting_lead`** — Boolean flag that short-circuits the intent router: once lead collection begins, every subsequent turn flows to `lead_collection_node` until all fields are gathered.
- **`lead_captured`** — Prevents `mock_lead_capture()` from firing twice.

### RAG Pipeline

The knowledge base (`knowledge_base/autostream_kb.md`) is chunked with `RecursiveCharacterTextSplitter`, embedded via Google's `text-embedding-004` model, and stored in a local **FAISS** index. On each product query, the top-3 relevant chunks are retrieved and injected into the LLM prompt as grounded context.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp:

1. **Create a Meta Developer App** at [developers.facebook.com](https://developers.facebook.com) and enable the **WhatsApp Business API**.

2. **Set up a Webhook endpoint** (e.g., FastAPI):
   ```python
   @app.post("/webhook")
   async def whatsapp_webhook(payload: dict):
       sender = payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
       text   = payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

       # Retrieve or create per-user AgentState from a DB/Redis store
       state = get_user_state(sender)
       state["messages"] += [HumanMessage(content=text)]
       new_state = graph.invoke(state)
       save_user_state(sender, new_state)

       # Extract and send AI reply via WhatsApp API
       reply = [m for m in new_state["messages"] if isinstance(m, AIMessage)][-1].content
       send_whatsapp_message(sender, reply)
   ```

3. **Persist state per user** using Redis or a database keyed by WhatsApp phone number, so each user's conversation context survives across sessions.

4. **Register the webhook URL** in the Meta developer console and verify it with the hub challenge handshake.

5. **Deploy** the FastAPI service on a public HTTPS URL (e.g., Railway, Render, or AWS Lambda + API Gateway).

---

## 📁 Project Structure

```
autostream-agent/
├── agent/
│   ├── __init__.py
│   ├── state.py              # AgentState TypedDict
│   ├── tools.py              # mock_lead_capture()
│   ├── intent_classifier.py  # LLM-based intent classification
│   ├── nodes.py              # All 5 LangGraph node functions
│   └── graph.py              # StateGraph assembly + routing
├── knowledge_base/
│   └── autostream_kb.md      # RAG knowledge base (pricing, features, policies)
├── vectorstore/              # FAISS index (auto-generated on first run)
├── app.py                    # Streamlit chat UI
├── requirements.txt
├── .env                      # GOOGLE_API_KEY (not committed)
└── README.md
```

---

## 🔍 Evaluation Criteria Coverage

| Criterion | Implementation |
|-----------|---------------|
| Agent reasoning & intent detection | LLM classifier with strict prompt → 3 intents |
| Correct use of RAG | FAISS + Google embeddings + top-k retrieval |
| Clean state management | LangGraph `AgentState` with `operator.add` reducer |
| Proper tool calling logic | `mock_lead_capture` called **only** after all 3 fields |
| Code clarity & structure | Modular `agent/` package, typed state, clear docstrings |
| Real-world deployability | WhatsApp webhook design documented above |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | gemini-2.0-flash |
| Orchestration | LangGraph 0.2+ |
| RAG | FAISS + LangChain + Google `text-embedding-004` |
| UI | Streamlit |
| State | LangGraph `StateGraph` with typed `AgentState` |
| Config | python-dotenv |
