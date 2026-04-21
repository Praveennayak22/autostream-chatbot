"""
AgentState — shared state across all LangGraph nodes.
"""
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class LeadInfo(TypedDict, total=False):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class AgentState(TypedDict):
    # Full conversation history; messages are APPENDED (reducer = operator.add)
    messages: Annotated[list[BaseMessage], operator.add]
    # Classified intent of the latest user turn
    intent: str
    # Incrementally collected lead fields
    lead_info: LeadInfo
    # True once mock_lead_capture() has fired
    lead_captured: bool
    # True while we are in the middle of collecting lead details
    collecting_lead: bool
