
from typing import Any, Dict, List, Optional, TypedDict

# Environment variables
from dotenv import load_dotenv
# LangChain and Google AI specific libraries
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage
)

class GraphState(TypedDict):
    messages: List[BaseMessage]
    table: Optional[Dict[str, Any]]
    answer: str
    bibliography: AIMessage
    research_queries: list
    finished: bool
    original_query: Optional[HumanMessage]
    request: Optional[AIMessage]
    answer_source: Optional[str]
    trys: int
    history: List[str]