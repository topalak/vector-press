from pydantic import BaseModel
from langchain_core.messages import BaseMessage, ToolMessage

class AgentState(BaseModel):
    """State class for LangGraph conversation flow"""
    context_window: list[BaseMessage]  # keeps every type of message with BaseMessage
    query: str
    tool_messages: ToolMessage | None  # Can be ToolMessage or None, this is a blueprint, and we need to set what could it take

#state is a Pydantic model (AgentState), not a dictionary. Pydantic models don't have a .get() method. we aren't able to pass it as dictionary like --> state['context_window'] we need to pass it like
# state.context_window