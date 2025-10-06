from typing import Annotated, Sequence, Union
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages

#if we use annotated we can add something like reduce_list which it was a tool, it will like a description

class AgentState(BaseModel):
    """State class for LangGraph conversation flow"""
    context_window: list[BaseMessage]  #typing.Annotated allows you to attach arbitrary metadata to a type hint, here is add_messsages
    # and Sequence is a type hint for an ordered, immutable collection (like list or tuple) that can be iterated over.

    query: Union[str,BaseMessage,int] #query can be a str or a BaseMessage, it's not a list we want only str or BaseMessage
    #pruned_message: Annotated[ToolMessage, add_messages] | None

#state is a Pydantic model (AgentState), not a dictionary. Pydantic models don't have a .get() method. we aren't able to pass it as dictionary like --> state['context_window'] we need to pass it like
# state.context_window