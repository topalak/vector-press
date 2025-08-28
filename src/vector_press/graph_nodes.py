from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages

# TODO response_content, _ =     , we aren't using retrieved chunks which returning from process_query
#LETS KEEP IT FOR NOW, WE WILL DECIDE IN MEETING AT SUNDAY

class AgentState(TypedDict, total=False):
    """State class for LangGraph conversation flow"""
    messages: Annotated[Sequence[BaseMessage], add_messages] #keeps every type of message with BaseMessage
    retrieved_chunks: list[str]
    query: str

class GraphNodes:
    """Contains all LangGraph nodes"""

    def __init__(self, rag_processor):
        """Initialize with RAG processor"""
        self.rag_processor = rag_processor

    def user_input_node(self, state: AgentState) -> AgentState:
        """User input node"""
        user_input = input("\nYou: ")
        return {
            **state,
            'messages': [HumanMessage(content=user_input)]
        }

    def generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response using RAG processor"""
        user_query = state['messages'][-1].content
        
        # Use RAG processor to handle the query
        response_content, _ = self.rag_processor.process_query(user_query, state['messages'][:-1])   #passing user's current query and every message in converstaion history without users query
        
        return {
            **state,
            'messages': [AIMessage(content=response_content)]
        }

    def should_exit(self, state: AgentState) -> str:
        """Conditional routing function"""
        messages = state['messages']
        last_message = messages[-1]
        
        if isinstance(last_message, HumanMessage) and last_message.content.lower() == "exit":
            return "end"
        else:
            return "continue"