from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from src.vector_press.db.supabase_db import SupabaseVectorStore
from src.vector_press.llm_embedding_initializer import LLMManager

INSTRUCTIONS = """You are a helpful AI assistant for The Guardian's articles. 
            The user's message doesn't seem to be asking about specific news information. 
            Respond naturally and helpfully, and suggest they can ask about subject to retrieving news, 
            technology, sport, politics etc.
            You are a helpful AI assistant for The Guardian's articles.

            If there are related chunks below, you will answer based on these chunks. If there are no relevant chunks 
            or the information is not in our database, politely let the user know and suggest they can ask about subjects 
            like technology, sports, politics, etc.
    
            Context from The Guardian articles:
            
            {context_text}

            Please respond helpfully based on the available information.
            
            """

class AgentState(TypedDict, total=False):
    """State class for LangGraph conversation flow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]  # keeps every type of message with BaseMessage
    retrieved_chunks: list[str]
    query: str


class RAGProcessor:
    """Handles RAG query processing and response generation"""

    def __init__(self, llm_manager: LLMManager, supabase_vector_store: SupabaseVectorStore):
        """Initialize with LLM manager and Supabase vector store"""
        self.llm = llm_manager.get_llm()  # Get LLM from manager
        self.supabase_vector_store = supabase_vector_store  # SupabaseVectorStore instance
        # Note: embedding_model available via supabase_vector_store if needed
    
    def initialize_state(self) -> AgentState:
        """Initialize state with INSTRUCTIONS as SystemMessage (called only once)"""
        return {
            "messages": [SystemMessage(content=INSTRUCTIONS)],
            "retrieved_chunks": [],
            "query": ""
        }

    def process_query(self, state: AgentState) -> AgentState:
        """Process user query with RAG and return updated state"""
        user_query = state['messages'][-1].content
        
        # Retrieve relevant chunks
        retrieved_chunks = self.supabase_vector_store.retrieve_relevant_chunks(user_query)
        
        # Create context text (empty if no chunks)
        context_text = "\n\n".join(retrieved_chunks) if retrieved_chunks else ""


        response = self.llm.invoke(state['messages'])
        print(f'\nBig Brother Bertan: {response.content}')

        #print(f'Current memory: {[*state["messages"], AIMessage(content=response.content)]}')

        # Return updated state
        return {
            **state,
            'messages': [
                state['messages'][-1],  # Keep the user message
                AIMessage(content=response.content)  # Add AI response
            ],
            'retrieved_chunks': retrieved_chunks,
            'query': user_query
        }

def should_continue(state: AgentState) -> str:
    """Conditional routing function"""
    messages = state['messages']
    last_message = messages[-1]

    if isinstance(last_message, HumanMessage) and last_message.content.lower() == "exit":
        return "end"
    else:
        return "continue"

def main():
    from langgraph.graph import StateGraph, START, END
    
    llm_manager = LLMManager()
    supabase_vector_store = SupabaseVectorStore(llm_manager)
    rag_processor = RAGProcessor(llm_manager, supabase_vector_store)

    # Build simplified graph with single node
    graph = StateGraph(AgentState)
    graph.add_node('process_query', rag_processor.process_query)

    # Simplified connections
    graph.add_edge(START, 'process_query')
    graph.add_conditional_edges('process_query', should_continue, {
        "continue": 'process_query',  # Loop back for next user input
        "end": END
    })

    app = graph.compile()

    # Start the conversation
    print("\nStarting RAG Chatbot (type 'exit' to quit)...")
    state = rag_processor.initialize_state()
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break
            
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        
        # Process through the graph
        state = app.invoke(state)

if __name__ == "__main__":
    main()