from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from typing import TypedDict, Annotated
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
# TODO look the video why is he using should continue

class AgentState(TypedDict):
    """State class for LangGraph conversation flow"""
    messages: Annotated[list[BaseMessage], add_messages]  # keeps every type of message with BaseMessage
    retrieved_chunks: list[str]
    query: str


class RAGProcessor:
    """Handles RAG query processing and response generation"""

    def __init__(self, llm_manager: LLMManager, supabase_vector_store: SupabaseVectorStore, state: AgentState):
        """Initialize with LLM manager, Supabase vector store, and add INSTRUCTIONS to state"""
        self.llm = llm_manager.get_llm()  # Get LLM from manager
        self.supabase_vector_store = supabase_vector_store  # SupabaseVectorStore instance

        state['messages'].append(SystemMessage(content=INSTRUCTIONS))

    def process_query(self, state: AgentState) -> AgentState:
        """Process user query with RAG and return updated state"""
        user_input = state.get('query', '')
        
        retrieved_chunks = self.supabase_vector_store.retrieve_relevant_chunks(
            user_input, 
            match_count=10, 
            similarity_threshold=0.5
        )
        # TODO we are not adding retrieved_chunks to the state's retrieved_chunks we can remove it because actually we dont need it because we are adding it to the user's_input
        # TODO we are retrieving 10 chunks and decreasing it to the 3 this means more computation
        context_text = "\n\n".join(retrieved_chunks) if retrieved_chunks else ""
        # ✅ Now filtering chunks with similarity threshold of 0.5
        enhanced_user_input = f"{user_input}\n\nContext:\n{context_text}" if context_text else user_input
        
        state['messages'].append(HumanMessage(content=enhanced_user_input))

        response = self.llm.invoke(state['messages'])
        print(f'\nBig Brother Bertan: {response.content}')

        # Return updated state
        return {
            **state,
            'messages': [
                *state['messages'],
                AIMessage(content=response.content)
            ],
            'retrieved_chunks': retrieved_chunks,
            'query': user_input  # Use original user input
        }


def main():
    from langgraph.graph import StateGraph, START, END
    
    # Start the conversation
    print("\nStarting RAG Chatbot (type 'exit' to quit)...")
    state: AgentState = {
        "messages": [],
        "retrieved_chunks": [],
        "query": ""
    }
    
    llm_manager = LLMManager()
    supabase_vector_store = SupabaseVectorStore(llm_manager)
    rag_processor = RAGProcessor(llm_manager, supabase_vector_store, state)

    # Build simplified graph with single node
    graph = StateGraph(AgentState)
    graph.add_node('process_query', rag_processor.process_query)

    # Simplified connections - no looping, just single execution
    graph.add_edge(START, 'process_query')
    graph.add_edge('process_query', END)

    app = graph.compile()
    
    while True:
        user_input = input("\nYou: ").strip()
        #why are we using strip?
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break
        
        # Store user input in query field for process_query to access
        state["query"] = user_input

        # Process through the graph
        state = app.invoke(state)

if __name__ == "__main__":
    main()