from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from .db.supabase_db import SupabaseVectorStore
from .llm_embedding_initializer import LLMManager

INSTRUCTIONS = """You are a helpful AI assistant for The Guardian's articles.

IMPORTANT: Always check if the user's current query is related to the provided context chunks. 

If there are related chunks below that are RELEVANT to the user's current question, you will answer based on these chunks and cite your sources at the end:

Sources:
- "Article Title" - Section, YYYY-MM-DD
- "Another Article Title" - Section, YYYY-MM-DD

If the provided chunks are NOT relevant to the user's current question OR if there are no chunks provided, you MUST:
- Ignore any irrelevant context from previous conversations
- Politely inform the user: "We don't have related articles about your query in our database for now"
- Kindly invite them to ask about news topics like technology, sports, politics, business, science, world events, etc.
- Do NOT include any source citations or references
- Do NOT use outdated context that doesn't match their current query

CRITICAL: Each response should ONLY use context that directly relates to the user's CURRENT question. Never mix information from previous unrelated queries."""


class AgentState(TypedDict):
    """State class for LangGraph conversation flow"""
    messages: Annotated[list[BaseMessage], add_messages]  # keeps every type of message with BaseMessage
    query: str


class RAGProcessor:
    """Handles RAG query processing and response generation"""

    def __init__(self, llm_manager: LLMManager, supabase_vector_store: SupabaseVectorStore, state: AgentState):
        """Initialize with LLM manager, Supabase vector store, and add INSTRUCTIONS to state"""
        self.llm = llm_manager.get_llm()  # Get LLM from manager
        self.supabase_vector_store = supabase_vector_store  # SupabaseVectorStore instance
        self.last_retrieved_chunks = []
        state['messages'].append(SystemMessage(content=INSTRUCTIONS))

    def process_query(self, state: AgentState) -> AgentState:
        """Process user query with RAG and return updated state"""
        user_input = state.get('query', '')
        
        retrieved_chunks = self.supabase_vector_store.retrieve_relevant_chunks(
            user_input, 
            match_count=10, 
            similarity_threshold=0.7
        )

        # Format chunks with metadata for enhanced context
        if retrieved_chunks:
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks):
                # Format: [Section | Date | "Title"] \n Content
                pub_date = chunk['publication_date'][:10] if chunk['publication_date'] else 'Unknown date'  # YYYY-MM-DD
                header = f"[{chunk['section']} | {pub_date} | \"{chunk['title']}\"]"
                context_parts.append(f"{header}\n{chunk['content']}")
            
            context_text = "\n\n".join(context_parts)
            enhanced_user_input = f"{user_input}\n\nContext:\n{context_text}"
        else:
            enhanced_user_input = user_input
        
        # Store retrieved chunks for terminal display (used in main())
        self.last_retrieved_chunks = retrieved_chunks if retrieved_chunks else []
        
        state['messages'].append(HumanMessage(content=enhanced_user_input))

        response = self.llm.invoke(state['messages'])
        print(f'\nBig Brother Bertan: {response.content}')

        # Return updated state
        return {
            **state,
            'messages': [*state['messages'], AIMessage(content=response.content)],
            'query': user_input  # Use original user input
            # TODO keep state['query']'s track to see how its storing user's queries
        }


def main():
    from langgraph.graph import StateGraph, START, END
    
    # Start the conversation
    print("\nStarting RAG Chatbot (type 'exit' to quit)...")
    state: AgentState = {
        "messages": [],
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
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break
        
        # Store user input in query field for process_query to access
        state["query"] = user_input

        # Process through the graph
        state = app.invoke(state)
        
        # Display retrieved chunks in terminal
        if hasattr(rag_processor, 'last_retrieved_chunks') and rag_processor.last_retrieved_chunks:
            print(f"\n📚 Retrieved {len(rag_processor.last_retrieved_chunks)} relevant chunks:")
            print("=" * 50)
            for i, chunk in enumerate(rag_processor.last_retrieved_chunks, 1):
                pub_date = chunk['publication_date'][:10] if chunk['publication_date'] else 'Unknown date'
                header = f"[{chunk['section']} | {pub_date} | \"{chunk['title']}\"]"
                print(f"\n{i}. {header}")
                print(f"   Similarity: {chunk['similarity']:.3f}")
                print(f"   Content: {chunk['content'][:200]}..." if len(chunk['content']) > 200 else f"   Content: {chunk['content']}")
            print("=" * 50)

if __name__ == "__main__":
    main()