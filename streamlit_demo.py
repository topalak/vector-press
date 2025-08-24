import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_press.llm_embedding_initializer import LLMManager
from src.vector_press.db.supabase_db import SupabaseVectorStore
from src.vector_press.rag_processor import RAGProcessor


def initialize_components():
    """Initialize all components with caching"""
    if 'components_initialized' not in st.session_state:
        with st.spinner("Initializing components..."):
            # Initialize LLM Manager
            llm_manager = LLMManager()
            
            # Initialize Supabase Vector Store
            supabase_vector_store = SupabaseVectorStore(llm_manager)
            
            # Initialize RAG Processor
            rag_processor = RAGProcessor(llm_manager.get_llm(), supabase_vector_store)
            
            st.session_state.rag_processor = rag_processor
            st.session_state.components_initialized = True


def main():
    st.set_page_config(page_title="Vector Press Demo", page_icon="🔍", layout="wide")
    
    st.title("🔍 Vector Press Demo")
    st.markdown("Ask questions about the content in your vector database!")
    
    # Initialize components
    initialize_components()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "chunks" in message:
                with st.expander("View retrieved chunks"):
                    for i, chunk in enumerate(message["chunks"]):
                        st.text_area(f"Chunk {i+1}", chunk, height=100, disabled=True)
    
    # Chat input
    if user_query := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Convert chat history to the format expected by RAGProcessor
                conversation_messages = []
                for msg in st.session_state.messages[:-1]:  # Exclude the current user message
                    if msg["role"] == "user":
                        from langchain_core.messages import HumanMessage
                        conversation_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        from langchain_core.messages import AIMessage
                        conversation_messages.append(AIMessage(content=msg["content"]))
                
                # Process query with RAG
                response, retrieved_chunks = st.session_state.rag_processor.process_query(
                    user_query, conversation_messages
                )
                
                st.markdown(response)
                
                # Show retrieved chunks
                if retrieved_chunks:
                    with st.expander("View retrieved chunks"):
                        for i, chunk in enumerate(retrieved_chunks):
                            st.text_area(f"Chunk {i+1}", chunk, height=100, disabled=True)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "chunks": retrieved_chunks
        })
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This is a simple demo of the Vector Press RAG system.
        
        **How it works:**
        1. Enter your question in the chat
        2. The system searches for relevant content
        3. Generates a response based on retrieved information
        
        **Features:**
        - 💬 Chat interface
        - 🔍 Vector similarity search
        - 📄 View retrieved chunks
        - 💾 Conversation memory
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()