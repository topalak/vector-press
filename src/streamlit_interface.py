import streamlit as st
import sys
import os

# Add src to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.vector_press import LLMManager, RAGProcessor, SupabaseVectorStore, AgentState

from config import settings

@st.cache_resource
def initialize_components():
    """Initialize components with caching"""
    # Initialize all components
    llm_manager = LLMManager()
    supabase_vector_store = SupabaseVectorStore(llm_manager)

    # Create initial state
    initial_state: AgentState = {
        "messages": [],
        "query": ""
    }

    # Initialize RAG Processor with state
    rag_processor = RAGProcessor(llm_manager, supabase_vector_store, initial_state)

    return rag_processor, initial_state


def main():
    os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')
    st.set_page_config(page_title="Vector Press RAG", page_icon="🔍", layout="wide")

    st.title("🔍 Vector Press RAG Chatbot")
    st.markdown("Ask questions about Guardian articles!")

    # Initialize components
    rag_processor, initial_state = initialize_components()

    # Initialize session state
    if "state" not in st.session_state:
        st.session_state.state = initial_state.copy()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask about Guardian articles..."):
        # Add user message to chat display
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process with RAG
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Big Brother is thinking..."):
                # Update state with user query
                st.session_state.state["query"] = user_input

                # Process through RAG
                result_state = rag_processor.process_query(st.session_state.state)

                # Get AI response from the result
                ai_response = result_state["messages"][-1].content

                # Update session state
                st.session_state.state = result_state

                # Display response with agent name
                st.markdown(f"**Big Brother:** {ai_response}")

        # Add assistant response to chat display with agent name
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": f"**Big Brother:** {ai_response}"})

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Vector Press RAG System**

        Ask questions about Guardian articles and get AI-powered responses based on:
        - 🔍 Semantic search through article database
        - 🧠 AI-generated responses using retrieved context
        - 💬 Conversation memory
        """)

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.session_state.state = initial_state.copy()
            st.rerun()

        # Show some stats
        if st.session_state.state.get("messages"):
            st.header("📊 Stats")
            total_messages = len(st.session_state.state["messages"])
            st.metric("Total Messages", total_messages)


if __name__ == "__main__":
    main()