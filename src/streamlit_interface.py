import streamlit as st
import sys
import os

# Add src to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_press import LLMManager
from vector_press.agent import VectorPressAgent, AgentState, should_continue
from langgraph.graph import StateGraph, START, END

from config import settings

@st.cache_resource
def initialize_components():
    """Initialize agent components with caching"""
    # Initialize LLM manager
    llm_manager = LLMManager()

    # Create initial state
    initial_state: AgentState = {
        "messages": [],
        "query": ""
    }

    # Initialize agent
    vectorpress_agent = VectorPressAgent(llm_manager, initial_state)

    # Create and compile LangGraph
    graph = StateGraph(AgentState)
    graph.add_node('llm_call', vectorpress_agent.llm_call)
    graph.add_node('tools_call', vectorpress_agent.tools_call)

    graph.add_edge(start_key=START, end_key='llm_call')
    graph.add_conditional_edges(
        source='llm_call',
        path=should_continue,
        path_map={'continue': 'tools_call', 'end': END}
    )
    graph.add_edge('tools_call', 'llm_call')

    agent_app = graph.compile()

    return agent_app, initial_state


def main():
    os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')
    st.set_page_config(page_title="Vector Press RAG", page_icon="🔍", layout="wide")

    st.title("🔍 Vector Press RAG Chatbot")
    st.markdown("Ask questions about Guardian articles!")

    # Initialize components
    agent_app, initial_state = initialize_components()

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

        # Process with Agent
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Big Brother is thinking..."):
                # Update state with user query
                st.session_state.state["query"] = user_input

                # Process through agent
                result_state = agent_app.invoke(st.session_state.state)

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
        **Vector Press Agent System**

        Ask questions about news and current events and get AI-powered responses using:
        - 🔍 Guardian API for news searches
        - 🌐 Tavily web search for general topics
        - 🧠 AI agent with tool calling capabilities
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