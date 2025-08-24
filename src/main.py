from langgraph.graph import StateGraph, END, START
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_press.llm_embedding_initializer import LLMManager
from src.vector_press.db.supabase_db import SupabaseVectorStore
from src.vector_press.rag_processor import RAGProcessor
from src.vector_press.graph_nodes import GraphNodes, AgentState


def main():
    """Main entry point - orchestrates all components"""

    # Initialize all components
    print("Initializing LLM Manager...")
    llm_manager = LLMManager()

    print("Initializing Supabase Vector Store...")
    supabase_vector_store = SupabaseVectorStore(llm_manager)

    print("Initializing RAG Processor...")
    rag_processor = RAGProcessor(llm_manager.get_llm(), supabase_vector_store)

    print("Initializing Graph Nodes...")
    graph_nodes = GraphNodes(rag_processor)

    # Build graph
    print("Building LangGraph...")
    graph = StateGraph(AgentState)
    graph.add_node('user_input', graph_nodes.user_input_node)
    graph.add_node('generate_response', graph_nodes.generate_response_node)

    # Connections
    graph.add_edge(START, 'user_input')
    graph.add_conditional_edges('user_input', graph_nodes.should_exit, {"continue": 'generate_response', "end": END})
    graph.add_edge('generate_response', 'user_input')

    app = graph.compile()

    # Start the conversation
    print("\nStarting RAG Chatbot (type 'exit' to quit)...")
    state = {
        "messages": [],
        "retrieved_chunks": [],
        "api_content": {},
        "query": ""
    }
    app.invoke(state)

    # Generate graph visualization
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("simple_rag_modular.png", "wb") as f:
            f.write(png_data)
        print("Graph visualization saved as 'simple_rag_modular.png'")
    except Exception as e:
        print(f"Could not generate graph: {e}")


if __name__ == "__main__":
    main()