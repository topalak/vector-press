from langgraph.graph import StateGraph, END, START
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_press.llm_embedding_initializer import LLMManager
from src.vector_press.db.supabase_db import SupabaseVectorStore
from src.vector_press.rag_processor import RAGProcessor
from src.vector_press.graph_nodes import GraphNodes, AgentState

# TODO, check for supabase slow queries and test the accuracy of calculating similarity function
# TODO check streamlit is working
# TODO remove einops if we aren't using that function, you'll see when you invoke the streamlit.py
# TODO removes bbbertan's load_ollama_from main while publishing it is in import part already and uv remove ollama

def main():

    llm_manager = LLMManager()

    supabase_vector_store = SupabaseVectorStore(llm_manager)

    rag_processor = RAGProcessor(llm_manager, supabase_vector_store)

    graph_nodes = GraphNodes(rag_processor)

    # Build graph
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
        "query": ""
    }
    app.invoke(state)

if __name__ == "__main__":
    main()