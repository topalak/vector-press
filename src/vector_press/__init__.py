# Import core classes
from .llm_embedding_initializer import LLMManager
from .rag_processor import RAGProcessor
from .graph_nodes import GraphNodes, AgentState
from .db.supabase_db import SupabaseVectorStore

# Import key methods directly from classes
from .llm_embedding_initializer import LLMManager as _LLMManager
from .rag_processor import RAGProcessor as _RAGProcessor
from .graph_nodes import GraphNodes as _GraphNodes
from .db.supabase_db import SupabaseVectorStore as _SupabaseVectorStore

# Make key methods available at module level
get_llm = _LLMManager.get_llm
get_embedding_model = _LLMManager.get_embedding_model
process_query = _RAGProcessor.process_query
retrieve_relevant_chunks = _SupabaseVectorStore.retrieve_relevant_chunks
user_input_node = _GraphNodes.user_input_node
generate_response_node = _GraphNodes.generate_response_node
should_exit = _GraphNodes.should_exit

__all__ = [
    # Core classes
    'LLMManager',
    'RAGProcessor',
    'GraphNodes',
    'AgentState',
    'SupabaseVectorStore',
    
    # Main user functions
    'get_llm',           # Users get LLM instance
    'get_embedding_model', # Users get embedding model
    'process_query',     # Users process queries
    'retrieve_relevant_chunks', # Users retrieve relevant chunks
    
    # Graph workflow functions
    'user_input_node',
    'generate_response_node',
    'should_exit',
]