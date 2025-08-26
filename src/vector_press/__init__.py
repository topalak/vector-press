# Import core classes
from .llm_embedding_initializer import LLMManager
from .rag_processor import RAGProcessor
from .graph_nodes import GraphNodes, AgentState

# Import key methods directly from classes
from .llm_embedding_initializer import LLMManager as _LLMManager
from .rag_processor import RAGProcessor as _RAGProcessor
from .graph_nodes import GraphNodes as _GraphNodes

# Make key methods available at module level
get_llm = _LLMManager.get_llm
get_embedding_model = _LLMManager.get_embedding_model
process_query = _RAGProcessor.process_query
_user_input_node = _GraphNodes._user_input_node
#_retrieve_chunks_node = _GraphNodes._retrieve_chunks_node
_generate_response_node = _GraphNodes._generate_response_node
_should_exit = _GraphNodes._should_exit

__all__ = [
    # Core classes
    'LLMManager',
    'RAGProcessor',
    'GraphNodes',
    'AgentState',
    
    # Main user functions
    'get_llm',           # Users get LLM instance
    'get_embedding_model', # Users get embedding model
    'process_query',     # Users process queries
    
    # Internal graph workflow functions
    '_user_input_node',
    #'_retrieve_chunks_node',
    '_generate_response_node',
    '_should_exit',
]