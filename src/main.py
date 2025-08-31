from langgraph.graph import StateGraph, END, START
import sys
import os
from langchain_core.messages import SystemMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_press.llm_embedding_initializer import LLMManager
from src.vector_press.db.supabase_db import SupabaseVectorStore
from src.vector_press.rag_processor import RAGProcessor, SYSTEM_MESSAGE  #from vectorpress import
from src.vector_press.graph_nodes import GraphNodes, AgentState

# TODO, check for supabase slow queries and test the accuracy of calculating similarity function
# TODO check streamlit is working
# TODO remove einops if we aren't using that function, you'll see when you invoke the streamlit.py
# TODO removes bbbertan's load_ollama_from main while publishing it is in import part already and uv remove ollama
# TODO add spending time calculator every functions, if _extracting_article_text wastes too much time we need to seperate it from search_articles because if fetched article already exist we shouldn't waste extra time to extracting it

