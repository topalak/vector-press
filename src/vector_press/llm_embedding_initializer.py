from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from ai_common.llm import load_ollama_model
import requests
import json

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


class LLMManager:
    """Manages LLM and embedding initialization with fallback logic"""

    # Embedding model option
    use_lightweight_embedding = False  # Set to True for low-resource systems

    def __init__(self):
        self.llm = None
        self.embedding_model = None
        
        print(f"🔧 [DEBUG] LLM Manager initialized")
        self._initialize_llm()
        self._initialize_embeddings()

    def _initialize_llm(self):
        """Initialize LLM with fallback logic"""
        try:
            # Load Ollama model to memory first
            load_ollama_model(model_name='qwen3:8b', ollama_url=settings.OLLAMA_HOST)
            
            self.llm = ChatOllama(
                model='qwen3:8b',
                base_url=settings.OLLAMA_HOST,
                temperature=0,
                num_ctx=8192
            )
            self.llm.invoke([HumanMessage(content="test")])
            print(f"✅ [DEBUG] Using Ollama (remote) with model: {self.llm.model}, context: {self.llm.num_ctx}, base_url: {self.llm.base_url}")
        except Exception as e:
            print(f"Failed to initialize Ollama: {e}")

    ###########################   EMBEDDING SUCKS WASTES TOO MUCH TIME   ##################################################
    def _initialize_embeddings(self):
        """Initialize embedding model with resource-aware selection"""

        try:
            # Pull the embedding model via HTTP API to remote Ollama server
            print(f"🔄 [DEBUG] Pulling nomic-embed-text:v1.5 to remote server...")
            response = requests.post(
                f"{settings.OLLAMA_HOST}/api/pull",
                #https://meet.google.com/vbp-rccg-tyj?authuser=0
                #api_endpoint?authuser=0
                json={"name": "nomic-embed-text:v1.5"},
                timeout=300
            )
            if response.status_code == 200:
                print(f"✅ [DEBUG] Successfully pulled nomic-embed-text:v1.5")
            else:
                print(f"⚠️ [DEBUG] Pull response: {response.text}")
            
            # Create a simple embedding wrapper that uses direct Ollama API
            class DirectOllamaEmbedding:
                def __init__(self, base_url, model):
                    self.base_url = base_url.rstrip('/')
                    self.model = model
                    
                def embed_query(self, text):
                    response = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    if response.status_code == 200:
                        return response.json()["embedding"]
                    else:
                        raise Exception(f"Embedding failed: {response.text}")
            
            self.embedding_model = DirectOllamaEmbedding(settings.OLLAMA_HOST, "nomic-embed-text:v1.5")
            print(f"✅ [DEBUG] Using direct Ollama embedding: nomic-embed-text:v1.5 at {settings.OLLAMA_HOST}")
        except Exception as e:
            print(f"⚠️ [DEBUG] Failed to initialize embedding: {e}")
            self.embedding_model = None

    def get_llm(self):
        """Get the initialized LLM"""
        return self.llm

    def get_embedding_model(self):
        """Get the initialized embedding model"""
        return self.embedding_model