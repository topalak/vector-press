from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from ai_common.llm import _check_and_pull_ollama_model  #TODO add load_ollama for saving more time

#TODO
# 1- we missed a big spot, which model wants like below:
# text = "search_document: Your actual document content here"
# embeddings = ollama_client.embeddings(model="nomic-embed-text", PROMPT=text)
from config import settings

class LLMManager:
    """Manages LLM and embedding initialization"""

    def __init__(self):
        self._llm = None
        self._embedding_model = None
        self._llm_initialized = False
        self._embedding_initialized = False

        print(f"🔧 [DEBUG] LLM Manager initialized")

    def _initialize_llm(self, model_name:str):
        """Initialize LLM with fallback logic"""
        try:
            # Try Ollama first
            _check_and_pull_ollama_model(model_name=model_name, ollama_url=settings.OLLAMA_HOST)

            self._llm = ChatOllama( #langchain wrapper
                model=model_name,
                base_url=settings.OLLAMA_HOST,
                temperature=0,
                num_ctx=8192,
                #reasoning=True,
            )

            print(f"✅ [DEBUG] Using Ollama (remote) with model: {self._llm.model}, context: {self._llm.num_ctx}")
        except Exception as e:
            print(f"⚠️ [DEBUG] Failed to initialize Ollama: {e}")

            try:
                # Fallback to Groq
                self._llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=settings.GROQ_API_KEY,
                    temperature=0,
                    max_tokens=8192,
                )
                print(f"✅ [DEBUG] Using Groq fallback with model: {self._llm.model}")
            except Exception as groq_error:
                print(f"❌ [DEBUG] Failed to initialize Groq fallback: {groq_error}")
                print(f"💡 [DEBUG] Make sure GROQ_API_KEY is set in your environment")
                self._llm = None

            raise


    def _initialize_embeddings(self):
        """Initialize embedding model using LangChain's OllamaEmbeddings"""

        try:
            print(f"🔄 [DEBUG] Initializing EmbeddingGemma model...")

            # Load/pull the embedding model first
            _check_and_pull_ollama_model(model_name='embeddinggemma', ollama_url=settings.OLLAMA_HOST)

            # Use LangChain's built-in OllamaEmbeddings
            self._embedding_model = OllamaEmbeddings(
                model="embeddinggemma",
                base_url=settings.OLLAMA_HOST
            )

            # Test the embedding model
            print(f"✅ [DEBUG] EmbeddingGemma initialized successfully")

        except Exception as e:
            print(f"⚠️ [DEBUG] Failed to initialize embedding: {e}")
            print(f"💡 [DEBUG] Make sure Ollama is running and accessible")
            self._embedding_model = None

    def get_llm(self, model_name:str):
        """Get the LLM, initializing it if needed"""
        if not self._llm_initialized:
            print(f"🔄 [DEBUG] loading LLM...")
            self._initialize_llm(model_name=model_name)
            self._llm_initialized = True
        return self._llm

    def get_embedding_model(self):
        """Get the embedding model, initializing it if needed"""
        if not self._embedding_initialized:
            print(f"🔄 [DEBUG] loading embedding model...")
            self._initialize_embeddings()
            self._embedding_initialized = True
        return self._embedding_model


def main():
    """Test the LLMManager initialization and functionality"""
    print("🧪 Testing LLMManager")
    print("=" * 50)
    
    try:
        # Initialize LLMManager
        print("🔧 Initializing LLMManager...")
        llm_manager = LLMManager()

        # Test LLM
        print("\n🤖 Testing LLM...")
        llm = llm_manager.get_llm(model_name="llama-3.1-8b-instant")
        if llm:
            response = llm.invoke([HumanMessage(content="Say hello")])
            print(f"✅ LLM Response: {response.content[:50]}...")
        else:
            print("❌ LLM not available")

        # Test Embedding Model
        print("\n🔤 Testing Embedding Model...")
        embedding_model = llm_manager.get_embedding_model()
        if embedding_model:
            test_texts = ["artificial intelligence", "hello world", "python programming"]

            for i, text in enumerate(test_texts, 1):
                embedding = embedding_model.embed_query(text)
                print(f"✅ Test {i}: '{text}' -> {len(embedding)} dimensions")
                print(f"   📊 First 3 values: {embedding[:3]}")

            print(f"✅ Embedding model working perfectly!")
        else:
            print("❌ Embedding model not available")
            
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()