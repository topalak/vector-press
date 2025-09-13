from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage
#from ai_common.llm import load_ollama_model
from ollama import Client, ListResponse
from tqdm import tqdm

#TODO we can change the embedding model to version 1.5 but we need to update with embedding every article again.
from config import settings

def check_and_pull_ollama_model(model_name: str, ollama_url: str) -> None:
    """Check if model exists, pull if not."""
    ollama_client = Client(host=ollama_url)
    response: ListResponse = ollama_client.list()
    available_model_names = [x.model for x in response.models]

    if model_name not in available_model_names:
        print(f'Pulling {model_name}')
        current_digest, bars = '', {}
        for progress in ollama_client.pull(model=model_name, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()
            if not digest:
                print(progress.get('status'))
                continue
            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)
            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)
            current_digest = digest


def load_ollama_model(model_name: str, ollama_url: str) -> None:
    """Load model into memory (works for both LLM and embedding models)."""
    check_and_pull_ollama_model(model_name=model_name, ollama_url=ollama_url)
    ollama_client = Client(host=ollama_url)

    # Try embedding first (for embedding models), fallback to generate (for LLMs)
    try:
        ollama_client.embeddings(model=model_name, prompt="test")
    except:
        ollama_client.generate(model=model_name, prompt="test")

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
                num_ctx=8192,
                reasoning=True,
            )
            print(f"✅ [DEBUG] Using Ollama (remote) with model: {self.llm.model}, context: {self.llm.num_ctx}")
        except Exception as e:
            print(f"Failed to initialize Ollama: {e}")

    def _initialize_embeddings(self):
        """Initialize embedding model using LangChain's OllamaEmbeddings"""

        try:
            print(f"🔄 [DEBUG] Initializing Nomic embedding model...")
            
            # Load/pull the embedding model first
            load_ollama_model(model_name='nomic-embed-text', ollama_url=settings.OLLAMA_HOST)
            
            # Use LangChain's built-in OllamaEmbeddings
            self.embedding_model = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=settings.OLLAMA_HOST
            )
            
            # Test the embedding model
            print(f"✅ [DEBUG] Nomic embedding initialized successfully")

        except Exception as e:
            print(f"⚠️ [DEBUG] Failed to initialize embedding: {e}")
            print(f"💡 [DEBUG] Make sure Ollama is running and accessible")
            self.embedding_model = None

    def get_llm(self):
        """Get the initialized LLM"""
        return self.llm

    def get_embedding_model(self):
        """Get the initialized embedding model"""
        return self.embedding_model


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
        llm = llm_manager.get_llm()
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