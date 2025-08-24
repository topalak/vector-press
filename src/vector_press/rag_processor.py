from langchain_core.messages import SystemMessage, AIMessage
from src.vector_press.db.supabase_db import SupabaseVectorStore


class RAGProcessor:
    """Handles RAG query processing and response generation"""

    def __init__(self, llm, supabase_vector_store: SupabaseVectorStore):
        """Initialize with LLM and Supabase vector store"""
        self.llm = llm  # From LLMManager
        self.supabase_vector_store = supabase_vector_store  # SupabaseVectorStore instance

    #add input's types
    def process_query(self, user_query : str, conversation_messages : list[str]):
        """Process user query with RAG"""
        # Retrieve relevant chunks
        retrieved_chunks = self.supabase_vector_store.retrieve_relevant_chunks(user_query)

        # Handle cases with no relevant chunks (e.g., greetings)
        if not retrieved_chunks: # TODO (change the meridian capital parts)
            system_message = SystemMessage(content="""You are a helpful AI assistant for AAAAAAAAAAAAAAAAAAAAAAAAAAA 
            The user's message doesn't seem to be asking about specific company information. 
            Respond naturally and helpfully, and suggest they can ask about Meridian Capital's services, 
            investment philosophy, team, or performance.""")
        else:
            # Create context from retrieved chunks
            context_text = "\n\n".join(retrieved_chunks)
            system_message = SystemMessage(content=f"""Based on the following context about Meridian Capital:

            {context_text}

      Please answer the user's questions based on the provided context.""")

        # Keep original conversation flow with system context
        last_prompt = conversation_messages + [system_message]

        response = self.llm.invoke(last_prompt)
        print(f'\nBig Brother Bertan: {response.content}')

        print(f'Current memory: {conversation_messages + [AIMessage(content=response.content)]}')

        return response.content, retrieved_chunks