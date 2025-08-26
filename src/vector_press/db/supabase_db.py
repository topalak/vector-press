from supabase import create_client, Client
from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import settings


class SupabaseVectorStore:
    """Handles Supabase database operations for vector storage and retrieval"""
    
    def __init__(self, llm_manager):
        """Initialize Supabase client and embedding model"""
        self.SUPABASE_URL = settings.SUPABASE_URL
        self.SUPABASE_KEY = settings.SUPABASE_SERVICE_KEY
        self.supabase: Client = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
        self.embedding_model = llm_manager.get_embedding_model()
        
        print(f"✅ [DEBUG] Supabase Vector Store initialized")

    def retrieve_relevant_chunks(self, query: str, match_count: int = 10, section_filter: str = None) -> List[str]:
        """
        Retrieve relevant chunks from Supabase using semantic search
        
        Args:
            query: Search query
            match_count: Number of chunks to retrieve
            section_filter: Optional section filter
            
        Returns:
            List of relevant chunk contents
        """
        print(f"\n🔍 [DEBUG] Retrieving chunks for query: '{query[:100]}...'")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.embed_query(query)
            print(f"🔍 [DEBUG] Generated query embedding with {len(query_embedding)} dimensions")
            
            # Call the match_article_chunks function
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }
            
            if section_filter:
                params['section_filter'] = section_filter
            
            result = self.supabase.rpc('match_article_chunks', params).execute()
            
            if result.data:
                chunks = [item['content'] for item in result.data]
                print(f"✅ [DEBUG] Retrieved {len(chunks)} relevant chunks")
                
                # Print similarity scores for debugging
                for i, item in enumerate(result.data[:3]):
                    print(f"🔍 [DEBUG] Chunk {i+1} similarity: {item['similarity']:.4f}")
                
                return chunks
            else:
                print(f"⚠️ [DEBUG] No relevant chunks found for query")
                return []
                
        except Exception as e:
            print(f"🔥 [DEBUG] Error retrieving chunks: {e}")
            return []

    def _check_article_exists(self, article_id: str) -> bool:
        """
        Check if an article already exists in the database
        
        Args:
            article_id: Guardian article ID
            
        Returns:
            True if article exists, False otherwise
        """
        try:
            result = self.supabase.table('guardian_articles').select('article_id').eq('article_id', article_id).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"🔥 [DEBUG] Error checking article existence: {e}")
            return False

    def _insert_guardian_article_metadata(self, metadata: Dict) -> bool:
        """
        Insert article metadata into guardian_articles table
        
        Args:
            metadata: Article metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if article already exists
            if self._check_article_exists(metadata['article_id']):
                print(f"⚠️ [DEBUG] Article {metadata['article_id']} already exists, skipping...")
                return True
            
            result = self.supabase.table('guardian_articles').insert(metadata).execute()
            
            if result.data:
                print(f"✅ [DEBUG] Inserted article metadata: {metadata['article_id']}")
                return True
            else:
                print(f"❌ [DEBUG] Failed to insert article metadata")
                return False
                
        except Exception as e:
            print(f"🔥 [DEBUG] Error inserting article metadata: {e}")
            return False

    def _insert_article_chunks(self, article_id: str, chunks: List[Dict]) -> bool:
        """
        Insert article chunks with embeddings into article_chunks table
        
        Args:
            article_id: Guardian article ID
            chunks: List of chunk dictionaries with content and embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare chunks for insertion
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_data.append({
                    'article_id': article_id,
                    'chunk_number': i,
                    'content': chunk['content'],
                    'embedding': chunk['embedding']
                })
            
            # Insert chunks in batches
            batch_size = 100
            for i in range(0, len(chunk_data), batch_size):
                batch = chunk_data[i:i + batch_size]
                result = self.supabase.table('article_chunks').insert(batch).execute()
                
                if not result.data:
                    print(f"❌ [DEBUG] Failed to insert chunk batch {i//batch_size + 1}")
                    return False
            
            print(f"✅ [DEBUG] Inserted {len(chunks)} chunks for article {article_id}")
            return True
            
        except Exception as e:
            print(f"🔥 [DEBUG] Error inserting article chunks: {e}")
            return False


    def _delete_article(self, article_id: str) -> bool:
        """
        Delete an article and all its chunks
        
        Args:
            article_id: Guardian article ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete chunks first (due to foreign key constraint)
            self.supabase.table('article_chunks').delete().eq('article_id', article_id).execute()
            
            # Delete article metadata
            result = self.supabase.table('guardian_articles').delete().eq('id', article_id).execute()
            
            print(f"✅ [DEBUG] Deleted article {article_id} and its chunks")
            return True
            
        except Exception as e:
            print(f"🔥 [DEBUG] Error deleting article: {e}")
            return False