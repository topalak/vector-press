from supabase import create_client, Client
import sys
import os
from typing import List, Dict
from datetime import datetime

# Add src to path for config access
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import settings

# Use clean package imports
from ..llm_embedding_initializer import LLMManager
from . import GuardianAPIClient

class SupabaseVectorStore:
    """Handles Supabase database operations for vector storage and retrieval"""
    
    def __init__(self, llm_manager):
        """Initialize Supabase client, embedding model and Guardian API client"""
        self.SUPABASE_URL = settings.SUPABASE_URL
        self.SUPABASE_KEY = settings.SUPABASE_SERVICE_KEY
        self.supabase: Client = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
        self.embedding_model = llm_manager.get_embedding_model()
        self.guardian_client = GuardianAPIClient(self)
        
        print(f"✅ [DEBUG] Supabase Vector Store initialized")

    def _insert_guardian_article_metadata(self, metadata: Dict) -> bool:
        """
        Insert article metadata into guardian_articles table
        
        Args:
            metadata: Article metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
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

    def check_article_exists(self, article_id: str) -> bool:
        """
        Check if an article already exists in the database
        
        Args:
            article_id: Guardian article ID
            
        Returns:
            True if article exists, False otherwise
        """
        try:
            import time
            start_time = time.time()
            result = self.supabase.table('guardian_articles').select('article_id').eq('article_id', 
                                                                                    article_id).execute()
            print(f"⏱️ [DEBUG] check_article_exists took {time.time() - start_time:.4f} seconds")
            return len(result.data) > 0
        except Exception as e:
            print(f"🔥 [DEBUG] Error checking article existence: {e}")
            return False

    def retrieve_relevant_chunks(self, query: str, match_count: int = 10, section_filter: str = None, similarity_threshold: float = 0.7) -> list[dict]:
        """
        Retrieve relevant chunks from Supabase using semantic search
        
        Args:
            query: Search query
            match_count: Number of chunks to retrieve
            section_filter: Optional section filter
            similarity_threshold: Minimum similarity score to include chunk (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing chunk content and metadata above the similarity threshold
            Each dict has: {'content': str, 'title': str, 'section': str, 'publication_date': str, 'similarity': float}
        """

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
                # Filter chunks by similarity threshold and include metadata
                filtered_chunks = []
                for item in result.data[:3]:
                    if item['similarity'] >= similarity_threshold:
                        filtered_chunks.append({
                            'content': item['content'],
                            'title': item['title'],
                            'section': item['section'],
                            'publication_date': item['publication_date'],
                            'similarity': item['similarity']
                        })
                
                # Track retrieved articles - increment search_metadata counter
                for item in result.data:
                    if item['similarity'] >= similarity_threshold:
                        try:
                            # Increment the counter in search_metadata
                            self.supabase.rpc('increment_search_count', {'target_article_id': item['article_id']}).execute()
                        except Exception as e:
                            print(f"⚠️ [DEBUG] Failed to update search counter for article {item['article_id']}: {e}")
                
                print(f"🔍 [DEBUG] Retrieved {len(result.data)} total chunks, {len(filtered_chunks)} above threshold {similarity_threshold}")
                
                # Print similarity scores for all chunks
                for i, item in enumerate(result.data):
                    above_threshold = "✅" if item['similarity'] >= similarity_threshold else "❌"
                    print(f"🔍 [DEBUG] Chunk {i+1} similarity: {item['similarity']:.4f} {above_threshold}")
                
                return filtered_chunks
            else:
                print(f"⚠️ [DEBUG] No relevant chunks found for query")
                return []
                
        except Exception as e:
            print(f"🔥 [DEBUG] Error retrieving chunks: {e}")
            return []

    def _process_extracted_article(self, extracted_data: Dict) -> bool:
        """
        Process already extracted article: chunk, embed, and store

        Args:
            extracted_data: Dictionary with 'metadata' and 'content' from extract_article_text()

        Returns:
            True if successful, False otherwise
        """
        print(f"\n📰 [DEBUG] Processing extracted article...")

        try:
            metadata = extracted_data['metadata']
            #metadata is _insert_guardian_article_metadata's input it's coming from extracted_data which it is in search_articles method
            content = extracted_data['content']
            #same as metadata it is for splitting and they will convert chunks to set ready to embedding and inserting to the article chunks

            if not content:
                print(f"❌ [DEBUG] No content to process")
                return False

            # Insert article metadata
            if not self._insert_guardian_article_metadata(metadata):
                print(f"❌ [DEBUG] Failed to insert article metadata")
                return False

            # Split content into chunks
            chunk_size = 1750
            chunk_overlap = 275
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size - chunk_overlap)]
            print(f"🔧 [DEBUG] Split content into {len(chunks)} chunks")

            if not chunks:
                print(f"⚠️ [DEBUG] No chunks created from content")
                return True

            # Create embeddings for chunks
            print(f"🔄 [DEBUG] Creating embeddings for {len(chunks)} chunks...")
            embedded_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.embedding_model.embed_query(chunk)
                    embedded_chunks.append({
                        'content': chunk,
                        'embedding': embedding
                    })
                    
                    if (i + 1) % 10 == 0:
                        print(f"🔄 [DEBUG] Processed {i + 1}/{len(chunks)} chunks")
                        
                except Exception as e:
                    print(f"🔥 [DEBUG] Error creating embedding for chunk {i}: {e}")
                    continue
            
            print(f"✅ [DEBUG] Created {len(embedded_chunks)} embeddings")
            
            if not embedded_chunks:
                print(f"❌ [DEBUG] Failed to create embeddings")
                return False

            # Insert chunks into database
            if not self._insert_article_chunks(metadata['article_id'], embedded_chunks):
                print(f"❌ [DEBUG] Failed to insert article chunks")
                return False

            print(f"✅ [DEBUG] Successfully processed article {metadata['article_id']}")
            return True

        except Exception as e:
            print(f"🔥 [DEBUG] Error processing extracted article: {e}")
            return False

    def database_uploading(self,
                           query: str = None,
                           section: str = None,
                           from_date: str = None,
                           page_size: int = 200,
                           order_by: str = None,
                           max_pages: int = 20) -> Dict:
        """
        Fetch articles from Guardian API and process them

        Args:
            query: Search query
            section: Guardian section (e.g. technology)
            from_date: Date filter (YYYY-MM-DD format)
            page_size: Number of articles per request
            order_by: Sort order for articles (e.g. relevance, newest, oldest)
            max_pages: Upper limit for page number

        Returns:
            Processing statistics
        """
        print(f"\n🚀 [DEBUG] Starting article fetch and processing...")
        print(f"🚀 [DEBUG] Query: {query}")
        print(f"🚀 [DEBUG] Section: {section}")
        print(f"🚀 [DEBUG] From date: {from_date}")
        print(f"🚀 [DEBUG] Page size: {page_size}")
        print(f"🚀 [DEBUG] Order by: {order_by}")

        stats = {
            'total_fetched': 0,
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': datetime.now(),
            'end_time': None
        }

        try:
            # Fetch articles from Guardian API
            extracted_articles = self.guardian_client.search_articles(
                query=query,
                section=section,
                from_date=from_date,
                page_size=page_size,
                order_by=order_by,
                max_pages=max_pages,
            )

            if not extracted_articles:
                print(f"❌ [DEBUG] Failed to fetch articles from API")
                return stats

            stats['total_fetched'] = len(extracted_articles)

            print(f"📡 [DEBUG] Fetched {len(extracted_articles)} articles from API")

            # Process each extracted article (already extracted in guardian_api.py)
            for i, extracted_article in enumerate(extracted_articles):
                print(f"\n📰 [DEBUG] Processing article {i + 1}/{len(extracted_articles)}")
                try:
                    stats['total_processed'] += 1

                    if self._process_extracted_article(extracted_article):
                        stats['successful'] += 1
                        print(f"✅ [DEBUG] Article {i + 1} processed successfully")
                    else:
                        stats['failed'] += 1
                        print(f"❌ [DEBUG] Article {i + 1} processing failed")

                except Exception as e:
                    stats['failed'] += 1
                    print(f"🔥 [DEBUG] Error processing article {i + 1}: {e}")

            stats['end_time'] = datetime.now()
            duration = stats['end_time'] - stats['start_time']

            print(f"\n📊 [DEBUG] Processing completed!")
            print(f"📊 [DEBUG] Total fetched: {stats['total_fetched']}")
            print(f"📊 [DEBUG] Total processed: {stats['total_processed']}")
            print(f"📊 [DEBUG] Successful: {stats['successful']}")
            print(f"📊 [DEBUG] Failed: {stats['failed']}")
            print(f"📊 [DEBUG] Skipped: {stats['skipped']}")
            print(f"📊 [DEBUG] Duration: {duration.total_seconds():.2f} seconds")

            # Print database statistics
            print(f"📊 [DEBUG] Processing completed successfully")

            return stats

        except Exception as e:
            print(f"🔥 [DEBUG] Error in fetch and process: {e}")
            stats['end_time'] = datetime.now()
            return stats


def main():
    """Main execution flow"""
    print("Vector-Press Guardian Database Population")
    print("=" * 50)
    # Initialize components
    llm_manager = LLMManager()
    supabase_store = SupabaseVectorStore(llm_manager)

    try:
        print("🚀 Populating database with Guardian articles...")

        # Fetch and process articles
        stats = supabase_store.database_uploading(query="artificial intelligence",
                                                  page_size=200,
                                                  order_by="relevance",
                                                  max_pages=10)

        if stats:
            print(f"\n✅ Database population completed successfully!")
            print(f"📊 Processing stats: {stats}")
        else:
            print("\n❌ Database population failed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

