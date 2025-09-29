from supabase import create_client, Client
from typing import List, Dict
from datetime import datetime
import time
import torch

from config import settings

from vector_press.llm_embedding_initializer import LLMManager
from vector_press.agent.news_api_client import GuardianAPIClient

#TODO explore the pytest

def _calculate_optimal_batch_size():
    """
    Calculate optimal batch size based on available GPU VRAM

    Formula:
    - Pure embedding: 768 × 4 = 3,072 bytes (3.072 KB) per chunk
    - Text tokenization memory: ~0.5-1 KB per chunk
    - GPU computation buffers: ~0.5-1 KB per chunk
    - Total: ~4.832 KB per chunk

    Dynamic 75% VRAM formula:
    - available_memory = total_memory * 0.75 (75% of VRAM)
    - optimal_batch_size = int(available_memory / 4.832)

    Returns:
        Optimal batch size for current GPU configuration
    """
    if not torch.cuda.is_available():
        print("⚠️ [VRAM] No CUDA available, using default batch size: 1,000")
        return 1_000

    try:
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory  # bytes
        total_gb = total_memory / (1024**3)

        # Use 75% of VRAM for embeddings (reserve 25% for system/other processes)
        available_memory = total_memory * 0.75

        # Memory per chunk: 4.832 KB = 4,832 bytes
        memory_per_chunk = 4_832

        # Calculate optimal batch size
        optimal_batch_size = int(available_memory / memory_per_chunk)
        #should we save it float instead of int 
        print(f"🔧 [VRAM] GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔧 [VRAM] Total VRAM: {total_gb:.1f} GB")
        print(f"🔧 [VRAM] Optimal batch size: {optimal_batch_size:,} chunks")

        return optimal_batch_size

    except Exception as e:
        print(f"⚠️ [VRAM] Error calculating optimal batch size: {e}")
        print(f"⚠️ [VRAM] Falling back to default: 1,000,000")
        return 1_000_000

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
            batch_size = 500
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
            start_time = time.time()
            result = self.supabase.table('guardian_articles').select('article_id').eq('article_id', 
                                                                                    article_id).execute()
            print(f"⏱️ [DEBUG] check_article_exists took {time.time() - start_time:.4f} seconds")
            return len(result.data) > 0
        except Exception as e:
            print(f"🔥 [DEBUG] Error checking article existence: {e}")
            return False

    def _create_mega_batch_embeddings(self, chunks: List[str]) -> List[Dict]:
        """
        Create embeddings optimized batch processing with EmbeddingGemma formatting

        Args:
            chunks: List of text chunks to embed

        Returns:
            List of dictionaries with content and embeddings
        """

        total_chunks = len(chunks)
        print(f"🚀 [MEGA-BATCH] Processing {total_chunks:,} chunks with EmbeddingGemma A100 optimization")

        # Format chunks with EmbeddingGemma document prompt
        formatted_chunks = [f"title: none | text: {chunk}" for chunk in chunks]
        print(f"🚀 [MEGA-BATCH] Formatted {len(formatted_chunks)} chunks with EmbeddingGemma prompts")

        all_embeddings = []
        total_start_time = time.time()

        # Calculate optimal batch size using dynamic VRAM formula
        optimal_batch_size = _calculate_optimal_batch_size()

        batch_size = min(total_chunks, optimal_batch_size)

        print(f"🚀 [MEGA-BATCH] Selected batch size: {batch_size:,} chunks")
        print(f"🚀 [MEGA-BATCH] Estimated memory usage: {(batch_size * 4_832 / 1024**3):.1f}GB")

        try:
            # Process in mega-batches
            total_batches = 0  # for mute the w
            for i in range(0, total_chunks, batch_size):
                batch = formatted_chunks[i:i + batch_size]  # for ex. [0 : 8.000.000]
                current_batch_size = len(batch)
                batch_num = i // batch_size + 1
                total_batches = (total_chunks - 1) // batch_size + 1

                print(f"🔥 [MEGA-BATCH] Processing batch {batch_num}/{total_batches}")
                print(f"🔥 [MEGA-BATCH] Batch size: {current_batch_size:,} chunks")

                batch_start_time = time.time()

                try:
                    # Clear GPU cache before processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        initial_memory = torch.cuda.memory_allocated() / 1024**3
                        print(f"🔥 [MEGA-BATCH] GPU memory before batch: {initial_memory:.1f}GB")

                    # THE MEGA-BATCH EMBEDDING CALL - Single HTTP request for massive batch
                    batch_embeddings = self.embedding_model.embed_documents(batch)

                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time

                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                        final_memory = torch.cuda.memory_allocated() / 1024**3
                        print(f"🔥 [MEGA-BATCH] Peak GPU memory: {peak_memory:.1f}GB")
                        print(f"🔥 [MEGA-BATCH] Final GPU memory: {final_memory:.1f}GB")
                        torch.cuda.reset_peak_memory_stats()

                    print(f"✅ [MEGA-BATCH] Batch {batch_num} completed in {batch_duration:.1f}s")
                    print(f"✅ [MEGA-BATCH] Throughput: {current_batch_size/batch_duration:.0f} chunks/second")
                    print(f"✅ [MEGA-BATCH] HTTP requests: 1 (for {current_batch_size:,} chunks)")

                    # Combine with original chunk content (without EmbeddingGemma formatting)
                    original_chunks = chunks[i:i + batch_size]
                    for chunk, embedding in zip(original_chunks, batch_embeddings):
                        all_embeddings.append({
                            'content': chunk,  # Store original content without prompt formatting
                            'embedding': embedding
                        })

                except Exception as e:
                    print(f"Could not process batch: {e}")
                    raise e

        except Exception as e:
            print(f"❌ [MEGA-BATCH] Critical error: {e}")
            raise e

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Final statistics
        print(f"\n🎉 [MEGA-BATCH] PROCESSING COMPLETE!")
        print(f"🎉 [MEGA-BATCH] Total chunks: {len(all_embeddings):,}")
        print(f"🎉 [MEGA-BATCH] Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"🎉 [MEGA-BATCH] Average throughput: {len(all_embeddings)/total_duration:.0f} chunks/second")
        if total_batches != 0:
            print(f"🎉 [MEGA-BATCH] HTTP requests saved: ~{len(all_embeddings)-total_batches:,}")
            print(f"🎉 [MEGA-BATCH] Estimated cost savings: {((len(all_embeddings)-total_batches)/len(all_embeddings)*100):.1f}%")

        return all_embeddings

    def retrieve_relevant_chunks(self, query: str, match_count: int = 10, section_filter: str = None, similarity_threshold: float = 0.6) -> list[dict]:
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
            # Generate embedding for the query with EmbeddingGemma format
            formatted_query = f"task: search result | query: {query}"
            query_embedding = self.embedding_model.embed_query(formatted_query)
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
                for item in result.data:
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
        Process extracted article: chunk, embed, and store

        Args:
            extracted_data: Dictionary with 'metadata' and 'content' from extract_article_text()

        Returns:
            True if successful, False otherwise
        """
        try:
            metadata = extracted_data['metadata']
            content = extracted_data['content']

            print(f"\n📰 [DEBUG] Processing article: {metadata.get('article_id', 'unknown')}")

            if not content:
                print(f"❌ [DEBUG] No content to process")
                return False

            # Insert article metadata first
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
            print(f"🚀 [DEBUG] Creating embeddings for {len(chunks)} chunks...")
            embedded_chunks = self._create_mega_batch_embeddings(chunks)

            if not embedded_chunks:
                print(f"❌ [DEBUG] Failed to create embeddings")
                return False

            print(f"✅ [DEBUG] Created {len(embedded_chunks)} embeddings")

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

            # Process each article individually
            print(f"\n🚀 [DEBUG] Processing {len(extracted_articles)} articles...")

            for i, extracted_article in enumerate(extracted_articles):
                try:
                    stats['total_processed'] += 1
                    print(f"\n📰 [DEBUG] Processing article {i+1}/{len(extracted_articles)}")

                    if self._process_extracted_article(extracted_article):
                        stats['successful'] += 1
                        print(f"✅ [DEBUG] Article {i+1} processed successfully")
                    else:
                        stats['failed'] += 1
                        print(f"❌ [DEBUG] Article {i+1} processing failed")

                except Exception as e:
                    stats['failed'] += 1
                    print(f"🔥 [DEBUG] Error processing article {i+1}: {e}")

            stats['end_time'] = datetime.now()
            duration = stats['end_time'] - stats['start_time']

            print(f"\n📊 [DEBUG] Processing completed!")
            print(f"📊 [DEBUG] Total fetched: {stats['total_fetched']}")
            print(f"📊 [DEBUG] Total processed: {stats['total_processed']}")
            print(f"📊 [DEBUG] Successful: {stats['successful']}")
            print(f"📊 [DEBUG] Failed: {stats['failed']}")
            print(f"📊 [DEBUG] Skipped: {stats['skipped']}")
            print(f"📊 [DEBUG] Duration: {duration.total_seconds():.2f} seconds")

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

        stats = supabase_store.database_uploading(query="artificial intelligence",
                                                  page_size=200,
                                                  order_by="relevance",
                                                  max_pages=5)


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

