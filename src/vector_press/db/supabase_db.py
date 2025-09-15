from supabase import create_client, Client
import sys
import os
from typing import List, Dict
from datetime import datetime
import time
import torch

# Add src to path for config access
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

# Use clean package imports
from vector_press.llm_embedding_initializer import LLMManager
from vector_press.db.guardian_api import GuardianAPIClient

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
            import time
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
        Create embeddings using A100 optimized mega-batch processing with EmbeddingGemma formatting

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

        # A100 80GB optimized batch size (targeting 50GB usage)
        optimal_batch_size = min(total_chunks, 8_000_000)  # 8M chunks = ~38GB (conservative)

        if total_chunks <= 1000:
            batch_size = total_chunks  # Process small batches entirely
        else:
            batch_size = optimal_batch_size

        print(f"🚀 [MEGA-BATCH] Selected batch size: {batch_size:,} chunks")
        print(f"🚀 [MEGA-BATCH] Estimated memory usage: {(batch_size * 4.832 / 1024**3):.1f}GB")

        try:
            # Process in mega-batches
            for i in range(0, total_chunks, batch_size):
                batch = formatted_chunks[i:i + batch_size]
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

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠️ [MEGA-BATCH] OOM detected! Falling back to smaller batches...")
                        # Fallback to smaller batch processing
                        smaller_batch_size = max(100, batch_size // 10)
                        print(f"⚠️ [MEGA-BATCH] Retrying with batch size: {smaller_batch_size:,}")

                        original_batch = chunks[i:i + current_batch_size]  # Original chunks without formatting

                        for j in range(0, current_batch_size, smaller_batch_size):
                            small_original_chunks = original_batch[j:j + smaller_batch_size]
                            small_formatted_chunks = [f"title: none | text: {chunk}" for chunk in small_original_chunks]

                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                small_embeddings = self.embedding_model.embed_documents(small_formatted_chunks)

                                for chunk, embedding in zip(small_original_chunks, small_embeddings):
                                    all_embeddings.append({
                                        'content': chunk,
                                        'embedding': embedding
                                    })
                                print(f"✅ [FALLBACK] Processed {len(small_original_chunks)} chunks")

                            except Exception as fallback_error:
                                print(f"❌ [FALLBACK] Even smaller batch failed: {fallback_error}")
                                # Final fallback: individual processing
                                for chunk in small_original_chunks:
                                    try:
                                        formatted_chunk = f"title: none | text: {chunk}"
                                        embedding = self.embedding_model.embed_query(formatted_chunk)
                                        all_embeddings.append({
                                            'content': chunk,
                                            'embedding': embedding
                                        })
                                    except:
                                        print(f"❌ [INDIVIDUAL] Failed to embed chunk")
                                        continue
                    else:
                        raise e

        except Exception as e:
            print(f"❌ [MEGA-BATCH] Critical error: {e}")
            return []

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Final statistics
        print(f"\n🎉 [MEGA-BATCH] PROCESSING COMPLETE!")
        print(f"🎉 [MEGA-BATCH] Total chunks: {len(all_embeddings):,}")
        print(f"🎉 [MEGA-BATCH] Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"🎉 [MEGA-BATCH] Average throughput: {len(all_embeddings)/total_duration:.0f} chunks/second")
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

    def _process_extracted_article(self, extracted_data: Dict, pre_computed_embeddings: List = None) -> bool:
        """
        Process already extracted article: chunk, embed (if needed), and store

        Args:
            extracted_data: Dictionary with 'metadata' and 'content' from extract_article_text()
            pre_computed_embeddings: Optional pre-computed embeddings for cross-article mega-batching

        Returns:
            True if successful, False otherwise
        """
        print(f"\n📰 [DEBUG] Processing extracted article...")

        try:
            metadata = extracted_data['metadata']
            #metadata is _insert_guardian_article_metadata's input it's coming from extracted_data which it is in search_articles method
            content = extracted_data['content']
            #same as metadata it is for splitting, and they will convert chunks to set ready to embedding and inserting to the article chunks

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
            if pre_computed_embeddings is not None:
                # Cross-article mega-batch mode - use pre-computed embeddings
                print(f"🚀 [CROSS-BATCH] Using pre-computed embeddings for {len(chunks)} chunks")
                if len(pre_computed_embeddings) != len(chunks):
                    print(f"❌ [CROSS-BATCH] Embedding count mismatch: expected {len(chunks)}, got {len(pre_computed_embeddings)}")
                    return False

                embedded_chunks = [
                    {'content': chunk, 'embedding': emb}
                    for chunk, emb in zip(chunks, pre_computed_embeddings)
                ]
            else:
                # Normal single-article mode - create embeddings
                print(f"🚀 [SINGLE-BATCH] Creating embeddings for {len(chunks)} chunks using mega-batch processing...")
                embedded_chunks = self._create_mega_batch_embeddings(chunks)
            
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

            # CROSS-ARTICLE MEGA-BATCH PROCESSING
            print(f"\n🚀 [CROSS-ARTICLE] Starting cross-article mega-batch processing for {len(extracted_articles)} articles")

            # Phase 1: Collect all chunks from all articles
            print(f"🔄 [CROSS-ARTICLE] Phase 1: Collecting chunks from all articles...")
            all_chunks = []
            article_chunk_mappings = []

            for i, extracted_article in enumerate(extracted_articles):
                content = extracted_article['content']
                if not content:
                    print(f"⚠️ [CROSS-ARTICLE] Article {i+1} has no content, skipping...")
                    stats['total_processed'] += 1
                    stats['failed'] += 1
                    continue

                # Create chunks (same logic as _process_extracted_article)
                chunk_size = 1750
                chunk_overlap = 275
                chunks = [content[j:j + chunk_size] for j in range(0, len(content), chunk_size - chunk_overlap)]

                if chunks:
                    start_idx = len(all_chunks)
                    all_chunks.extend(chunks)

                    article_chunk_mappings.append({
                        'article_data': extracted_article,
                        'chunk_count': len(chunks),
                        'start_idx': start_idx,
                        'article_index': i + 1
                    })
                    print(f"📝 [CROSS-ARTICLE] Article {i+1}: {len(chunks)} chunks (total: {len(all_chunks)})")

                stats['total_processed'] += 1

            print(f"🚀 [CROSS-ARTICLE] Collected {len(all_chunks):,} chunks from {len(article_chunk_mappings)} articles")

            # Phase 2: MEGA-BATCH EMBEDDING - SINGLE HTTP REQUEST!
            if all_chunks:
                print(f"🔥 [CROSS-ARTICLE] Phase 2: Creating embeddings for {len(all_chunks):,} chunks in SINGLE request...")
                all_embeddings = self._create_mega_batch_embeddings(all_chunks)

                if len(all_embeddings) != len(all_chunks):
                    print(f"❌ [CROSS-ARTICLE] Embedding count mismatch: expected {len(all_chunks)}, got {len(all_embeddings)}")
                    stats['failed'] = len(extracted_articles)
                    return stats

                print(f"✅ [CROSS-ARTICLE] Successfully created {len(all_embeddings):,} embeddings!")

                # Phase 3: Distribute results back to articles and save to database
                print(f"🔄 [CROSS-ARTICLE] Phase 3: Saving {len(article_chunk_mappings)} articles to database...")

                for mapping in article_chunk_mappings:
                    try:
                        start_idx = mapping['start_idx']
                        count = mapping['chunk_count']
                        article_embeddings = [emb['embedding'] for emb in all_embeddings[start_idx:start_idx + count]]

                        # Save article with its pre-computed embeddings
                        if self._process_extracted_article(mapping['article_data'], article_embeddings):
                            stats['successful'] += 1
                            print(f"✅ [CROSS-ARTICLE] Article {mapping['article_index']} saved successfully")
                        else:
                            stats['failed'] += 1
                            print(f"❌ [CROSS-ARTICLE] Article {mapping['article_index']} save failed")

                    except Exception as e:
                        stats['failed'] += 1
                        print(f"🔥 [CROSS-ARTICLE] Error processing article {mapping['article_index']}: {e}")

            else:
                print(f"⚠️ [CROSS-ARTICLE] No chunks collected, nothing to process")
                stats['failed'] = len(extracted_articles)

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

        # Fetch and process articles with cross-article mega-batch
        stats = supabase_store.database_uploading(query="artificial intelligence",
                                                  page_size=200,
                                                  order_by="relevance",
                                                  max_pages=5)  # Medium test: 1000 articles


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

