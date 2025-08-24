from typing import List, Dict, Optional
import re
from datetime import datetime
import sys
import os

# TODO
#add primary key to first table

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from vector_press.db.guardian_api import GuardianAPIClient
from supabase_db import SupabaseVectorStore



class ArticleProcessor:
    """Processes Guardian articles for vector storage"""
    
    def __init__(self, llm_manager, supabase_vector_store: SupabaseVectorStore):
        """
        Initialize processor with required components
        
        Args:
            llm_manager: LLM manager instance
            supabase_vector_store: Supabase vector store instance
        """
        self.guardian_client = GuardianAPIClient()
        self.supabase_store = supabase_vector_store
        self.embedding_model = llm_manager.get_embedding_model()
        
        print(f"🔧 [DEBUG] Article Processor initialized")
        
    
    def create_embeddings(self, chunks: List[str]) -> List[Dict]:
        """
        Create embeddings for text chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of dictionaries with content and embeddings
        """
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
        return embedded_chunks
    
    def process_article(self, article_data: Dict) -> bool:
        """
        Process a single article: extract, chunk, embed, and store
        
        Args:
            article_data: Article data from Guardian API
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n📰 [DEBUG] Processing article...")
        
        try:
            # Extract article content and metadata
            extracted = self.guardian_client.extract_article_text(article_data)
            
            if not extracted:
                print(f"❌ [DEBUG] Failed to extract article content")
                return False
            
            metadata = extracted['metadata']
            content = extracted['content']
            
            if not content:
                print(f"❌ [DEBUG] No content to process")
                return False
            
            # Insert article metadata
            if not self.supabase_store.insert_article_metadata(metadata):
                print(f"❌ [DEBUG] Failed to insert article metadata")
                return False
            
            # Split content into chunks
            chunk_size = 1000
            chunk_overlap = 200
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size-chunk_overlap)]
            print(f"🔧 [DEBUG] Split content into {len(chunks)} chunks")
            
            if not chunks:
                print(f"⚠️ [DEBUG] No chunks created from content")
                return True
            
            # Create embeddings for chunks
            embedded_chunks = self.create_embeddings(chunks)
            
            if not embedded_chunks:
                print(f"❌ [DEBUG] Failed to create embeddings")
                return False
            
            # Insert chunks into database
            if not self.supabase_store.insert_article_chunks(metadata['id'], embedded_chunks):
                print(f"❌ [DEBUG] Failed to insert article chunks")
                return False
            
            print(f"✅ [DEBUG] Successfully processed article {metadata['id']}")
            return True
            
        except Exception as e:
            print(f"🔥 [DEBUG] Error processing article: {e}")
            return False
    
    def database_uploading(self,
                                   query: str = None,
                                   section: str = "technology",
                                   from_date: str = None,
                                   page_size: int = 50,
                                   max_articles: int = None,
                                   order_by: str = None) -> Dict:
        """
        Fetch articles from Guardian API and process them
        
        Args:
            query: Search query
            section: Guardian section (e.g. technology)
            from_date: Date filter (YYYY-MM-DD format)
            page_size: Number of articles per request
            max_articles: Maximum number of articles to process
            order_by: Sort order for articles (e.g. relevance, newest, oldest)
            
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
            api_response = self.guardian_client.search_articles(
                query=query,
                section=section,
                from_date=from_date,
                page_size=page_size,
                order_by=order_by
            )
            
            if not api_response or 'response' not in api_response:
                print(f"❌ [DEBUG] Failed to fetch articles from API")
                return stats
            
            articles = api_response['response'].get('results', [])
            stats['total_fetched'] = len(articles)
            
            print(f"📡 [DEBUG] Fetched {len(articles)} articles from API")
            
            if max_articles:
                articles = articles[:max_articles]
                print(f"🔧 [DEBUG] Limited to {len(articles)} articles")
            
            # Process each article
            for i, article in enumerate(articles):
                print(f"\n📰 [DEBUG] Processing article {i+1}/{len(articles)}")
                
                try:
                    # Check if article already exists
                    article_id = article.get('id', '')
                    if self.supabase_store.check_article_exists(article_id):
                        print(f"⚠️ [DEBUG] Article {article_id} already exists, skipping...")
                        stats['skipped'] += 1
                        continue
                    
                    stats['total_processed'] += 1
                    
                    if self.process_article(article):
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
            
            # Print database statistics
            print(f"📊 [DEBUG] Processing completed successfully")
            
            return stats
            
        except Exception as e:
            print(f"🔥 [DEBUG] Error in fetch and process: {e}")
            stats['end_time'] = datetime.now()
            return stats


def main():
    """Example usage of the article processor"""
    from llm_embedding_initializer import LLMManager
    
    # Initialize components
    print("Initializing components...")
    llm_manager = LLMManager()
    supabase_store = SupabaseVectorStore(llm_manager)
    processor = ArticleProcessor(llm_manager, supabase_store)
    
    # Process some articles
    stats = processor.database_uploading(
        query="artificial intelligence",
        section="technology",
        page_size=10,
        max_articles=5
    )
    
    print(f"\nProcessing completed with stats: {stats}")


if __name__ == "__main__":
    main()