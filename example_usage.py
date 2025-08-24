#!/usr/bin/env python3
"""
Example usage of the Vector-Press Guardian system

This script demonstrates how to:
1. Fetch articles from Guardian API
2. Process and embed them
3. Store in Supabase
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.vector_press.llm_embedding_initializer import LLMManager
from src.vector_press.db.supabase_db import SupabaseVectorStore
from src.vector_press.db.inserting_supabase import ArticleProcessor


def database_uploading_initialize():
    """Fetch and process articles from Guardian API"""
    print("🚀 Populating database with Guardian articles...")
    
    # Initialize components
    llm_manager = LLMManager()
    supabase_store = SupabaseVectorStore(llm_manager)
    processor = ArticleProcessor(llm_manager, supabase_store)
    
    # Fetch and process articles
    stats = processor.database_uploading(
        query="artificial intelligence",
        section="technology",
        page_size=20,
        max_articles=10,
        order_by="relevance"
    )
    
    print(f"✅ Database population completed!")
    return stats




def main():
    """Main execution flow"""
    print("Vector-Press Guardian Database Population Demo")
    print("=" * 50)
    
    try:
        # Populate database with Guardian articles
        stats = database_uploading_initialize()
        
        print("\n✅ Database population completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()