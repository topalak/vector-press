#!/usr/bin/env python3
"""
Test script to verify embedding model is working
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.vector_press.llm_embedding_initializer import LLMManager

def test_embedding():
    """Test the embedding model"""
    print("🧪 Testing Embedding Model...")
    print("=" * 50)
    
    try:
        # Initialize LLM Manager
        llm_manager = LLMManager()
        embedding_model = llm_manager.get_embedding_model()
        
        if not embedding_model:
            print("❌ No embedding model available!")
            return False
            
        # Test texts
        test_texts = [
            "artificial intelligence",
            "machine learning algorithms", 
            "natural language processing",
            "hello world",
            "python programming"
        ]
        
        print(f"\n🔤 Testing {len(test_texts)} sample texts:")
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Text: '{text}'")
            
            # Generate embedding
            embedding = embedding_model.embed_query(text)
            
            # Check embedding
            print(f"   ✅ Embedding generated!")
            print(f"   📊 Dimensions: {len(embedding)}")
            print(f"   🔢 First 5 values: {embedding[:5]}")
            print(f"   📈 Range: {min(embedding):.4f} to {max(embedding):.4f}")
            
        # Test similarity
        print(f"\n🔍 Testing similarity:")
        emb1 = embedding_model.embed_query("artificial intelligence")
        emb2 = embedding_model.embed_query("machine learning")  
        emb3 = embedding_model.embed_query("hello world")
        
        # Simple cosine similarity
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
        sim1 = cosine_similarity(emb1, emb2)  # Related terms
        sim2 = cosine_similarity(emb1, emb3)  # Unrelated terms
        
        print(f"   📊 AI vs ML similarity: {sim1:.4f} (should be high)")
        print(f"   📊 AI vs Hello similarity: {sim2:.4f} (should be low)")
        
        print(f"\n✅ Embedding model is working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing embedding: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")