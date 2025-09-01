#!/usr/bin/env python3
"""
Vector Press - RAG Chatbot Application
Simple entry point to run the application
"""

from src.vector_press.rag_processor import main

# TODO, check for supabase slow queries and test the accuracy of calculating similarity function
# TODO check streamlit is working
# TODO remove einops if we aren't using that function, you'll see when you invoke the streamlit.py
# TODO removes bbbertan's load_ollama_from main while publishing it is in import part already and uv remove ollama
# TODO add spending time calculator every functions, if _extracting_article_text wastes too much time we need to seperate it from search_articles because if fetched article already exist we shouldn't waste extra time to extracting it

if __name__ == "__main__":
    main()

