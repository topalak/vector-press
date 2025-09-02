#!/usr/bin/env python3
"""
Vector Press - RAG Chatbot Application
Simple entry point to run the application
"""

from vector_press.rag_processor import main

# TODO, check for supabase slow queries and test the accuracy of calculating similarity function
# TODO removes bbbertan's load_ollama_from main while publishing it is in import part already and uv remove ollama
# TODO check api response when you are in debug mode

if __name__ == "__main__":
    main()

