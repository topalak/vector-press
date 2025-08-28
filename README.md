# Vector-Press

![Vector Press Graph](vector_press_graph.png)

## 📁 Project Structure

```
vector-press/
├── src/
│   ├── config.py                    # Configuration management
│   ├── main.py                      # Main application entry point
│   └── vector_press/
│       ├── __init__.py
│       ├── llm_embedding_initializer.py  # LLM & embedding management
│       ├── rag_processor.py              # RAG orchestration
│       ├── graph_nodes.py               # LangGraph conversation nodes
│       └── db/
│           ├── __init__.py
│           ├── supabase_db.py           # Vector database operations
│           ├── guardian_api.py          # The Guardian API client
│           ├── inserting_supabase.py    # Article processing pipeline
│           └── database_uploading.py    # Database population script
├── test/                            # Test files
├── out/                            # Build artifacts (ignored)
├── .env                           # Environment variables (ignored)
├── .gitignore                     # Git ignore rules
├── pyproject.toml                 # Project dependencies
├── CLAUDE.md                      # Development guidelines
└── README.md                      # This file
```

**Latest News RAG System** - An intelligent news retrieval and chat system powered by The Guardian API, vector embeddings, and conversational AI.

## 🚀 Overview

Vector-Press is a modular RAG system that fetches news articles from The Guardian API, stores them in a vector database, and provides conversational AI responses based on the content.

### How It Works

1. **Database Check**: Searches vector database for relevant content
2. **API Fallback**: Fetches new articles from The Guardian API if needed
3. **Smart Response**: Returns contextual answers based on retrieved articles
4. **Database Update**: New articles are automatically embedded and stored

## 📦 Installation

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) installed and running
- Supabase account with pgvector extension
- The Guardian API key

### Setup

1. **Clone and install**
   ```bash
   git clone <repository-url>
   cd vector-press
   uv sync
   ```

2. **Setup Supabase database**
   - Create a [Supabase](https://supabase.com/) account and new project
   - Copy and execute the SQL from `src/vector_press/db/supabase_setup.sql` in your Supabase SQL editor

3. **Configure environment**
   ```env
   OLLAMA_HOST=http://localhost:11434
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   GUARDIAN_API_KEY=your_guardian_api_key
   ```

## 🚀 Usage

### Start Chat Interface
```bash
uv run python src/main.py
```

### Populate Database
```bash
uv run python src/vector_press/db/database_uploading.py
```

## 🏗️ Architecture

- **GraphNodes**: LangGraph-based conversation flow
- **RAGProcessor**: Orchestrates retrieval and response generation  
- **LLMManager**: Handles Ollama model initialization
- **SupabaseVectorStore**: Vector database operations
- **GuardianAPIClient**: The Guardian API integration
- **ArticleProcessor**: Article processing pipeline

## 🛠️ Additional Commands

```bash
# Sync dependencies
uv sync

# Run main application
uv run python src/main.py

# Run tests
uv run python test/test_embedding.py
```

Follow modular design principles outlined in `CLAUDE.md`.