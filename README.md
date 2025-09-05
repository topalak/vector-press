# Vector-Press 🚀

**Latest News RAG System** - An intelligent news retrieval and chat system powered by The Guardian API, vector embeddings, and conversational AI.

## 📁 Project Structure

```
vector-press/
├── src/
│   ├── config.py                         # Environment configuration management
│   ├── main.py                          # Terminal application entry point
│   ├── streamlit_interface.py           # Web UI interface (Streamlit)
│   └── vector_press/
│       ├── __init__.py                  # Package initialization
│       ├── llm_embedding_initializer.py # Ollama LLM & embedding setup
│       ├── rag_processor.py             # RAG orchestration & LangGraph flow
│       └── db/
│           ├── __init__.py              # Database package init
│           ├── supabase_db.py           # Vector database operations & search
│           ├── guardian_api.py          # The Guardian API client & extraction
│           └── supabase_setup.sql       # Database schema & functions
├── test/                                # Development test files (gitignored)
├── out/                                 # Build artifacts (gitignored)
├── .env                                 # Environment variables (gitignored)
├── .gitignore                          # Git ignore configuration
├── pyproject.toml                      # Project dependencies (uv)
├── CLAUDE.md                           # Modular development guidelines
└── README.md                           # Project documentation
```

**Latest News RAG System** - An intelligent news retrieval and chat system powered by The Guardian API and conversational AI.

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
   git clone <https://github.com/topalak/vector-press.git>
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

### Terminal Chat Interface
```bash
uv run python src/main.py
```

### Web UI (Streamlit)
```bash
uv run streamlit run src/streamlit_interface.py
```

### Populate Database with Guardian Articles
```bash
uv run python src/vector_press/db/supabase_db.py
```

## 🏗️ Architecture

### Core Components
- **RAGProcessor**: LangGraph-based conversation flow and retrieval orchestration
- **LLMManager**: Ollama model initialization and management (Qwen + Nomic embeddings)
- **SupabaseVectorStore**: Vector database operations, similarity search, and analytics
- **GuardianAPIClient**: The Guardian API integration and article extraction
- **Dual Interfaces**: Terminal CLI and Streamlit web UI

### Key Features
- **Smart Retrieval**: Database-first approach with API fallback
- **Vector Search**: Cosine similarity search with configurable thresholds
- **Source Citations**: Automatic article attribution with publication dates
- **Search Analytics**: Tracks article retrieval frequency
- **Modular Design**: Clean separation 
