# Vector-Press 🤖

**AI-Powered News Assistant** - An intelligent agent system that leverages LangGraph, vector databases, and multiple APIs to provide contextual news information and web search capabilities.

## 📁 Project Structure

```
vector-press/
├── src/
│   ├── config.py                      # Environment configuration management
│   ├── main.py                        # Application entry point
│   ├── streamlit_interface.py         # Web UI interface (optional)
│   └── vector_press/
│       ├── __init__.py
│       ├── llm_embedding_initializer.py # LLM & embedding model management
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── agent.py               # Main LangGraph agent implementation
│       │   ├── api_clients.py         # Guardian API client & utilities
│       │   └── tools_validation.py    # Pydantic models for tool schemas
│       └── db/
│           ├── __init__.py
│           └── supabase_db.py         # Vector database operations
├── test/                              # Test files
├── out/                               # Output artifacts
├── .env                               # Environment variables (gitignored)
├── pyproject.toml                     # Project dependencies (uv)
├── CLAUDE.md                          # Development guidelines
└── README.md                          # Project documentation
```

## 🚀 Overview

Vector-Press is an intelligent news assistant powered by LangGraph that provides:

- **Multi-tool Agent**: Uses Guardian API for news and Tavily for general web search
- **LangGraph Workflow**: Structured agent flow with tool calling capabilities
- **Vector Database**: Supabase with pgvector for semantic search and storage
- **Intelligent Routing**: Automatically selects appropriate tools based on query context

### Agent Architecture

The system uses a **LangGraph-based agent** named "Big Brother" that:

1. **Analyzes** user queries to determine the best tool(s) to use
2. **Routes** news-related queries to Guardian API and general queries to Tavily
3. **Processes** tool results and provides contextual responses
4. **Maintains** conversation state and history

### Available Tools

1. **search_guardian_articles**: Searches The Guardian API for news articles
   - Supports section filtering, date ranges, pagination
   - Extracts full article content with metadata

2. **tavily_web_search**: General web search for non-news topics
   - Technology guides, finance information, general queries
   - Configurable result limits and topic filtering

## 📦 Installation

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) installed and running
- Supabase account with pgvector extension
- API keys: Guardian API, Tavily API

### Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/efetopalak/vector-press.git
   cd vector-press
   uv sync
   ```

2. **Configure Ollama models**
   ```bash
   # Install required models
   ollama pull llama3.2:3b
   ollama pull embeddinggemma
   ```

3. **Setup environment variables**
   Create `.env` file in project root:
   ```env
   # Ollama Configuration
   OLLAMA_HOST=http://localhost:11434

   # Database Configuration
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key

   # API Keys
   GUARDIAN_API_KEY=your_guardian_api_key
   TAVILY_API_KEY=your_tavily_api_key

   # Optional: LangSmith Tracing
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_TRACING=false

   # Optional: Groq Fallback
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Setup Supabase database** (optional, for vector storage)
   - Create tables for article storage and vector search
   - Install pgvector extension

## 🚀 Usage

### Interactive Terminal Agent
```bash
uv run python src/main.py
```

Example conversation:
```
You: What's the latest news about artificial intelligence?
Big Brother: [Searches Guardian API and provides recent AI news with sources]

You: How does machine learning work?
Big Brother: [Uses Tavily for general ML information]

You: exit
```

### Web Interface (Optional)
```bash
uv run streamlit run src/streamlit_interface.py
```

### Database Population (Optional)
```bash
uv run python src/vector_press/db/supabase_db.py
```

## 🏗️ Architecture Details

### Tool Integration

**Guardian API Tool**:
- Searches news articles with advanced filtering
- Extracts full article content and metadata
- Supports pagination and section filtering

**Tavily Web Search Tool**:
- General web search capabilities
- Topic-specific filtering (general, finance)
- Configurable result limits

### LLM & Embedding Management

- **Primary LLM**: Llama 3.2 3B (via Ollama)
- **Fallback LLM**: Llama 3.1 8B Instant (via Groq)
- **Embeddings**: EmbeddingGemma (via Ollama)
- **GPU Optimization**: Dynamic batch sizing based on VRAM

### Vector Database Features

- **Semantic Search**: Cosine similarity with configurable thresholds
- **Batch Processing**: Optimized embedding generation
- **Metadata Storage**: Article metadata and search analytics
- **Duplicate Detection**: Prevents duplicate article storage

## 🛠️ Development

### Agent Debugging

The agent includes comprehensive logging:
- Tool call tracking
- Message flow debugging
- Performance timing
- Error handling with context

## 🔧 Configuration

### Agent Behavior
Modify `INSTRUCTIONS` in `src/vector_press/agent/agent.py` to customize:
- Agent personality and responses
- Tool selection logic
- Response formatting

### Tool Parameters
Configure tool behavior in `src/vector_press/agent/tools_validation.py`:
- Search result limits
- API timeout settings
- Validation rules

### LLM Settings
Adjust model parameters in `src/vector_press/llm_embedding_initializer.py`:
- Temperature and context size
- Model selection priority
- Fallback configuration

## 📚 Dependencies

### Core Framework
- **LangGraph** >= 0.6.0 - Agent orchestration
- **LangChain-Ollama** - Local LLM integration
- **Pydantic** >= 2.11.7 - Data validation

### APIs & Databases
- **Supabase** >= 2.18.1 - Vector database
- **Tavily** - Web search API
- **Requests** >= 2.32.4 - HTTP client

### AI & ML
- **Torch** >= 2.8.0 - GPU optimization
- **ai-common** - Shared AI utilities

## 📊 Agent Features

### Intelligent Tool Selection
- Automatically detects news vs. general queries
- Supports parallel tool execution
- Maintains conversation context

### Performance Optimization
- GPU memory management
- Batch processing for embeddings
- Efficient vector search

## 📄 License

MIT License - see LICENSE file for details

## 🙋‍♂️ Author

**Efe Topalak** - [topalakefe@gmail.com](mailto:topalakefe@gmail.com)

---

*Built with LangGraph, Ollama, and modern AI/ML practices for intelligent news assistance.*