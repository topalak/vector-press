# Vector-Press 🤖

**AI-Powered News Assistant** - An intelligent agentic system that retrieves news and web information using LangGraph, The Guardian API, Tavily search, and conversational AI.

## 🚀 Overview

Vector-Press is an **agentic news assistant** named **Big Brother** that intelligently orchestrates multiple tools to answer user queries about news and current events. Built with LangGraph and LangChain, it features a tool-calling loop that decides which data sources to use based on the query.

### How It Works

The agent operates in a **reasoning → action → observation** loop:

1. **User Query**: You ask a question (e.g., "What's the latest news about AI?")
2. **LLM Reasoning**: The agent analyzes your query and decides which tool(s) to use
3. **Tool Selection**:
   - **Guardian News API** for news-related queries
   - **Tavily Web Search** for general web searches
4. **Tool Execution**: Fetches relevant information from selected sources
5. **Response Generation**: Synthesizes information into a conversational answer
6. **Context Management**: Maintains conversation history for follow-up questions

### Key Features

- 🧠 **Agentic Architecture**: LangGraph-based autonomous decision-making
- 🔧 **Multi-Tool Orchestration**: Automatic tool selection (Guardian API + Tavily Search)
- ✅ **Pydantic Validation**: Two-step validation (schema guidance + runtime checking)
- 🎯 **Stateful Conversations**: Maintains context across multiple queries
- ⚡ **GPU-Optimized Embeddings**: Dynamic batch sizing based on VRAM availability

## 📁 Project Structure

```
vector-press/
├── src/
│   ├── config.py                                   # Environment configuration
│   ├── main.py                                     # CLI entry point
│   └── vector_press/
│       ├── __init__.py                             # Package initialization
│       ├── llm_embedding_initializer.py            # LLM & embedding model setup
│       ├── agent/
│       │   ├── __init__.py                         # Agent package init
│       │   ├── agent.py                            # Main agent with LangGraph
│       │   ├── state.py                            # AgentState (Pydantic model)
│       │   ├── tools_validation.py                 # Tool schemas (TavilySearch, GuardianSearchRequest)
│       │   ├── web_search_client.py                # Tavily web search client
│       │   └── news_api_client.py                  # Guardian API client
│       └── db/
│           ├── __init__.py                         # Database package init
│           └── supabase_db.py                      # Vector database operations
├── test/                                           # Development test files
├── out/                                            # Build artifacts
├── .env                                            # Environment variables
├── .gitignore                                      # Git ignore configuration
├── pyproject.toml                                  # Project dependencies (uv)
├── uv.lock                                         # Lock file for reproducible builds
└── README.md                                       # Project documentation
```

## 📦 Installation

### Prerequisites

- **Python** >= 3.13
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **[Ollama](https://ollama.com/)** - Local LLM runtime (for Llama, Qwen, etc.)
- **Supabase** - PostgreSQL database with pgvector extension
- **API Keys**:
  - The Guardian API key ([get one here](https://open-platform.theguardian.com/access/))
  - Tavily API key ([get one here](https://tavily.com/))

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/topalak/vector-press.git
   cd vector-press
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```env
   # LLM Configuration
   OLLAMA_HOST=http://localhost:11434
   GROQ_API_KEY=your_groq_api_key_here  # Optional fallback

   # Database
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key

   # API Keys
   GUARDIAN_API_KEY=your_guardian_api_key
   TAVILY_API_KEY=your_tavily_api_key

   # Optional: LangSmith Tracing
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_TRACING=false
   ```

4. **Setup Supabase database**
   - Create a new Supabase project at [supabase.com](https://supabase.com/)
   - Enable the **pgvector** extension in your database
   - Create the required tables and functions (see [Database Schema](#database-schema))

## 🚀 Usage

### Interactive CLI Chat

```bash
uv run python src/main.py
```

**Example conversation:**
```
You: Who is Cristiano Ronaldo?
Agent: [Uses Tavily web search to fetch information]

You: Can you fetch latest news about Ukraine and Russia war?
Agent: [Uses Guardian News API to fetch articles]

You: exit
```

### Run as Python Module

```python
from vector_press.agent.agent import VectorPressAgent

# Initialize agent with your preferred model
agent = VectorPressAgent(model_name='llama3.2:3b')

# Ask questions
agent.ask("What's the latest news about AI?")
```

### Populate Database with Guardian Articles

```bash
uv run python src/vector_press/db/supabase_db.py
```

This will:
- Fetch articles from The Guardian API
- Generate embeddings using GPU-optimized batching
- Store articles in Supabase vector database

### Core Components

#### 1. **VectorPressAgent** (`agent.py`)
- **LangGraph orchestration**: Manages the agent's reasoning loop
- **Tool binding**: Connects tools to the LLM
- **State management**: Maintains conversation context
- **Two-step validation**:
  - Schema guidance to LLM via `bind_tools()`
  - Runtime validation with Pydantic models

#### 2. **Tool Clients**

**GuardianAPIClient** (`news_api_client.py`):
- Fetches news articles from The Guardian
- Validates requests with `GuardianSearchRequest` Pydantic model
- Supports pagination, sections, and date filtering

**TavilyWebSearchClient** (`web_search_client.py`):
- General web search using Tavily API
- Validates requests with `TavilySearch` Pydantic model
- Configurable max results and topic filtering

#### 3. **LLMManager** (`llm_embedding_initializer.py`)
- **Lazy initialization**: Loads models only when needed
- **Fallback logic**: Ollama → Groq if local model unavailable
- **Embedding support**: EmbeddingGemma for vector generation

#### 4. **SupabaseVectorStore** (`supabase_db.py`)
- **Vector operations**: Similarity search with cosine distance
- **GPU-optimized batching**: Dynamic batch sizing based on VRAM
- **Article chunking**: Smart text splitting with overlap
- **Search analytics**: Tracks article retrieval frequency
