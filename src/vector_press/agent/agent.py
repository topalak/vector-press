from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from src.vector_press.agent.news_api_client import GuardianAPIClient
from src.vector_press.agent.web_search_client import TavilyWebSearchClient
from src.vector_press.agent.rss_client import TechnologyRSSClient, SportsRSSClient

from src.vector_press.agent.tools_validation import TavilySearch, GuardianSearchRequest, TechnologyRSSFeed, SportsRSSFeed

from src.vector_press.model_config import ModelConfig
from config import settings

import logging
import time

from src.vector_press.agent.state import AgentState

logger = logging.getLogger(__name__)

pruning_llm_config = ModelConfig(model="qwen3:0.6b", model_provider_url=settings.OLLAMA_HOST)
embedding_model_config = ModelConfig(model='all-minilm:33m', model_provider_url=settings.OLLAMA_HOST)

INSTRUCTIONS = """You are a smart and helpful research assistant. Your name is Big Brother.

<task>
Your job is using tools to perform user's commands and find related information to answer user's questions.
You can use any of the tools provided to you.
You can call these tools in series or in parallel. Your functionality is conducted in a tool-calling loop.
</task>

<available_tools>
You have access to 4 specialized tools. Choose carefully based on the user's intent:

1. **TavilySearch** - General Web Search
   Use when:
   - User asks for tutorials, guides, or how-to information (e.g., "how to learn Python")
   - User wants historical information (e.g., "history of Bitcoin", "what is quantum computing")
   - User asks about concepts, definitions, or explanations (e.g., "explain blockchain")
   - User wants financial market data or analysis (set topic='finance')
   - User asks for general knowledge not requiring current news

   Do NOT use when:
   - User explicitly asks for NEWS or current events
   - User wants very recent/breaking news (use RSS feeds instead)

   Think first: Is this a general information query or a how-to question? If yes, use this tool.

2. **GuardianSearchRequest** - General News Archive
   Use when:
   - User asks for news about world events, politics, or general current affairs
   - User wants business news, economics, or corporate stories
   - User asks for culture, lifestyle, or opinion pieces
   - User wants ARCHIVED news articles (Guardian has extensive archives)

   Do NOT use when:
   - User wants TECHNOLOGY news (try TechnologyRSSFeed first)
   - User wants SPORTS news (try SportsRSSFeed first)
   - User wants non-news information (use TavilySearch instead)

   Think first: Is this a general news query (politics, world, business, culture)? If yes, use this tool.

3. **TechnologyRSSFeed** - Current Technology News
   Use when:
   - User asks about recent tech news (e.g., "latest AI developments", "new iPhone release")
   - User wants current events in: AI, cybersecurity, startups, tech products, semiconductors

   Do NOT use when:
   - User wants historical tech information (use TavilySearch)
   - User wants tech tutorials or guides (use TavilySearch)

   Think first: Does the user want CURRENT TECHNOLOGY NEWS? If yes, use this tool.

4. **SportsRSSFeed** - Current Sports News
   Use when:
   - User asks about recent sports news (e.g., "latest football scores", "NBA results")
   - User wants current events in: football, basketball, tennis, cricket, olympics, motorsports

   Do NOT use when:
   - User wants historical sports info or statistics (use TavilySearch)
   - User wants sports guides or rules (use TavilySearch)

   Think first: Does the user want CURRENT SPORTS NEWS? If yes, use this tool.

FALLBACK STRATEGY:
If RSS feeds return no results or insufficient information:
   Step 1: Try GuardianSearchRequest for news-related queries
   Step 2: If Guardian also fails, try TavilySearch as final fallback
</available_tools>

<decision_process>
Before calling any tool, think through these questions:

1. What TYPE of information does the user want?
   - Current news? → RSS feeds 
   - Outdated or historical news? → GuardianSearchRequest
   - Tutorials/guides, general information? → TavilySearch

2. What DOMAIN is the query about?
   - Technology news? → TechnologyRSSFeed
   - Sports news? → SportsRSSFeed
   - General news (politics, world, business)? → GuardianSearchRequest
   - Everything else? → TavilySearch

3. How RECENT must the information be?
   - Last 24-48 hours? → RSS feeds (TechnologyRSSFeed/SportsRSSFeed)
   - Last week to months? → GuardianSearchRequest or TavilySearch
   - Historical/timeless? → TavilySearch
</decision_process>

<response_quality>
- Each response should ONLY use context that directly relates to the user's CURRENT question
- Never mix information from previous unrelated queries unless the user explicitly requests it
- If tool results are insufficient, acknowledge limitations rather than hallucinating
- Synthesize information from multiple sources when relevant
</response_quality>
"""

tool_pruning_prompt = """You are an expert at extracting relevant information from documents.

Your task: Analyze the provided document and extract ONLY the information that directly answers or supports the user's specific request. Remove all irrelevant content.

User's Request: {user_request}

Instructions for pruning:
1. Keep information that directly addresses the user's question
2. Preserve key facts, data, and examples that support the answer
3. Remove tangential discussions, unrelated topics, and excessive background
4. Maintain the logical flow and context of relevant information
5. If multiple subtopics are discussed, focus only on those relevant to the request
6. Preserve important quotes, statistics, and research findings when relevant

Return the pruned content in a clear, concise format that maintains readability while focusing solely on what's needed to answer the user's request."""

#TODO add response format for make it more reliable, because I have changed the model from llama3.2:3b to qwen3:8b output format is changed totally
# response_format: This adds a node before END. This will call and LLM.with_structured_output and the output will be formatted to
# match the given schema and returned in the 'structured_response' state key.

class VectorPressAgent:
    """Handles Agent's processing and response generation"""

    def __init__(self, llm):
        """Initialize agent with model and build graph."""
        self.pruning_llm = pruning_llm_config.get_llm()
        self.embedding_model = embedding_model_config.get_embedding()
        self.llm = llm

        self.tavily_search_client = TavilyWebSearchClient()
        self.guardian_client = GuardianAPIClient()
        self.rss_client = TechnologyRSSClient(self.embedding_model)  # we are injecting the embedding model here
        self.sports_client = SportsRSSClient(self.embedding_model)

        tools = [TavilySearch, GuardianSearchRequest, TechnologyRSSFeed, SportsRSSFeed]
        self.structured_llm = self.llm.bind_tools(tools=tools)

        self.state: AgentState = AgentState(  #check is there tools_call in default llm
            context_window=[SystemMessage(content=INSTRUCTIONS)],
            query="",
        )
        self.app = self._build_graph()

    def _llm_call(self, state: AgentState) -> AgentState:
        """LLM call that handles both initial user input and continuation after tools"""

        start_time = time.time()
        response = self.structured_llm.invoke(state.context_window)  #state AIMessage
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"LLM response generation took {elapsed_time:.2f} seconds")

        state.context_window.append(response)
        return state

    def _tools_call(self, state: AgentState) -> AgentState:
        """Execute tool calls and add results as ToolMessages"""

        raw_tool_result = ""  # TODO ask it to BBB, im getting error if i remove that variable

        for tool_call in state.context_window[-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})

            match tool_name:
                case "GuardianSearchRequest":
                    try:
                        raw_tool_result = self._search_guardian_articles(GuardianSearchRequest(**args))
                    except Exception as e:
                        logger.warning(f"Guardian API error: {e}")
                        #TODO
                        #try:
                            #bbc search
                        #except Exception as e:
                            #logger.error(f"Both Guardian and BBC failed: {e}")
                            #state.context_window.append(ToolMessage(
                           # content="News sources unavailable",
                           # name=tool_name,
                            #tool_call_id=tool_call["id"]
                            #))
                            #continue
                    #TODO you need to search error types to understand what kind of errors for tool calls. I need to update except block with those specific errors
                case "TavilySearch":
                    try:
                        raw_tool_result = self._tavily_web_search(TavilySearch(**args))
                    except Exception as e:
                        logger.warning(f"Tavily API error: {e}")
                        #TODO same try except block like above, use it linkup as alternative
                        continue
                case "SportsRSSFeed":
                    try:
                        raw_tool_result = self._sports_rss(SportsRSSFeed(**args))
                    except Exception as e:
                        logger.warning(f"Sports API error: {e}")
                case "TechnologyRSSFeed":
                    try:
                        raw_tool_result = self._technology_rss(TechnologyRSSFeed(**args))
                    except Exception as e:
                        logger.warning(f"Technology API error: {e}")
                case _:
                    logger.warning(f"Unknown tool requested: {tool_name}")
                    raw_tool_result = f"Unknown tool: {tool_name}"

            if raw_tool_result:
                raw_tool_result = '\n'.join(raw_tool_result)
            print('ossuruk')


            if len(raw_tool_result) > 0:
                #TODO ask BBB how to monitor what pruning_llm takes, I want to both approaches behaviour
                start_time = time.time()
                pruned_tool_result = self.pruning_llm.invoke([
                    {"role": "system", "content": tool_pruning_prompt.format(user_request=state.query), },
                    {"role": "user", "content": raw_tool_result},
                ])
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f"Tool pruning took {elapsed_time:.2f} seconds")

                state.context_window.append(ToolMessage(
                    content=pruned_tool_result.content,
                    name=tool_name,
                    tool_call_id=tool_call["id"]
                ))

            #else:

        return state

    def _tavily_web_search(self, validation: TavilySearch) -> list[str]:
        """Web Search Tool"""
        return self.tavily_search_client.search(validation)

    def _search_guardian_articles(self, validation: GuardianSearchRequest) -> list[str]:
        """News Retrieve Tool"""
        return self.guardian_client.search(validation)

    def _technology_rss(self, validation: TechnologyRSSFeed) -> list[str]:
        """Technology RSS Feed"""
        return self.rss_client.search(validation)

    def _sports_rss(self, validation: SportsRSSFeed) -> list[str]:
        """Sports RSS Feed"""
        return self.sports_client.search(validation=validation)


    def _build_graph(self):
        """Build and return the LangGraph pipeline (internal method)."""
        graph = StateGraph(AgentState)

        graph.add_node('llm_call', self._llm_call)
        graph.add_node('tools_call', self._tools_call)

        graph.add_edge(START, 'llm_call')
        graph.add_conditional_edges(
            source='llm_call',
            path=_should_continue,
            path_map={'continue': 'tools_call', 'end': END}
        )
        graph.add_edge('tools_call', 'llm_call')

        return graph.compile()

    def ask(self, query: str) -> str:
        """Ask the agent a question and maintain conversation state."""

        while True:
            if query.lower() == "exit":
                print("\nGoodbye!")
                break

            self.state.query = query

            # Invoke graph (returns dict) and convert back to AgentState
            result = self.app.invoke(self.state)
            self.state = AgentState(**result)

            if self.state.context_window:
                last_message = self.state.context_window[-1]
                print(last_message.content)

            query = input("\nYou: ").strip()
            self.state.context_window.append(HumanMessage(content=query))

        return "No response generated"

def _should_continue(state: AgentState):
    """Determine whether to continue with tool calls or end"""
    last_message = state.context_window[-1]
    if last_message.tool_calls:
        return 'continue'
    else:
        return 'end'

def main():

    # Configure logging to show in terminal
    logging.basicConfig(
        level=logging.INFO,  # Show INFO, WARNING, ERROR
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = ModelConfig(model="qwen3:1.7b", model_provider_url=settings.OLLAMA_HOST, reasoning=False)
    llm = config.get_llm()
    agent = VectorPressAgent(llm)

    agent.ask(query="I want to macbook m3 pro, what do you suggest? is it worth to buy it")
    #can you multiple 15 and 764 by calling tools?
    #Who is Cristiano Ronaldo?
    #Can you fetch 200 articles about Ukraine and Russia war?
    #I want to buy Imac mini m4, what do you think? should I buy it?
    #Can you fetch latest news about Ukraine and Russia war?
    #JPMorgan backs ‘America First’ push with up to $10bn investment
    #I want you to fetch latest news about new Mac Mini m4, I want to buy a new one


    '''
    1- Cristiano Ronaldo
    2- mac mini m4 launching 
    3- NBA results
    
    '''

if __name__ == '__main__':
    main()
