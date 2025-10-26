from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from config import settings
import logging
from langchain_core.tools import StructuredTool
from functools import partial

# Import shared models first to avoid circular imports
from src.vector_press.agent.models import ToDo

# PlanningAgent imported lazily in Tools.__init__ to avoid circular import
from src.vector_press.model_config import ModelConfig

#Clients
from src.vector_press.agent.news_api_client import GuardianAPIClient,NewYorkTimesAPIClient
from src.vector_press.agent.web_search_client import TavilyWebSearchClient
from src.vector_press.agent.rss_client import TechnologyRSSClient, SportsRSSClient
# write_todos and read_todos imported lazily in Tools.__init__ to avoid circular import
#from vector_press import AgentState


#state is a Pydantic model (AgentState), not a dictionary. Pydantic models don't have a .get() method. we aren't able to pass it as dictionary like --> state['context_window'] we need to pass it like
# state.context_window

################################  TOOL'S SCHEMAS  #########################################

class Query(BaseModel):
    """This is base parameter"""
    query: str = Field(...,min_length=1,max_length=500,
            description="get the most related query as possible as you can")
            #"Extract relevant keywords from user's query for semantic matching. "
            #"Keep it focused on 3-5 keywords for best results."

class TavilySearchSchema(Query):

    max_results: int = Field(default=4,ge=4,le=10,description= #TODO I shouldn't set that optionally, handle that by using runnableconfig
            "Number of search results to return. "
            "Use 2-3 for quick answers, 5-10 for comprehensive research, "
            "10+ for deep exploration.")

    topic: Literal['general', 'finance', 'news'] = Field(
        default='general',description=
            "Search topic type: "
            "'general' - for most queries (tech, science, tutorials, concepts). "
            "'finance' - ONLY when query is about stocks, markets, trading, "
            "financial data, or economic indicators.")

class TheGuardianApiSchema(Query):

    #section: Optional[str] = Field(default=None, description="Guardian section (e.g., 'world', 'politics', 'business', 'technology')")  #section is messing up the results lets comment it
    max_pages: int = Field(default=1,ge=1,le=20,
            description="Number of pages to fetch. "
            "Use 1-2 for quick results, 3-5 for moderate results, "
            "10+ for comprehensive results. "
            "Note: Total articles = page_size × max_pages.")
    page_size: int = Field(default = 3, ge=1,le=50,
            description="Number of articles per page. ")

class NewYorkTimesApiSchema(Query):
    """
    This is New York Times API.
    Use this tool for GENERAL NEWS searches (world, politics, business, culture, etc.).

    When to use:
    - User asks for news about world events, politics, or general current affairs
    - User wants business news, economics, or corporate stories
    - User asks for culture, lifestyle, or opinion pieces
    - User wants ARCHIVED news articles (Guardian has extensive archives)

    Think first: Is this a general news query (politics, world, business, culture)?
    If yes, use this tool.
    """

class TechnologyRSSFeedSchema(Query):
    """
    Use this tool for TECHNOLOGY-RELATED CURRENT NEWS queries only.

    When to use:
    - User asks about recent tech news (e.g., "latest AI developments", "new iPhone release")
    - User wants current events in: AI, cybersecurity, startups, tech products, semiconductors

    Think first: Does the user want CURRENT TECHNOLOGY NEWS? If yes, use this tool.
    """

class SportsRSSFeedSchema(Query):
    """
    Use this tool for SPORTS-RELATED CURRENT NEWS queries only.

    When to use:
    - User asks about recent sports news (e.g., "latest football scores", "NBA results")
    - User wants current events in: football, basketball, tennis, cricket, olympics, motorsports

    Think first: Does the user want CURRENT SPORTS NEWS? If yes, use this tool.
    """





class PlanningAgentSchema(BaseModel):
    """
    **Planning Agent - Intelligent Query Decomposition and Task Orchestration**

    Use this tool when the user's request requires MULTIPLE distinct operations,
    coordination across different data sources, or sequential task execution.

    ## When to Use Planning Agent

    ### ✅ USE when user query contains:

    1. **Multiple Topics in One Query**
       - "Fetch news about AI developments, Ukraine war, and NBA results"
       - "Get me information on Bitcoin, Tesla stock, and climate change"

    2. **Sequential Dependencies**
       - "Find latest AI news, summarize it, then search for related research papers"
       - "Get Ukraine war updates, then fetch historical context"
       - Indicator: One task's result influences the next task

    3. **Large Volume Requests**
       - "Fetch 50+ articles about climate change from multiple sources"
       - "Get comprehensive coverage of tech industry (AI, semiconductors, startups)"
       - Indicator: Words like "comprehensive", "detailed", "all", large numbers

    4. **Time-Consuming Multi-Step Operations**
       - Need to fetch → filter → analyze → summarize
       - Requires multiple API calls across different endpoints
       - Indicator: Complex workflow that benefits from explicit task tracking

    ## Decision Tree
    ```
    Does query mention multiple distinct topics/items?
      ├─ YES → Use PlanningAgent
      └─ NO → Single topic?
            ├─ YES → Use appropriate tool directly
            └─ NO → Complex workflow?
                  ├─ YES → Use PlanningAgent
                  └─ NO → Use appropriate tool directly
    ```

    ## Examples

    ### ✅ Good Use Cases:

    1. **Multi-Topic Query**
       Input: "Get me news about SpaceX launches, Apple earnings, and Formula 1 results"
       → Planning Agent creates 3 tasks (tech RSS, finance search, sports RSS)

    2. **Cross-Source Research**
       Input: "Find recent AI breakthroughs and compare with what Guardian reported"
       → Planning Agent: Task 1 (Tavily Search), Task 2 (Guardian API), Task 3 (Compare)

    3. **Comprehensive Coverage**
       Input: "I want everything about the Trump trial - news, background, analysis"
       → Planning Agent: Task 1 (Recent news), Task 2 (Historical context), Task 3 (Analysis)
    """

    query: str = Field(
        ...,
        description="The original user query that needs to be decomposed into tasks"
    )

class WriteTodos(BaseModel):
    """
    Create and Manage Structured Task Lists for Tracking Progress

    ##When to Use
    - Multi-step or non-trivial tasks requiring coordination.
    - When a user provides multiple tasks or explicitly requests a to-do list.
    -Avoid for single, trivial actions.

    ##Structure
    -Maintain one list containing multiple to-do objects (content, status, id).
    -Use clear, actionable content descriptions.
    -Status must be one of: pending, in_progress, or completed.

    ##Best Practices
    -Only one in_progress task at a time.
    -Mark completed immediately when a task is fully done.
    -Always send the full updated list when making changes.
    -Prune irrelevant items to keep the list focused.

    ##Progress Updates
    -Call WriteTodos again to change a task status or edit content.
    -Reflect real-time progress; don't batch completions.
    -If blocked, keep the status as in_progress and add a new task describing the blocker.

    ##Parameters
    -todos: A list of TODO items with content and status fields.

    ##Returns
    -Updates the agent state with the new to-do list.
    """
    todos: list[ToDo]

class ReadTodos(BaseModel):
    """Read the current TODO list from the agent state every time after recieve tool message.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    """
    todos: list[ToDo]








################################  VALIDATION SCHEMAS  #######################################
class QueryReWrite(BaseModel):
    """
    You'll be passed a query and user wants to re-generate it's query.
    Query optimization schema for rewriting user queries to match specific tool requirements.

    This schema generates tool-specific optimized queries from the user's original input.
    Each tool (TavilySearch, Guardian API, RSS feeds, etc.) has different search characteristics
    and expects queries in different formats for optimal results.

    Rewriting Rules:
    1. Remove conversational elements ("can you", "please", "I want to know")
    2. Extract core entities (people, products, organizations, events)
    3. Transform questions into keyword phrases
    4. Add temporal context when relevant (e.g., "2024", "latest")
    5. Maintain user's original intent and specificity
    6. Keep queries concise (3-10 words optimal)

    The rewritten query should be clear, focused, and optimized for the target tool's
    search mechanism while preserving the user's information need.
    """
    rewritten_query: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description=(
            "The optimized query rewritten for the specific tool. "
            "Must be concise, keyword-focused, and free of conversational fluff. "
            "Should extract core entities and concepts from the user's original query "
            "while adapting to the target tool's search requirements."
        )
    )

class SimpleReflectionSchema(BaseModel):
    """
    Structured output for fact-checking Tavily search results.
    LLM uses this schema to evaluate factual accuracy.
    """
    is_results_factual: bool = Field(description=
            "True if the content appears factually accurate and reliable, "
            "False if it contains misinformation, contradictions, or suspicious claims. "
            "Earth is flat' → False" )

    results_rating : int = Field(ge=0,le=10,description="I want you to rank the results depending on the quality of the content."
                                     "please rate it 1-10")

    reasoning : str = Field(description='Explain why did you decide whether true or false?')


class Tools:
    def __init__(self):
        # Lazy imports to avoid circular dependency
        from src.vector_press.agent.planning_agent import PlanningAgent, write_todos, read_todos

        #embedding_model_config = ModelConfig(model="all-minilm:33m",model_provider_url=settings.OLLAMA_HOST)
        fact_check_llm_config = ModelConfig(model="gpt-oss:120b-cloud",model_provider_url=settings.OLLAMA_HOST, reasoning=False, use_cloud=True)
        query_rewriter_llm_config = ModelConfig(model="gpt-oss:120b-cloud",model_provider_url=settings.OLLAMA_HOST, reasoning=False, use_cloud=True)

        #self.embedding_model = embedding_model_config.get_embedding()

        self.fact_check_llm = fact_check_llm_config.get_llm()
        self.fact_check_structured_llm = self.fact_check_llm.bind_tools([SimpleReflectionSchema])

        self.query_rewriter_llm = query_rewriter_llm_config.get_llm()
        self.query_rewriter_llm.bind_tools([QueryReWrite])

        #self.tavily_search_client = TavilyWebSearchClient()
        self._tavily_without_summary = None
        self._tavily_with_summary = None
        self.guardian_client = GuardianAPIClient()
        self.new_york_times_client = NewYorkTimesAPIClient()
        self.technology_rss_client = TechnologyRSSClient()
        self.sports_rss_client = SportsRSSClient()
        self.planning_agent = PlanningAgent(tools_instance=self)

        # Tool registry: maps schema class to (handler_method, schema_class)
        self.tool_registry = {  #REGISTERY
            #"TavilySearchSchema": (self.tavily_web_search, TavilySearchSchema),
            "TheGuardianApiSchema": (self.guardian_api, TheGuardianApiSchema),
            "NewYorkTimesApiSchema": (self.new_york_times_api, NewYorkTimesApiSchema),
            "TechnologyRSSFeedSchema": (self.technology_rss, TechnologyRSSFeedSchema),
            "SportsRSSFeedSchema": (self.sports_rss, SportsRSSFeedSchema),
            "WriteTodos" : (write_todos, WriteTodos),
            #"ReadTodos" : (read_todos, ReadTodos),
            "PlanningAgentSchema": (self.planning_agent.execute, PlanningAgentSchema),
        }

    def execute_tool(self, tool_name: str, args: dict, state):
        """
        Execute a tool by name with validation.

        Args:
            tool_name: Name of the tool schema (e.g., "TavilySearchSchema")
            args: Dictionary of arguments to validate and pass to the tool
            state: Current state of agent (only used by planning tools)

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool_name is not registered
        """

       # re_written_args = self.query_rewriter(tool_name=tool_name,query=query)
        #args['query'] = re_written_args
       # print(args)
        handler, schema = self.tool_registry[tool_name] #handler is our called tool

        # Special handling for PlanningAgentSchema - skip validation, pass state directly
        if tool_name == "PlanningAgentSchema":
            query = state.query
            print(query)
            return handler(query)

        validated_args = self.checks_args_true_or_not(current_fields=args,true_fields=schema)

        # Planning tools need state access, other tools don't
        if tool_name in ["WriteTodos", "ReadTodos"]:
            return handler(state, validated_args)
        else:
            return handler(validated_args)

    def _reflection(self, tool_result:str ) -> dict:
        """
        Validates the result of the tool.

        Args:
            tool_result : Result of the tool
        Returns:
        """

        evaluating_prompt = f"""You are a fact-checking expert. Your job is to 
        validate web search results for factual accuracy.
                
        Tavily Search Result:
        {tool_result}
        
        Your task:
        1. Identify any factual claims in the content
        2. Check if claims contradict widely known facts (e.g., election results, 
        historical events, scientific consensus)
        3. Look for red flags: outdated info, speculation presented as fact, contradictions
        4. Assign a confidence level
        
        Examples of red flags:
        - "Trump won the 2020 election" (FALSE - Biden won)
        - "Bitcoin was invented in 1995" (FALSE - 2008/2009)
        - Mixing past events with future speculation
        
        Use the passed schema to return your structured assessment."""


        after_validate = self.fact_check_structured_llm.invoke([
            {
                "role" : "system",
                "content" : evaluating_prompt
            }
        ])

        tool_call = after_validate.tool_calls
        args = tool_call[0].get("args", {})
        return args

    @staticmethod
    def checks_args_true_or_not(current_fields, true_fields) -> BaseModel:
        try:
            validated = true_fields(**current_fields)
            print(f"validated: {validated}")
            return validated
        except ValidationError as e:
            logging.error(e)

    def query_rewriter(self, tool_name: str, query:str) -> dict:
        """
        Re-generates the user's query for enhance the tool's result

        Args:
            tool_name: Name of the tool (e.g., "TavilySearchSchema")
            query: Query string to pass to the tool
        Returns:
            args: Keeps rest of the arguments, only re-writes the query
        """
        #TODO birkac (3-5) tane, kullanicinin query sine gore 3 5 tane farkli query daha uretecek

        users_wants = f""" You MUST use QueryReWrite tool here. If you see that message you MUST use QueryReWrite tool.
        You are an expert query optimizer for news retrieval systems. Your job is to transform the user's 
        raw query into optimized search queries tailored for different news sources and search engines.
        query to re generate : {query}
        which tool is using currently: {tool_name}
        """

        system_prompt = f"""
          Original query: {query}
          Target tool: {tool_name}
          Please return the optimized query depending below instructions.

          <examples>
          User: "I want to buy iMac mini m4, what do you think? should I buy it?"
          - TavilySearch: "Mac Mini M4 2024 review performance value analysis"
          - Guardian/NYT: "Mac Mini M4 Apple announcement"
          - TechnologyRSS: "Mac Mini M4 Apple release"
        
          User: "Can you fetch latest news about Ukraine and Russia war?"
          - TavilySearch: "Ukraine Russia war latest news 2024"
          - Guardian/NYT: "Ukraine Russia conflict"
          - TechnologyRSS: N/A (not tech-related)
        
          User: "NBA results"
          - SportsRSS: "NBA results scores"
          - TavilySearch: "NBA game results scores today"
        
          User: "What is the rank of Arsenal in Premier League?"
          - SportsRSS: "Arsenal Premier League standings"
          - TavilySearch: "Arsenal Premier League standings table 2024"
        
          User: "Did Kamala Harris win the last election?"
          - TavilySearch: "Kamala Harris 2024 election results winner"
          - Guardian/NYT: "Kamala Harris election results"
          </examples>
        
          <output_format>
          Return the optimized query as a string. Do NOT include explanations, just the rewritten query.
          </output_format>
        """

        re_written_query = self.query_rewriter_llm.invoke([
            {
                "role" : "user",
                "content" : users_wants
            },
            #{"role" : "system", "content" : system_prompt}
        ])

        tool_call = re_written_query.tool_calls
        print(tool_call)
        args = tool_call[0].get("args",{})

        re_written_query = self.checks_args_true_or_not(args, QueryReWrite)
        print(re_written_query)
        return re_written_query
        






















#########################  TOOL HANDLERS  ########################################

    def guardian_api(self, query: str, max_pages: int = 1, page_size: int = 3) -> list[dict]:
        """News Retrieve Tool"""
        validation = TheGuardianApiSchema(query=query, max_pages=max_pages, page_size=page_size)
        return self.guardian_client.search(validation)

    def new_york_times_api(self, query: str) -> list[dict]:
        """News Retrieve Tool"""
        validation = NewYorkTimesApiSchema(query=query)
        return self.new_york_times_client.search(validation)

    def technology_rss(self, query: str) -> list[dict]:
        """Technology RSS Feed"""
        validation = TechnologyRSSFeedSchema(query=query)
        return self.technology_rss_client.search(validation)

    def sports_rss(self, query: str) -> list[dict]:
        """Sports RSS Feed"""
        validation = SportsRSSFeedSchema(query=query)
        return self.sports_rss_client.search(validation)


    #DEPRECATED
    def tavily_web_search(self,summarize:bool, query: str, max_results: int = 4, topic: str = 'general') -> str:
        """
        Web Search Tool.

        Args:
            query: Search query string
            max_results: Number of search results to return
            topic: Search topic type ('general', 'finance', 'news')
            summarize: summarize search results
        Returns:
            AI-generated answer from Tavily
        """
        client = TavilyWebSearchClient(summarize=summarize)

        validation = TavilySearchSchema(query=query, max_results=max_results, topic=topic)
        search_results = client.search(validation, summarize=summarize)   #TODO config ekle buraya, hepsi alir ihtiyaci olan kullanir (mesela burada max_result olacak)

        return search_results









    def guardian_api_tool(self) -> list[dict]:
        """Guardian API Tool."""

        # Create StructuredTool
        description ="""
            This is The Guardian API
            Use this tool for NEWS searches (world, politics, business, culture, etc.).
        
            When to use:
            - User asks for news about world events, politics, or general current affairs
            - User wants business news, economics, or corporate stories
            - User asks for culture, lifestyle, or opinion pieces
            - User wants ARCHIVED news articles (Guardian has extensive archives)
        
            Think first: Is this a general news query (politics, world, business, culture)?
            If yes, use this tool.
            """

        guardian_api_tool = StructuredTool.from_function(
            func=self.guardian_api,
            name="guardian_api",
            description=description,
            args_schema=TheGuardianApiSchema,
        )
        return guardian_api_tool

    def web_search_tool(self, summarize:bool= True):
        """
        Web Search Tool.

        Args:

            summarize: summarize search results
        Returns:
            AI-generated answer from Tavily
        """

        if summarize:
            if self._tavily_with_summary is None:
                self._tavily_with_summary = TavilyWebSearchClient(summarize=True)
            client = self._tavily_with_summary
        else:
            if self._tavily_without_summary is None:
                self._tavily_without_summary = TavilyWebSearchClient(summarize=False)
            client = self._tavily_without_summary


        web_search_func = partial(self.tavily_web_search, summarize=summarize)

        description = """
                Use this tool for general web searches.
            
                When to use:
                - User asks for tutorials, guides, or how-to information (e.g., "how to learn Python")
                - User wants historical information (e.g., "history of Bitcoin", "what is quantum computing")
                - User asks about concepts, definitions, or explanations (e.g., "explain blockchain")
                - User wants financial market data or analysis (use topic='finance')
                - User asks for general knowledge not requiring current news
            
                Think first: Is this a general information query or a how-to question? If yes, use this tool.
            """

        web_search_tool = StructuredTool.from_function(
            func=web_search_func, #summary is tavily's parameter
            name="web_search",
            description=description,
            args_schema=TavilySearchSchema,
        )
        return web_search_tool

    def technology_rss_feed(self) -> list[dict]:
        """Technology RSS Feed"""

        description = """
            Use this tool for TECHNOLOGY-RELATED CURRENT NEWS queries only.
            You can use this tool to get Technology current topics.
            This tool returns you article's link, title and publication dates.

            When to use:
            - User asks about recent tech news (e.g., "latest AI developments", "new iPhone release")
            - User wants current events in: AI, cybersecurity, startups, tech products, semiconductors
        
            Think first: Does the user want CURRENT TECHNOLOGY NEWS? If yes, use this tool.
            """

        technology_rss_feed = StructuredTool.from_function(
            func=self.technology_rss,
            name="technology_rss_feed",
            description=description,
            args_schema=TechnologyRSSFeedSchema,
        )
        return technology_rss_feed

    def sports_rss_feed(self) -> list[dict]:
        """Sports RSS Feed"""

        description ="""
            Use this tool for SPORTS-RELATED CURRENT NEWS queries only.
            You can use this tool to get Sports current topics.
            This tool returns you article's link, title and publication dates.
        
            When to use:
            - User asks about recent sports news (e.g., "latest football scores", "NBA results")
            - User wants current events in: football, basketball, tennis, cricket, olympics, motorsports
        
            Think first: Does the user want CURRENT SPORTS NEWS? If yes, use this tool.
        """

        sports_rss_feed = StructuredTool.from_function(
            func=self.sports_rss,
            name="sports_rss_feed",
            description=description,
            args_schema=SportsRSSFeedSchema,
        )
        return sports_rss_feed




    def reflection_tool(self):
        """Reflection Tool"""

        description = """
        
        """


# TODO toollarin tamamina (gerekli olanlarin) reflection ekle, handlerlar icerisine, cunku ana classlarinin isi bu degil, _tools_call icerisinde yapmak da uygun olmayacak
# TODO reflectionu da baska bir web search ile external ekleyebilirsin

