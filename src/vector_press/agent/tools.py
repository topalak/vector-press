from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from config import settings

import logging
from src.vector_press.model_config import ModelConfig

from src.vector_press.agent.news_api_client import (
    GuardianAPIClient,
    NewYorkTimesAPIClient,
)
from src.vector_press.agent.web_search_client import TavilyWebSearchClient
from src.vector_press.agent.rss_client import TechnologyRSSClient, SportsRSSClient


#state is a Pydantic model (AgentState), not a dictionary. Pydantic models don't have a .get() method. we aren't able to pass it as dictionary like --> state['context_window'] we need to pass it like
# state.context_window

def validate_data(fields, actual_fields):
    try:
        validated = actual_fields(**fields)
        print(f"validated: {validated}")
        return validated
    except ValidationError as e:
        logging.error(e)

class Query(BaseModel):
    """This is base parameter"""
    query: str = Field(...,min_length=1,max_length=500,
            description="get the most related query as possible as you can")
            #"Extract relevant keywords from user's query for semantic matching. "
            #"Keep it focused on 3-5 keywords for best results."

class TavilySearchSchema(Query):
    """
    Use this tool for GENERAL WEB SEARCHES and NON-CURRENT information queries.

    When to use:
    - User asks for tutorials, guides, or how-to information (e.g., "how to learn Python")
    - User wants historical information (e.g., "history of Bitcoin", "what is quantum computing")
    - User asks about concepts, definitions, or explanations (e.g., "explain blockchain")
    - User wants financial market data or analysis (use topic='finance')
    - User asks for general knowledge not requiring current news

    Think first: Is this a general information query or a how-to question? If yes, use this tool.
    """
    max_results: int = Field(default=2,ge=1,le=20,description=
            "Number of search results to return. "
            "Use 2-3 for quick answers, 5-10 for comprehensive research, "
            "10+ for deep exploration.")

    topic: Literal['general', 'finance'] = Field(
        default='general',description=
            "Search topic type: "
            "'general' - for most queries (tech, science, tutorials, concepts). "
            "'finance' - ONLY when query is about stocks, markets, trading, "
            "financial data, or economic indicators.")

class TheGuardianApiSchema(Query):
    """
    This is The Guardian API
    Use this tool for GENERAL NEWS searches (world, politics, business, culture, etc.).

    When to use:
    - User asks for news about world events, politics, or general current affairs
    - User wants business news, economics, or corporate stories
    - User asks for culture, lifestyle, or opinion pieces
    - User wants ARCHIVED news articles (Guardian has extensive archives)

    Think first: Is this a general news query (politics, world, business, culture)?
    If yes, use this tool.
    """
    show_fields: Literal['all'] = Field(default='all', description="Field names to show")
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


class Tools:
    def __init__(self):
        embedding_model_config = ModelConfig(
            model='all-minilm:33m',
            model_provider_url=settings.OLLAMA_HOST
        )
        self.embedding_model = embedding_model_config.get_embedding()
        self.tavily_search_client = TavilyWebSearchClient()
        self.guardian_client = GuardianAPIClient()
        self.new_york_times_client = NewYorkTimesAPIClient()
        self.technology_rss_client = TechnologyRSSClient(embedding_model=self.embedding_model)
        self.sports_rss_client = SportsRSSClient(embedding_model=self.embedding_model)

        # Tool registry: maps schema class to (handler_method, schema_class)
        self.tool_registry = {
            "TavilySearchSchema": (self.tavily_web_search, TavilySearchSchema),
            "TheGuardianApiSchema": (self.guardian_api, TheGuardianApiSchema),
            "NewYorkTimesApiSchema": (self.new_york_times_api, NewYorkTimesApiSchema),
            "TechnologyRSSFeedSchema": (self.technology_rss, TechnologyRSSFeedSchema),
            "SportsRSSFeedSchema": (self.sports_rss, SportsRSSFeedSchema),
        }

    def execute_tool(self, tool_name: str, args: dict):
        """
        Execute a tool by name with validation.

        Args:
            tool_name: Name of the tool schema (e.g., "TavilySearchSchema")
            args: Dictionary of arguments to validate and pass to the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool_name is not registered
        """
        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler, schema = self.tool_registry[tool_name]
        validated_args = validate_data(args, schema)
        return handler(validated_args)

    def tavily_web_search(self, validation: TavilySearchSchema) -> list[str]:
        """Web Search Tool"""
        return self.tavily_search_client.search(validation)

    def guardian_api(self, validation: TheGuardianApiSchema) -> list[dict]:
        """News Retrieve Tool"""
        return self.guardian_client.search(validation)

    def new_york_times_api(self, validation: NewYorkTimesApiSchema) -> list[dict]:
        """News Retrieve Tool"""
        return self.new_york_times_client.search(validation)

    def technology_rss(self, validation: TechnologyRSSFeedSchema) -> list[str]:
        """Technology RSS Feed"""
        return self.technology_rss_client.search(validation)

    def sports_rss(self, validation: SportsRSSFeedSchema) -> list[str]:
        """Sports RSS Feed"""
        return self.sports_rss_client.search(validation=validation)
