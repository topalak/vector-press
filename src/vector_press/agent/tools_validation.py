import logging

from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional

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
            description="You can get the whole user's question as 'query' parameter, if you decide the question is long, "
            "get the most related query as possible as you can")
            #"Extract relevant keywords from user's query for semantic matching. "
            #"Keep it focused on 3-5 keywords for best results.")

class TavilySearch(Query):
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

class GuardianSearchRequest(Query):
    """
    Use this tool for GENERAL NEWS searches (world, politics, business, culture, etc.).

    When to use:
    - User asks for news about world events, politics, or general current affairs
    - User wants business news, economics, or corporate stories
    - User asks for culture, lifestyle, or opinion pieces
    - User wants ARCHIVED news articles (Guardian has extensive archives)

    Think first: Is this a general news query (politics, world, business, culture)?
    If yes, use this tool.
    """
    show_fields: str = Field(default="all", description="Field names to show")
    section: Optional[str] = Field(default=None, description="Guardian section (e.g., 'world', 'politics', 'business', 'technology')")
    pages: int = Field(description="Maximum number of results to return, dont forget pages X page_size = Article number ")
    page_size: int = Field(default=23,#ge=1,le=200),
                           )
        #description="Number of articles PER PAGE. "
          #  "Use 2-5 for quick results, 10-20 for moderate results, "
          #  "50+ for comprehensive results. "
          #  "Note: Total articles = page_size × max_pages.")

    max_pages: int = Field(default=1,ge=1,le=20,
            description="Number of pages to fetch. "
            "Use 1 for most queries (combine with higher page_size for more results). "
            "Use 2-5+ only if user explicitly wants very comprehensive results. "
            "Note: Total articles = page_size × max_pages.")

class TechnologyRSSFeed(Query):
    """
    Use this tool for TECHNOLOGY-RELATED CURRENT NEWS queries only.

    When to use:
    - User asks about recent tech news (e.g., "latest AI developments", "new iPhone release")
    - User wants current events in: AI, cybersecurity, startups, tech products, semiconductors

    Think first: Does the user want CURRENT TECHNOLOGY NEWS? If yes, use this tool.
    """

class SportsRSSFeed(Query):
    """
    Use this tool for SPORTS-RELATED CURRENT NEWS queries only.

    When to use:
    - User asks about recent sports news (e.g., "latest football scores", "NBA results")
    - User wants current events in: football, basketball, tennis, cricket, olympics, motorsports

    Think first: Does the user want CURRENT SPORTS NEWS? If yes, use this tool.
    """
