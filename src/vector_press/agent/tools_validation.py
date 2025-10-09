from pydantic import BaseModel, Field
from typing import Literal

class TavilySearch(BaseModel):
    """For general web searches - use this for GENERAL searches and topics like technology guides, finance information, etc."""
#Literal is about VALUE constraints, not TYPE coercion, we can not solve it with literal
    query: str = Field(..., min_length=1, max_length=500, description="Write the best web search keywords depending user's input ")
    max_results: int = Field(default=2, ge=1, le=20, description="Max results to return")  #ge = "greater than or equal to" (≥)  and   le = "less than or equal to" (≤)
    topic: Literal['general', 'finance'] = Field(default='general', description="Search topic, you can select one of those 'general', 'finance'")

class GuardianSearchRequest(BaseModel):
    """For NEWS-related searches. If there is 'news' word in query you probably need to call search_guardian_articles."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query for news articles")
    section: str = Field(default=None, description="Guardian section (e.g., 'world', 'politics', 'business', 'technology')")
    page_size: int = Field(default=2, ge=1, le=200, description="Articles per page. Use 3 for quick/single result")
    max_pages: int = Field(default=1, ge=1, le=20, description="Maximum pages to fetch. Use 1 for quick results") #TODO I need to write more detailed explanations, because when I say fetch 200 articles, it defines both max_pages and page_size 200
    order_by: str = Field(default="relevance", description="Sort order: 'relevance', 'newest', 'oldest'. If user insist for one of them in it's query just change it in that situation")

#state is a Pydantic model (AgentState), not a dictionary. Pydantic models don't have a .get() method. we aren't able to pass it as dictionary like --> state['context_window'] we need to pass it like
# state.context_window