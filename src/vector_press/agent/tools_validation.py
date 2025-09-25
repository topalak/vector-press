from pydantic import BaseModel, Field
from typing import List, Optional


class TavilySearchRequest(BaseModel):
    """Tavily Web Search is for general web searches"""

    query: str = Field(..., min_length=1, max_length=50, description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Max results to return")  #ge = "greater than or equal to" (≥)  and   le = "less than or equal to" (≤)
    topic: Optional[str] = Field(default='general',exclude='news', description="Search topic, you can select one of those 'general', 'finance'")

class GuardianSearchRequest(BaseModel):
    '''This tool is for news searches'''

    query: str = Field(..., min_length=1, max_length=50, description="Search query for news articles")
    section: Optional[str] = Field(default=None, description="Guardian section (e.g., 'world', 'politics', 'business', 'technology')")
    page_size: int = Field(default=5, ge=1, le=200, description="Articles per page. Use 3 for quick/single result")
    max_pages: int = Field(default=1, ge=1, le=20, description="Maximum pages to fetch. Use 1 for quick results")
    order_by: Optional[str] = Field(default="relevance", description="Sort order: 'relevance', 'newest', 'oldest'")

