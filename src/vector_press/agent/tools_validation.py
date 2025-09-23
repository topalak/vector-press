from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class TavilySearchRequest(BaseModel):
    """Validation model for Tavily web search tool."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Max results to return")  #ge = "greater than or equal to" (≥)  and   le = "less than or equal to" (≤)
    include_domains: Optional[List[str]] = Field(
        default=["theguardian.com", "bbc.com", "cnn.com", "reuters.com"],
        description="Domains to include in search"
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Domains to exclude from search"
    )






