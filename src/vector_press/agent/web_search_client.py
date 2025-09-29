from abc import ABC, abstractmethod
from config import settings
from datetime import datetime
from tavily import TavilyClient
from vector_press.agent.tools_validation import TavilySearch




class BaseWebSearchClient(ABC):
    def __init__(self, api_key: str):
        self._api_key = api_key

    @abstractmethod
    def search(self, validation) -> list[str]:
        pass

class TavilyWebSearchClient(BaseWebSearchClient):
    def __init__(self):
        super().__init__(api_key=settings.TAVILY_API_KEY)
        # Initialize the actual Tavily client here
        self.tavily_client = TavilyClient(api_key=self._api_key)

    def search(self, validation: TavilySearch) -> list[str]:
        """Main search method - this is what agent.py should call"""
        try:
            response = self.tavily_client.search(
                query=validation.query,
                max_results=validation.max_results,
                topic=validation.topic,
            )
            # Extract all content values
            contents = [result['content'] for result in response['results']]
            return contents

        except Exception as e:
            print(f"Couldn't retrieve any chunk: {datetime.now().astimezone(tz=settings.TIME_ZONE)}")
            return [f"Web search failed: {str(e)}"]