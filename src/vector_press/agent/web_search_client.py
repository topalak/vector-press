from abc import ABC, abstractmethod
from idlelib import query

from config import settings
from datetime import datetime
from tavily import TavilyClient
import asyncio

from vector_press.model_config import ModelConfig

SUMMARIZER_PROMPT = """
You are a world class researcher who is working on a report about a specific topic.

<goal>
Generate a very high quality informative summary of the given context in accordance with the topic.
</goal>

The topic you are working on:
<topic>
{topic}
</topic>

The context to use in generating the informative summary:
<context>
{context}
</context>

Prepare your summary according to the topic. 
Include all necessary information related with the topic in your summary.
"""



class BaseWebSearchClient(ABC):
    def __init__(self, api_key: str, summarize: bool = False):
        self._api_key = api_key
        self._summarize = summarize
        self.llm = None  # Lazy load

        if summarize:
            llm_config = ModelConfig(
                model="gpt-oss:20b-cloud",
                num_ctx=8192,
                use_cloud=True,
                api_key=settings.OLLAMA_API_KEY
            )
            self.llm = llm_config.get_llm()

# TODO add a threshold for outputs of tavily search, but analyze them first to see when outputs starts shitting
# TODO add many exception parts logger and keep continue

    @abstractmethod
    def search(self, validation, summary:bool,
               #summarize:bool, query:str, max_results:int, topic:str
               ) -> list[str]:
        pass

    async def _summarize_async(self, query: str, results: list):
        """
        Asynchronously summarize multiple search results in parallel.

        Args:
            query: Query string for context
            results: List of result dicts from Tavily API

        Returns:
            List of formatted summary strings
        """

        async def _summarize_single_async(query: str, result: dict, index: int):
            """Summarize a single result"""
            raw_content = result['raw_content']
            formatted_output = SUMMARIZER_PROMPT.format(topic=query, context=raw_content)
            summary = await self.llm.ainvoke([  #this await is for async llm invoking
                {"role": "system", "content": formatted_output},
            ])

            results_as_str = "\n\n".join([
                f"SOURCE: {index + 1}\n"
                f"Title: {result.get('title', 'N/A')}\n"  #it returns N/A if title isn't exist
                f"URL: {result.get('url', 'N/A')}\n"
                f"Content: {summary.content}\n"
            ])

            return results_as_str

        # Create tasks with index for proper numbering
        tasks = [_summarize_single_async(query=query, result=result, index=i) for i, result in enumerate(results) ]

        # Run all tasks in parallel
        summarized_content = await asyncio.gather(*tasks)  #this await is for all tasks to run async

        return summarized_content


class TavilyWebSearchClient(BaseWebSearchClient):
    def __init__(self, summarize: bool = False):
        super().__init__(api_key=settings.TAVILY_API_KEY, summarize=summarize)
        # Initialize the actual Tavily client here
        self.tavily_client = TavilyClient(api_key=self._api_key)

    def search(self, validation, summarize: bool
               # summarize:bool, query:str, max_results:int, topic:str
               ) -> str:
        """Main search method - this is what base_agent.py should call"""

        try:
            base_params = validation.model_dump()

            response = self.tavily_client.search(
                query=base_params['query'],
                max_results=base_params['max_results'],
                topic=base_params['topic'],
                include_answer=True,
                include_raw_content=True,
            )

            results = response['results']

            if summarize:
                # Run async function from sync context
                summarized = asyncio.run(
                    self._summarize_async(query=base_params['query'], results=results)
                )
                # Join list of summaries into single string
                return "\n\n".join(summarized)
            else:
                ai_response_from_tavily = response.get('answer', '')
                return ai_response_from_tavily

        except Exception as e:
            print(f"Couldn't retrieve anything: {datetime.now().astimezone(tz=settings.TIME_ZONE)}")
            return f"Web search failed: {str(e)}"

class LinkUpwebSearchClient(BaseWebSearchClient):
    def __init__(self, summarize: bool = False):
        super().__init__(api_key=settings.ADD_LINKUPS_API_KEY, summarize=summarize)


def main():
    client = TavilyWebSearchClient(summarize=True)
    response = client.search(summarize=True,
                            query='Chinese citizens died because CS:GO skin update news rumor',
                             max_results=10,
                             topic='general')


    answer = response

    #contents = [result['content'] for result in response]

if __name__ == '__main__':
    main()
