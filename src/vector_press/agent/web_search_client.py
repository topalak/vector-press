from abc import ABC, abstractmethod
from config import settings
from datetime import datetime
from tavily import TavilyClient

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

# TODO 0.5 scoredan altini alma, ilk olarak ciktilari incele ve ona gore threshold koy
# TODO add many exception parts logger and keep continue

    @abstractmethod
    def search(self, validation, summary:bool,
               #query:str, max_results:int, topic:str
               ) -> list[str]:
        pass

    def _summarizer(self, topic: str, results: list) -> str:
        """
        Optional summarizer function

        Args:
            topic: topic to summarize
            results: list of summary results
        """

        summarized_list = []

        for i, result in enumerate(results):
            raw_content = result['raw_content']
            # Format the prompt with topic and context
            formatted_prompt = SUMMARIZER_PROMPT.format(topic=topic, context=raw_content)
            summary = self.llm.invoke([
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": raw_content},
            ])

            results_as_str = "\n\n".join([
                f"SOURCE: {i + 1}\n"
                f"Title: {result.get('title', 'N/A')}\n"  #it returns N/A if title isn't exist
                f"URL: {result.get('url', 'N/A')}\n"
                f"Content: {summary.content}\n"
                # f"Score: {result.get('score', 'N/A')}\n\n"
            ])

            summarized_list.append(results_as_str)

        response = "\n".join(summarized_list)

        return response

class TavilyWebSearchClient(BaseWebSearchClient):
    def __init__(self, summarize: bool = False):
        super().__init__(api_key=settings.TAVILY_API_KEY, summarize=summarize)
        # Initialize the actual Tavily client here
        self.tavily_client = TavilyClient(api_key=self._api_key)

    def search(self, validation, summarize:bool,
              # query:str, max_results:int, topic:str
               ) -> list[str]:
        """Main search method - this is what base_agent.py should call"""

        #TODO ADD HERE TOPIC, PROBABLY I NEED TO USE SCHEMA FOR THAT, MAYBE I NEED TO PASS DIRECTLY
        # USERS QUERY

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
                summarized = self._summarizer(topic=base_params['topic'], results=results)
                return summarized
            else:
                ai_response_from_tavily = response.get('answer')
            return ai_response_from_tavily

        except Exception as e:
            print(f"Couldn't retrieve anything: {datetime.now().astimezone(tz=settings.TIME_ZONE)}")
            return [f"Web search failed: {str(e)}"]

class LinkUpwebSearchClient(BaseWebSearchClient):
    def __init__(self, summarize: bool = False):
        super().__init__(api_key=settings.ADD_LINKUPS_API_KEY, summarize=summarize)


def main():
    client = TavilyWebSearchClient()
    response = client.search(query='Apple M5 chip information',
                             max_results=10,
                             topic='general')


    answer = response

    #contents = [result['content'] for result in response]

    print('ossuruk')
    print(response)
    print(answer)

if __name__ == '__main__':
    main()
