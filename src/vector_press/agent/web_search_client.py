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


llm_config = ModelConfig(model="gpt-oss:20b-cloud",use_cloud=True, api_key=settings.OLLAMA_API_KEY)
llm = llm_config.get_llm()

class BaseWebSearchClient(ABC):
    def __init__(self, api_key: str):
        self._api_key = api_key

    @abstractmethod
    def search(self, #validation
               query:str, max_results:int, topic:str
               ) -> list[str]:
        pass

class TavilyWebSearchClient(BaseWebSearchClient):
    def __init__(self):
        super().__init__(api_key=settings.TAVILY_API_KEY)
        # Initialize the actual Tavily client here
        self.tavily_client = TavilyClient(api_key=self._api_key)

    def search(self, #validation
               query:str, max_results:int, topic:str
               ) -> list[str]:
        """Main search method - this is what base_agent.py should call"""


        try:
            '''
            base_params = validation.model_dump()

            response = self.tavily_client.search(
                query=base_params['query'],
                max_results=base_params['max_results'],
                topic=base_params['topic'],
                #include_answer=True
                include_raw_content=True,
            )


            '''

            response = self.tavily_client.search(
                query=query,
                max_results=max_results,
                topic=topic,
                #include_raw_content=True,  # Get full cleaned content
                include_raw_content=True,
            )
           # '''
            # Extract all content values


            results = response['results']

            summarized_list = []

            for i, result in enumerate(results):
                raw_content = result['raw_content']
                summary = llm.invoke([
                    {"role": "system", "content": SUMMARIZER_PROMPT},
                    {"role": "user", "content": raw_content},
                ])

                results_as_str = "\n\n".join([
                    f"SOURCE: {i+1}\n"
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"Content: {(summary.content, 'N/A')}\n"
                    #f"Score: {result.get('score', 'N/A')}\n\n"
                ])

                summarized_list.append(results_as_str)

            response =  "\n".join(summarized_list)


            print('ossuruk')
            #contents = [result['content'] for result in response['results']]
            #return contents
            return response

        except Exception as e:
            print(f"Couldn't retrieve anything: {datetime.now().astimezone(tz=settings.TIME_ZONE)}")
            return [f"Web search failed: {str(e)}"]

class LinkUpwebSearchClient(BaseWebSearchClient):
    def __init__(self):
        super().__init__(api_key=settings.ADD_LINKUPS_API_KEY)


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
