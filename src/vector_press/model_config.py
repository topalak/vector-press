import logging

from langchain_ollama import ChatOllama, OllamaEmbeddings
#from ai_common.llm import load_ollama_model #,_check_and_pull_ollama_model
from config import settings


from ollama import ListResponse, Client
from tqdm import tqdm
from ollama import Client


def _check_and_pull_ollama_model(model_name: str, ollama_url: str) -> None:
    ollama_client = Client(host=ollama_url)
    response: ListResponse = ollama_client.list()
    available_model_names = [x.model for x in response.models]

    # Modified from https://github.com/ollama/ollama-python/blob/main/examples/pull.py
    if model_name not in available_model_names:
        print(f'Pulling {model_name}')
        current_digest, bars = '', {}
        for progress in ollama_client.pull(model=model_name, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get('status'))
                continue

            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest
def load_ollama_model(model_name: str, ollama_url: str) -> None:
    _check_and_pull_ollama_model(model_name=model_name, ollama_url=ollama_url)
    ollama_client = Client(host=ollama_url)
    try:
        ollama_client.generate(model=model_name)
    except Exception as e:
        logging.error(f'Failed to generate {model_name}: {e},it throws an error because of that, it will try to embed rn')
        ollama_client.embed(model_name)



class ModelConfig:
    def __init__(
        self,
        model:str,
        model_provider_url:str,   #as we can see ":" means its required
        num_ctx:int = 8192,         #when we look it here there is "=" and that means is optional, this is default value, and you can change it in your config
        reasoning:bool = False,
        temperature:int = 0,
        #num_predict:int = 128,   that causes tool call error, model cant generate tool call because of the limitation.
    ):

        self.model = model
        self.model_provider_url = model_provider_url
        self.num_ctx = num_ctx      #when __init__ invokes it creates dummies when it comes that line they are coming real variables
        self.reasoning = reasoning
        self.temperature = temperature
        #self.num_predict = num_predict

    def get_llm(self):
        #we are reaching that method by line 55, and we have self dict which equiv
        load_ollama_model(self.model, self.model_provider_url)

        return ChatOllama(  #let's wrap our model
            model=self.model,
            base_url=self.model_provider_url,
            num_ctx=self.num_ctx,
            reasoning=self.reasoning,
            temperature=self.temperature,
            keep_alive="5m",
            #num_predict=self.num_predict,
        )

    def get_embedding(self):
        #_check_and_pull_ollama_model(self.model_name, self.model_provider_url)
        #ollama_client.embed(model=model_name)  #upload the llm to memory
        load_ollama_model(self.model, self.model_provider_url)

        return OllamaEmbeddings(
            model=self.model,
            base_url=self.model_provider_url,
            keep_alive=2,
        )


def main():
    from vector_press.agent import VectorPressAgent  #avoiding circular import dependency, model_config.py: "I need VectorPressAgent first!" -> agent.py: "I need ModelConfig first!" -> model_config.py: "But I'm not finished loading!" ->  It will cause circular dependency error

    config = ModelConfig(model='qwen3:8b', model_provider_url=settings.OLLAMA_HOST)
    #now when above line has invoked it creates a dict which contains model's parameters
    llm = config.get_llm() #This is our llm which we invoked in get_llm method

    agent = VectorPressAgent(llm)
    agent.ask(query= 'Who is Cristiano Ronaldo')


if __name__ == '__main__':
    main()
