from langchain_ollama import ChatOllama, OllamaEmbeddings
from ai_common.llm import load_ollama_model #,_check_and_pull_ollama_model
from config import settings


class ModelConfig:
    def __init__(
        self,
        model:str,
        model_provider_url:str,   #as we can see ":" means its required
        num_ctx:int = 8192,         #when we look it here there is "=" and that means is optional, this is default value and you can change it in your config
        reasoning:bool = False,
        temperature:int = 0,
        num_predict:int = 128,
    ):

        self.model = model
        self.model_provider_url = model_provider_url
        self.num_ctx = num_ctx      #when __init__ invokes it creates dummies when it comes that line they are coming real variables
        self.reasoning = reasoning
        self.temperature = temperature
        self.num_predict = num_predict

    def get_llm(self):
        #we are reaching that method by line 55, and we have self dict which equiv
        load_ollama_model(self.model, self.model_provider_url)

        return ChatOllama(
            model=self.model,
            base_url=self.model_provider_url,
            num_ctx=self.num_ctx,
            reasoning=self.reasoning,
            temperature=self.temperature,
            num_predict=self.num_predict,
        )

    def get_embedding(self):
        #_check_and_pull_ollama_model(self.model_name, self.model_provider_url)
        #ollama_client.embed(model=model_name)  #upload the llm to memory
        load_ollama_model(self.model, self.model_provider_url)

        return OllamaEmbeddings(
            model=self.model,
        )


def main():
    from vector_press.agent import VectorPressAgent  #avoiding circular import dependency, ModelConfig.py: "I need VectorPressAgent first!" -> agent.py: "I need ModelConfig first!" -> ModelConfig.py: "But I'm not finished loading!" ->  It will cause circular dependency error

    config = ModelConfig(model='qwen3:8b', model_provider_url=settings.OLLAMA_HOST)
    #now when above line has invoked it creates a dict which contains model's parameters
    llm = config.get_llm() #This is our llm which we invoked in get_llm method

    agent = VectorPressAgent(llm)
    agent.ask(query= 'Who is Cristiano Ronaldo')


if __name__ == '__main__':
    main()
