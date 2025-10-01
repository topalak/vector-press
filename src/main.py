from vector_press.agent.agent import VectorPressAgent

agent = VectorPressAgent(model_name='llama3.2:3b')
response = agent.ask("Can you fetch latest news about Ukraine and Russia war?")


#TODO
# for now I need to convert database uploading method to a tool
# re-write a query generator like re-writing topic (especially topic because it has news, general, finance topics) , query, time_range etc.
# tool definitionlari rag yapabilirsin, bunun yerine cok daha basit bir llm judge ile yaparsan daha iyi bir sonuc alirsi

# TODO guardian ya da nyt mcp yapilabilir
# TODO warning python arastir
