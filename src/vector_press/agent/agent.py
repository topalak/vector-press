from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from src.vector_press.agent.news_api_client import GuardianAPIClient
from src.vector_press.agent.web_search_client import TavilyWebSearchClient
from src.vector_press.agent.tools_validation import TavilySearch, GuardianSearchRequest
from src.vector_press.model_config import ModelConfig
from config import settings

import logging
import time

from src.vector_press.agent.state import AgentState

logger = logging.getLogger(__name__)

pruning_llm_config = ModelConfig(model="qwen3:0.6b", model_provider_url=settings.OLLAMA_HOST)

INSTRUCTIONS = """You are a smart and helpful news assistant. Your name is Big Brother.

<task>
Your job is to use tools to perform user's commands and find information to answer user's questions about news and current events.
You can use any of the tools provided to you.
You can call these tools in series or in parallel, your functionality is conducted in a tool-calling loop.
</task>

<available_tools>
1. **search_guardian_articles**: For NEWS-related searches. If there is 'news' word in query you probably need to call search_guardian_articles.
2. **tavily_web_search**: For general web searches - use this for GENERAL searches and topics like technology guides, finance information, etc.
</available_tools>

<tool_guideline>
1. You MUST set user's query as 'query' not as 'q' every time while you passing it through the tool caller.
    <example>
          {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "User's search query"
              }
            },
            "required": ["query"],
            "additionalProperties": false
          }
    </example>
</tool_guideline>

<pay_attention>
Each response should ONLY use context that directly relates to the user's CURRENT question. Never mix information 
from previous unrelated queries unless user wants it.
</pay_attention>
"""
tool_pruning_prompt = """You are an expert at extracting relevant information from documents.

Your task: Analyze the provided document and extract ONLY the information that directly answers or supports the user's specific request. Remove all irrelevant content.

User's Request: {user_request}

Instructions for pruning:
1. Keep information that directly addresses the user's question
2. Preserve key facts, data, and examples that support the answer
3. Remove tangential discussions, unrelated topics, and excessive background
4. Maintain the logical flow and context of relevant information
5. If multiple subtopics are discussed, focus only on those relevant to the request
6. Preserve important quotes, statistics, and research findings when relevant

Return the pruned content in a clear, concise format that maintains readability while focusing solely on what's needed to answer the user's request."""

#TODO add response format for make it more reliable, because I have changed the model from llama3.2:3b to qwen3:8b output format is changed totally
# response_format: This adds a node before END. This will call and LLM.with_structured_output and the output will be formatted to
# match the given schema and returned in the 'structured_response' state key.

class VectorPressAgent:
    """Handles Agent's processing and response generation"""

    def __init__(self, llm):
        """Initialize agent with model and build graph."""
        self.pruning_llm = pruning_llm_config.get_llm()
        self.llm = llm

        self.tavily_search_client = TavilyWebSearchClient()
        self.guardian_client = GuardianAPIClient()

        tools = [TavilySearch, GuardianSearchRequest]
        self.structured_llm = self.llm.bind_tools(tools=tools)

        self.state: AgentState = AgentState(  #check is there tools_call in default llm
            context_window=[SystemMessage(content=INSTRUCTIONS)],
            query="",
           # pruned_message=None
        )
        self.app = self._build_graph()

    def _llm_call(self, state: AgentState) -> AgentState:
        """LLM call that handles both initial user input and continuation after tools"""
        user_input = state.query

        if not isinstance(state.context_window[-1], ToolMessage):   #IF (last message is NOT a ToolMessage)
            state.context_window.append(HumanMessage(content=user_input))

        start_time = time.time()
        response = self.structured_llm.invoke(state.context_window)  #state AIMessage
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"LLM response generation took {elapsed_time:.2f} seconds")

        state.context_window.append(response)
        return state

    def _tools_call(self, state: AgentState) -> AgentState:
        """Execute tool calls and add results as ToolMessages"""

        raw_tool_result = ""  # TODO ask it to BBB, im getting error if i remove that variable

        for tool_call in state.context_window[-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})

            match tool_name:
                case "GuardianSearchRequest":
                    try:
                        raw_tool_result = self._search_guardian_articles(GuardianSearchRequest(**args))
                    except Exception as e:
                        logger.warning(f"Guardian API error: {e}")
                        #TODO
                        #try:
                            #bbc search
                        #except Exception as e:
                            #logger.error(f"Both Guardian and BBC failed: {e}")
                            #state.context_window.append(ToolMessage(
                           # content="News sources unavailable",
                           # name=tool_name,
                            #tool_call_id=tool_call["id"]
                            #))
                            #continue
                    #TODO you need to search error types to understand what kind of errors for tool calls. I need to update except block with those specific errors
                case "TavilySearch":
                    try:
                        raw_tool_result = self._tavily_web_search(TavilySearch(**args))
                    except Exception as e:
                        logger.warning(f"Tavily API error: {e}")
                        #TODO same try except block like above, use it linkup as alternative
                        continue
                case _:
                    logger.warning(f"Unknown tool requested: {tool_name}")
                    raw_tool_result = f"Unknown tool: {tool_name}"

            raw_tool_result = '\n'.join(raw_tool_result) #ACTUALLY THAT'S UNNECESSARY, BECAUSE BELOW APPROACH HANDLES ITSELF

            if len(raw_tool_result) > 0:
                #TODO ask BBB how to monitor what pruning_llm takes, I want to both approaches behaviour
               # '''
                pruned_tool_result = self.pruning_llm.invoke([
                    {"role": "system", "content": tool_pruning_prompt.format(user_request=state.query), },
                    {"role": "user", "content": raw_tool_result},
                ])
                #'''

                #below approach has a problem, probably about its tool_pruning_prompt
                '''
                tool_result = self.pruning_llm.invoke([
                SystemMessage(content=tool_pruning_prompt.format(user_request=state.query)),
                HumanMessage(content=tool_result),
                ])
                '''

                state.context_window.append(ToolMessage(
                    content=pruned_tool_result.content,
                    name=tool_name,
                    tool_call_id=tool_call["id"]
                ))
                print('ossururk')
            '''
            else:
                state.context_window.append(ToolMessage(content=f"Nothing retrieved from tool id: {tool_call['id']}",
                                                        name=tool_name,
                                                        tool_call_id=tool_call["id"]))
                                                        '''

        return state

    def _tavily_web_search(self, validation: TavilySearch) -> list[str]:
        """Web Search Tool"""
        return self.tavily_search_client.search(validation)

    def _search_guardian_articles(self, validation: GuardianSearchRequest) -> list[str]:
        """News Retrieve Tool"""
        return self.guardian_client.search_articles(validation)

    def _build_graph(self):
        """Build and return the LangGraph pipeline (internal method)."""
        graph = StateGraph(AgentState)

        graph.add_node('llm_call', self._llm_call)
        graph.add_node('tools_call', self._tools_call)

        graph.add_edge(START, 'llm_call')
        graph.add_conditional_edges(
            source='llm_call',
            path=_should_continue,
            path_map={'continue': 'tools_call', 'end': END}
        )
        graph.add_edge('tools_call', 'llm_call')

        return graph.compile()

    def ask(self, query: str) -> str:
        """Ask the agent a question and maintain conversation state."""

        while True:
            if query.lower() == "exit":
                print("\nGoodbye!")
                break

            self.state.query = query

            # Invoke graph (returns dict) and convert back to AgentState
            result = self.app.invoke(self.state)
            self.state = AgentState(**result)

            if self.state.context_window:
                last_message = self.state.context_window[-1]
                print(last_message.content)

            query = input("\nYou: ").strip()

        return "No response generated"



def _should_continue(state: AgentState):
    """Determine whether to continue with tool calls or end"""
    last_message = state.context_window[-1]
    if last_message.tool_calls:
        return 'continue'
    else:
        return 'end'

def main():

    # Configure logging to show in terminal
    logging.basicConfig(
        level=logging.INFO,  # Show INFO, WARNING, ERROR
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = ModelConfig(model="qwen3:1.7b", model_provider_url=settings.OLLAMA_HOST, reasoning=False)
    llm = config.get_llm()
    agent = VectorPressAgent(llm)

    agent.ask("Can you fetch latest news about Ukraine and Russia war?")
    #can you multiple 15 and 764 by calling tools?
    #Who is Cristiano Ronaldo?
    #Can you fetch 200 articles about Ukraine and Russia war?
    #Can you fetch latest news about Ukraine and Russia war?

if __name__ == '__main__':
    main()
