from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from src.vector_press.agent.news_api_client import GuardianAPIClient
from src.vector_press.agent.web_search_client import TavilyWebSearchClient
from src.vector_press.agent.tools_validation import TavilySearch, GuardianSearchRequest
from src.vector_press.llm_embedding_initializer import LLMManager


from src.vector_press.agent.state import AgentState

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


class VectorPressAgent:
    """Handles Agent's processing and response generation"""

    def __init__(self, model_name: str = 'llama3.2:3b'):
        """Initialize agent with model and build graph."""
        llm_manager = LLMManager()
        self.llm = llm_manager.get_llm(model_name=model_name)

        self.tavily_search_client = TavilyWebSearchClient()
        self.guardian_client = GuardianAPIClient()

        tools = [TavilySearch, GuardianSearchRequest]
        self.structured_llm = self.llm.bind_tools(tools=tools)

        self.state: AgentState = AgentState(
            context_window=[SystemMessage(content=INSTRUCTIONS)],
            query="",
            tool_messages=None
        )
        self.app = self._build_graph()

    def llm_call(self, state: AgentState) -> AgentState:
        """LLM call that handles both initial user input and continuation after tools"""
        user_input = state.query

        if not isinstance(state.context_window[-1], ToolMessage):   #IF (last message is NOT a ToolMessage)
            state.context_window.append(HumanMessage(content=user_input))

        response = self.structured_llm.invoke(state.context_window)  #state AIMessage
        state.context_window.append(response)
        return state

    def tools_call(self, state: AgentState) -> AgentState:
        """Execute tool calls and add results as ToolMessages"""
        for tool_call in state.context_window[-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})

            if tool_name == "GuardianSearchRequest":
                print(f"Guardian's args{GuardianSearchRequest(**args)}")
                tool_result = self.search_guardian_articles(GuardianSearchRequest(**args))

            elif tool_name == "TavilySearch":
                print(f"Tavily's args{TavilySearch(**args)}")
                tool_result = self.tavily_web_search(TavilySearch(**args))  #we are checking values with pydantic HERE by "TavilySearch(**args))

            else:
                tool_result= f"There is no tool for that"
                print('') #TODO handle here by using tool_call's args

            # TODO, add a summarizer here for tool message

            state.context_window.append(ToolMessage(
                content=tool_result,
                name=tool_name,
                tool_call_id=tool_call["id"]
            ))
        return state  # TODO 'Using **tavily_web_search** tool to find the product of 15 and 764:   15 * 764 = 11460'   its not related with web search and model has answered that with its knowledge we need to handle that problem, actually there is no problem with that, the only issue is LLM does not adding a value invalid tool calls.

    def tavily_web_search(self, validation: TavilySearch) -> list[str]:
        """Web Search Tool"""
        print(f"validation : {validation}")
        return self.tavily_search_client.search(validation)

    def search_guardian_articles(self, validation: GuardianSearchRequest) -> list[dict]:
        """News Retrieve Tool"""
        return self.guardian_client.search_articles(validation) #look for what shape is returns and set it in signature

    def _build_graph(self):
        """Build and return the LangGraph pipeline (internal method)."""
        graph = StateGraph(AgentState)

        graph.add_node('llm_call', self.llm_call)
        graph.add_node('tools_call', self.tools_call)

        graph.add_edge(START, 'llm_call')
        graph.add_conditional_edges(
            source='llm_call',
            path=should_continue,
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



def should_continue(state: AgentState):
    """Determine whether to continue with tool calls or end"""
    last_message = state.context_window[-1]
    if last_message.tool_calls:
        return 'continue'
    else:
        return 'end'


def main():

    agent = VectorPressAgent(model_name='qwen3:8b')  #TODO, add num_ctx and reasoning and please add docstring for reasoning, but when user goes to source of VectorPressAgent wont be able to see llm manager
    #TODO handle it, you can call directly llm manager
    agent.ask("Can you fetch 200 articles about Ukraine and Russia war?")
    #can you multiple 15 and 764 by using tool calls?
    #Who is Cristiano Ronaldo?
    #Can you fetch latest news about Ukraine and Russia war?

if __name__ == '__main__':
    main()