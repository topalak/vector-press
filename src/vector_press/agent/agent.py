from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from src.vector_press.agent.news_api_client import GuardianAPIClient
from src.vector_press.agent.web_search_client import TavilyWebSearchClient
from src.vector_press.agent.tools_validation import TavilySearch, GuardianSearchRequest
from src.vector_press.llm_embedding_initializer import LLMManager

INSTRUCTIONS = """You are a smart and helpful news assistant. Your name is Big Brother.

<task>
Your job is to use tools to perform user's commands and find information to answer user's questions about news and current events.
You can use any of the tools provided to you.
You can call these tools in series or in parallel, your functionality is conducted in a tool-calling loop.
</task>

<available_tools>
1. **search_guardian_articles**: For NEWS-related searches. If there is 'news' word in query you probably need to call search_guardian_articles.
2. **tavily_web_search**: For general web searches - use this for non-news topics like technology guides, finance information, etc.
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

class AgentState(TypedDict):
    """State class for LangGraph conversation flow"""
    context_window: Annotated[list[BaseMessage], add_messages]  # keeps every type of message with BaseMessage
    query: str
    tool_messages: BaseMessage

class VectorPressAgent:
    """Handles Agent's processing and response generation"""

    def __init__(self, llm_manager: LLMManager, state: AgentState):
        """Initialize with LLM manager, Supabase vector store, and add INSTRUCTIONS to state"""
        self.llm = llm_manager.get_llm()  # Get LLM from manager
        self.tavily_search_client = TavilyWebSearchClient()  # Web search client
        self.guardian_client = GuardianAPIClient()  # News API client

        tools = [self.tavily_web_search, self.search_guardian_articles, #self.newyorktimes
                 ]
        self.structured_llm = self.llm.bind_tools(tools=tools)
        state['context_window'].append(SystemMessage(content=INSTRUCTIONS))


    def llm_call(self, state: AgentState) -> AgentState:
        """LLM call that handles both initial user input and continuation after tools"""
        user_input = state.get('query', '')

        if not state['context_window'] or not isinstance(state['context_window'][-1], ToolMessage):   #IF (messages list is empty) OR (last message is NOT a ToolMessage)
            state['context_window'].append(HumanMessage(content=user_input))

        response = self.structured_llm.invoke(state['context_window'])  #state AIMessage
        state['context_window'].append(response)
        return state

    def tools_call(self, state: AgentState) -> AgentState:
        """Execute tool calls and add results as ToolMessages"""
        for tool_call in state['context_window'][-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})

            if tool_name == "search_guardian_articles":
                #validation_args = args.get('validation') if args.get('validation') else args
                validation_args = args
                validation = GuardianSearchRequest(**validation_args)
                tool_result = self.search_guardian_articles(validation)

            elif tool_name == "tavily_web_search":
                #validation_args = args.get('validation') if args.get('validation') else args
                validation_args = args
                validation = TavilySearch(**validation_args)
                tool_result = self.tavily_web_search(validation)
            else:
                continue

            # TODO, add a summarizer here for tool message

            # Add tool response
            state['context_window'].append(ToolMessage(
                content=tool_result,
                name=tool_name,
                tool_call_id=tool_call["id"]
            ))
        return state

    def tavily_web_search(self, validation: TavilySearch) -> list[str]:
        return self.tavily_search_client.search(validation)

    def search_guardian_articles(self, validation: GuardianSearchRequest):
        return self.guardian_client.search_articles(validation)


def should_continue(state: AgentState):
    """Determine whether to continue with tool calls or end"""
    last_message = state['context_window'][-1]
    if last_message.tool_calls:
        return 'continue'
    else:
        return 'end'

def main():

    print("\nStarting (type 'exit' to quit)...")
    state: AgentState = {
        "context_window": [],
        "query": ""
    }

    llm_manager = LLMManager()
    vectorpress_agent = VectorPressAgent(llm_manager, state)

    graph = StateGraph(AgentState)
    graph.add_node('llm_call', vectorpress_agent.llm_call)
    graph.add_node('tools_call', vectorpress_agent.tools_call)


    graph.add_edge(start_key=START,end_key= 'llm_call')
    graph.add_conditional_edges(source='llm_call',path= should_continue, path_map={'continue':'tools_call', 'end':END})
    graph.add_edge('tools_call', 'llm_call')

    app = graph.compile()

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        state["query"] = user_input

        state = app.invoke(state)

        if state['context_window']:
            print(f"\nBig Brother: {state['context_window'][-1].content}")

if __name__ == "__main__":
    import warnings
    warnings.warn("this function is deprecated", DeprecationWarning)
    main()