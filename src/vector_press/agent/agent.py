from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from datetime import datetime
import datetime
from vector_press.agent.api_clients import GuardianAPIClient
from vector_press.agent.tools_validation import TavilySearchRequest, GuardianSearchRequest
from vector_press.llm_embedding_initializer import LLMManager
from config import settings
from tavily import TavilyClient

# TODO get hints from system instruction leakages
INSTRUCTIONS = """You are a smart and helpful news assistant. Your name is Big Brother.

Your job is to use tools to perform user's commands and find information to answer user's questions about news and current events.
You can use any of the tools provided to you.
You can call these tools in series or in parallel, your functionality is conducted in a tool-calling loop.

You have access to the following tools:
1. **search_guardian_articles**: For NEWS-related searches - use this for current news, breaking news, politics, world events, etc.
2. **tavily_web_search**: For general web searches - use this for non-news topics like technology guides, finance information, etc.

Tool Selection Guidelines:
- For NEWS topics (politics, war, breaking news, current events): Use search_guardian_articles
- For GENERAL topics (how-to guides, technical info, finance): Use tavily_web_search

If no relevant information is available, search for current information using the appropriate tool based on the topic type.

CRITICAL: Each response should ONLY use context that directly relates to the user's CURRENT question. Never mix information from previous unrelated queries. When database context is not relevant, always use web search to provide current information.
</News Database Context>
"""

class AgentState(TypedDict):
    """State class for LangGraph conversation flow"""
    messages: Annotated[list[BaseMessage], add_messages]  # keeps every type of message with BaseMessage
    query: str

class VectorPressAgent:
    """Handles Agent's processing and response generation"""

    def __init__(self, llm_manager: LLMManager, state: AgentState):
        """Initialize with LLM manager, Supabase vector store, and add INSTRUCTIONS to state"""
        self.llm = llm_manager.get_llm()  # Get LLM from manager
        self.tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        self.guardian_client = GuardianAPIClient()
        tools = [self.tavily_web_search, self.search_guardian_articles]
        self.structured_llm = self.llm.bind_tools(tools=tools)
        state['messages'].append(SystemMessage(content=INSTRUCTIONS))


    def llm_call(self, state: AgentState) -> AgentState:
        """LLM call that handles both initial user input and continuation after tools"""
        user_input = state.get('query', '')

        if not state['messages'] or not isinstance(state['messages'][-1], ToolMessage):   #IF (messages list is empty) OR (last message is NOT a ToolMessage)
            state['messages'].append(HumanMessage(content=user_input))

        response = self.structured_llm.invoke(state['messages'])  #state AIMessage
        state['messages'].append(response)
        return state

    def tools_call(self, state: AgentState) -> AgentState:
        """Execute tool calls and add results as ToolMessages"""
        for tool_call in state['messages'][-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})

            if tool_name == "search_guardian_articles":
                # Extract nested validation data if present
                validation_args = args.get('validation', args)
                validation = GuardianSearchRequest(**validation_args)
                tool_result = self.search_guardian_articles(validation)

            elif tool_name == "tavily_web_search":
                # Extract nested validation data if present
                validation_args = args.get('validation', args)
                validation = TavilySearchRequest(**validation_args)
                tool_result = self.tavily_web_search(validation)
            else:
                continue

            # Add tool response
            state['messages'].append(ToolMessage(
                content=tool_result,
                name=tool_name,
                tool_call_id=tool_call["id"]
            ))

        return state

    def tavily_web_search(self, validation: TavilySearchRequest) -> str:

        try:
            response = self.tavily_client.search(
                query=validation.query,
                max_results= validation.max_results,
                topic= validation.topic,
            )

            # Extract all content values
            contents = [result['content'] for result in response['results']]
            return contents

        except Exception as e:
            print(f"Couldn't retrieve any chunk: {datetime.datetime.now().astimezone(tz=settings.TIME_ZONE)}")
            return f"Web search failed: {str(e)}"

    def search_guardian_articles(self, validation: GuardianSearchRequest):
            return self.guardian_client.search_articles(
                query=validation.query,
                section=validation.section,
                max_pages= validation.max_pages,
                page_size= validation.page_size,
                order_by=validation.order_by,
                )


def should_continue(state: AgentState):
    """Determine whether to continue with tool calls or end"""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'continue'
    else:
        return 'end'

def main():

    print("\nStarting (type 'exit' to quit)...")
    state: AgentState = {
        "messages": [],
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

        # Store user input in query field for process_query to access
        state["query"] = user_input

        state = app.invoke(state)

        if state['messages']:
            print(f"\nBig Brother: {state['messages'][-1].content}")

if __name__ == "__main__":
    main()