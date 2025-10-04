from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from src.vector_press.agent.news_api_client import GuardianAPIClient
from src.vector_press.agent.web_search_client import TavilyWebSearchClient
from src.vector_press.agent.tools_validation import TavilySearch, GuardianSearchRequest
from src.vector_press.ModelConfig import ModelConfig
from config import settings

from src.vector_press.agent.state import AgentState

pruning_llm_config = ModelConfig(model="llama3.2:3b", model_provider_url=settings.OLLAMA_HOST)

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

class VectorPressAgent:
    """Handles Agent's processing and response generation"""

    def __init__(self, llm):
        """Initialize agent with model and build graph."""


        self.pruning_llm = pruning_llm_config.get_llm()
        #self.pruning_llm = self.pruning_llm.append(SystemMessage(content=tool_pruning_prompt))

        self.llm = llm

        self.tavily_search_client = TavilyWebSearchClient()
        self.guardian_client = GuardianAPIClient()

        tools = [TavilySearch, GuardianSearchRequest]
        self.structured_llm = self.llm.bind_tools(tools=tools)

        self.state: AgentState = AgentState(  #check is there tools_call in default llm
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

            tool_result = ""  #TODO ask it to BBB, im getting error if i remove that variable

            match tool_name:
                case "GuardianSearchRequest":
                    tool_result = self.search_guardian_articles(GuardianSearchRequest(**args))
                case "TavilySearch":
                    tool_result = self.tavily_web_search(TavilySearch(**args))
                case _:
                    tool_result = f"Unknown tool: {tool_name}"

            tool_result = '\n'.join(tool_result)
            if len(tool_result) > 0:
                tool_result = self.pruning_llm.invoke([
            {"role": "system", "content": tool_pruning_prompt.format(user_request=state.query[-1]),},
            {"role": "user", "content": tool_result},
        ])

            state.context_window.append(ToolMessage(
                content=tool_result,
                name=tool_name,
                tool_call_id=tool_call["id"]
            ))
        return state

    def tavily_web_search(self, validation: TavilySearch) -> list[str]:   #TODO check both method's return types (debug)
        """Web Search Tool"""
        print(f"validation : {validation}")
        return self.tavily_search_client.search(validation)

    def search_guardian_articles(self, validation: GuardianSearchRequest) -> list[str]:
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


    config = ModelConfig(model="qwen3:8b", model_provider_url=settings.OLLAMA_HOST, reasoning=True)
    llm = config.get_llm()
    agent = VectorPressAgent(llm)

    agent.ask("Can you fetch latest news about Ukraine and Russia war?")
    #can you multiple 15 and 764 by calling tools?
    #Who is Cristiano Ronaldo?
    #Can you fetch 200 articles about Ukraine and Russia war?
    #Can you fetch latest news about Ukraine and Russia war?

if __name__ == '__main__':
    main()
