from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from src.vector_press.agent.tools import (
    TavilySearchSchema,
    TheGuardianApiSchema,
    PlanningAgentSchema,
TechnologyRSSFeedSchema,
SportsRSSFeedSchema,

    Tools,
)

#TODO can you give me comprehensive summarization of last 24 hours,
# TODO add langchain tracing
from src.vector_press.model_config import ModelConfig
from config import settings

import logging
import time

from src.vector_press.agent.states import AgentState

logger = logging.getLogger(__name__)
#embeddinggemma:latest
#embedding_model_config = ModelConfig(model='all-minilm:33m',model_provider_url=settings.OLLAMA_HOST)
tools_validation = [
    TavilySearchSchema,
    TheGuardianApiSchema,
    #PlanningAgentSchema,
    TechnologyRSSFeedSchema,
SportsRSSFeedSchema,


]

#You can call these tools in series or in parallel. Your functionality is conducted in a tool-calling loop.
#TODO add new york times tool to system instruction
INSTRUCTIONS = """You are a smart and helpful assistant. Your one and only mission is NEWS retrieving and answering based 
query and retrieved items. Your name is Big Brother.
Users will ask you about news, it can be politics, technology, AI, sport, business, world, recipe, general, whether, etc.
It can be any news topic. 

<task>
Your job is using tools to perform user's commands and find related information to answer user's news questions.
You can use any of the tools provided to you.
</task>

<available_agents>

1. **WebSearchAgent** - General Web Search
   Use for general information, historical data, tutorials, and fact-checking.
   
2. **TheGuardianApiAgent** - TheGuardianApi
   Use for news related queries, this is safe source to fetch news

3. **PlanningAgentSchema**
   Use when:
   1- User query contains MULTIPLE distinct tasks.
   2- If user's query too complicate and it needs to break into smaller steps
   3- Planning Agent has WebSearchAgent and TheGuardianApiAgent, 
   

</available_agents>


## Think Before You Plan
Ask yourself:
- Can I answer this with ONE tool call? → Don't use PlanningAgentSchema
- Do I need 2+ different tools? → Consider PlanningAgentSchema
- Are there multiple distinct topics? → Use PlanningAgentSchema
- Is this complex and multi-step? → Use PlanningAgentSchema
- Would breaking this down improve results? → Use PlanningAgentSchema


### ❌ DO NOT USE PlanningAgentSchema when:

1. **Single, Simple Query**
   - "What's the latest AI news?" → Just use TechnologyRSSFeedSchema
   - "NBA scores today" → Just use SportsRSSFeedSchema
   - Can be answered with ONE tool call

2. **Trivial Operations**
   - "Define artificial intelligence" → Just use TavilySearchSchema
   - Quick fact-checking or single definition lookup

3. **Already Narrow and Specific**
   - "Tesla stock price today" → Single search, no decomposition needed

    **Remember:** PlanningAgentSchema adds overhead. Only use when the benefits
    (organization, completeness, quality) outweigh the cost (extra LLM calls,
    context usage, execution time).
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
        self.llm = llm
        self.structured_llm = self.llm.bind_tools(tools=tools_validation)
        # Initialize tools using the Tools class
        self.tools = Tools()
        self.state: AgentState = AgentState(
            context_window=[SystemMessage(content=INSTRUCTIONS)],
            query="",
            meta_data=[],
            files="",  #TODO we need to offload the results into here
        )
        self.app = self._build_graph()

    def _llm_call(self, state: AgentState) -> AgentState:
        """LLM call, makes tool call and responses to user's query if it needed"""

        start_time = time.time()
        response = self.structured_llm.invoke(state.context_window)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"LLM response generation took {elapsed_time:.2f} seconds")

        state.context_window.append(response)
        return state

    def _tools_call(self, state: AgentState) -> AgentState:
        """Execute tool calls and add results as ToolMessages"""

        for tool_call in state.context_window[-1].tool_calls:
            # TODO we except model makes parallel tool calls for API NEWS by using The Guardian and NYT
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})
            #TODO LLM had generated the parameters of tool's till here, but we need to re-write the query for more robust news specially API calls, and it needs to generate queries
            # each RSS and Api calls because they act different

            '''
            raw_tool_result = self.tools.execute_tool(tool_name, args, state.query)

            if tool_name == "TheGuardianApiSchema":
                for article in raw_tool_result:
                    state.meta_data.append({
                        "source": article.get("source", ""),
                        "publication_date": article.get("publication_date", "")
                    })

                raw_tool_result = [article.get("body_text", "") for article in raw_tool_result]
            '''
            #'''
            try:
                raw_tool_result = self.tools.execute_tool(tool_name, args, state)
                '''
                if tool_name == "TheGuardianApiSchema":
                    for article in raw_tool_result:
                            state.meta_data.append({
                                "source": article.get("source", ""),
                                "publication_date": article.get("publication_date", "")
                            })

                    raw_tool_result = [article.get("body_text", "") for article in raw_tool_result]
                '''
                    #take validation like
                    #base_params = validation.model_dump()
                    #base_params['query'] = state.query
                    #llm in PlanningAgent icin uretecegi query'den kurtulacagiz

                # Convert result to string if it's not already
                if not isinstance(raw_tool_result, str):
                    raw_tool_result = str(raw_tool_result)

                state.context_window.append(ToolMessage(content=raw_tool_result,
                                                    tool_name=tool_name,
                                                    tool_id=tool_call["id"]))
            except Exception as e:
                logger.warning(f"{tool_name} execution error: {e}")
                import traceback
                traceback.print_exc()
                continue

        return state

    def _build_graph(self):
        """Build and return the LangGraph pipeline."""
        graph = StateGraph(AgentState)

        graph.add_node('llm_call', self._llm_call)
        graph.add_node('tools_call', self._tools_call)

        graph.add_edge(START, 'llm_call')
        graph.add_conditional_edges(
            source='llm_call',
            path=self._should_continue,
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
            self.state.context_window.append(HumanMessage(content=query))

            # Invoke graph (returns dict) and convert back to AgentState
            result = self.app.invoke(self.state)
            self.state = AgentState(**result)

            if self.state.context_window:
                last_message = self.state.context_window[-1]
                print(last_message.content)

            query = input("\nYou: ").strip()

        return "No response generated"

    @staticmethod
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

    # Suppress noisy HTTP logs from httpx (used by Ollama client)
    #logging.getLogger("httpx").setLevel(logging.WARNING)
    #logging.getLogger("httpcore").setLevel(logging.WARNING)
    #gpt-oss:120b-cloud
    config = ModelConfig(model="gpt-oss:120b-cloud", model_provider_url=settings.OLLAMA_HOST, reasoning=False, use_cloud=True)
    llm = config.get_llm()
    agent = VectorPressAgent(llm)

    agent.ask(query="Just make a web search tool call for mac mini m4?")
    #can you multiple 15 and 764 by calling tools?
    #Who is Cristiano Ronaldo?
    #Can you fetch 200 articles about Ukraine and Russia war?
    #I want to buy Imac mini m4, what do you think? should I buy it?
    #Can you fetch latest news about Ukraine and Russia war?
    #I want you to fetch latest news about new Mac Mini m4, I want to buy a new one
    #NBA results
    #Did kamala harris win the last election?
    #What is the rank of the Arsenal in Premier League?

#TODO there is a big problem that we are totally hoping the tools retrieve true answers but its not going like that. TavilySearch tool get the news which says kamala harris won the last selection.
#TODO if user asks 2 different topics at the same query, we need to make different tool calls and i think we can handle it with planning tool

#TODO we need to add summarizer when user wants comprehensive information (that means we need to fetch bigger sized pages)
if __name__ == '__main__':
    main()
