from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from src.vector_press.agent.tools import (
    TavilySearchSchema,
    TheGuardianApiSchema,
    NewYorkTimesApiSchema,
    TechnologyRSSFeedSchema,
    SportsRSSFeedSchema,
    WriteTodos,
    ReadTodos,
    Tools,
)
#TODO can you give me comprehensive summarization of last 24 hours,
# TODO add langchain tracing
from src.vector_press.model_config import ModelConfig
from config import settings

import os
import logging
import time

from src.vector_press.agent.state import AgentState

logger = logging.getLogger(__name__)

pruning_llm_config = ModelConfig(model="qwen3:0.6b",model_provider_url=settings.OLLAMA_HOST)
embedding_model_config = ModelConfig(model='all-minilm:33m',model_provider_url=settings.OLLAMA_HOST)
#embeddinggemma:latest


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

<available_tools>

1. **TavilySearchSchema** - General Web Search
   Use for general information, historical data, tutorials, and fact-checking.

2. **WriteTodos** - Task Planning and Progress Tracking
   Use when:
   1- User query contains MULTIPLE distinct tasks.
   2- eger kullancinin girdigi islem buyuk bir arama kapsamli bir calisma istyiorsa bunu kucuk parcalara bolerek todolar yarat
   
3. **ReadTodos** - Review Current Task List
   Use to check remaining pending tasks after completing a task.

</available_tools>

<workflow_for_multi_task_queries>

When user asks for MULTIPLE things in ONE query (e.g., "Fetch news about AI, Ukraine war, and NBA"):

STEP 1: IDENTIFY & CREATE TODO LIST
   Example: "Fetch news about AI developments, Ukraine war, and NBA results"
   → Break into 3 separate tasks:

   Call WriteTodos with ALL tasks as 'pending':
   {
     "todos": [
       {"content": "Fetch news about AI developments", "status": "pending"},
       {"content": "Fetch news about Ukraine war", "status": "pending"},
       {"content": "Fetch NBA results", "status": "pending"}
     ]
   }

STEP 2: START FIRST TASK
   → Call ReadTodos and understand which steps you have and what to do step by step. Begin your duty by first pending
   step.
   {
     "todos": [
       {"content": "Fetch news about AI developments", "status": "pending"},
       {"content": "Fetch news about Ukraine war", "status": "pending"},
       {"content": "Fetch NBA results", "status": "pending"}
     ]
   }
   
   
STEP 3: START FIRST TASK
   → Call WriteTodos and update your next step as "in_progress"
   {
     "todos": [
       {"content": "Fetch news about AI developments", "status": "in_progress"},
       {"content": "Fetch news about Ukraine war", "status": "pending"},
       {"content": "Fetch NBA results", "status": "pending"}
     ]
   }

STEP 4: EXECUTE THE TOOL
   → Call TavilySearchSchema (or appropriate tool) to fetch news

STEP 5: UPDATE TASK AFTER GETTING RESULT
   After receiving ToolMessage result call WriteTodos to update the status of current step:
   - If tool succeeded → Mark 'completed'
   - If tool failed → Keep 'in_progress' and try again.

   Call WriteTodos with updated status:
   {
     "todos": [
       {"content": "Fetch news about AI developments", "status": "completed"},
       {"content": "Fetch news about Ukraine war", "status": "pending"},
       {"content": "Fetch NBA results", "status": "pending"}
     ]
   }

STEP 6: CHECK REMAINING TASKS
   → Call ReadTodos to see what's next

STEP 7: REPEAT Steps 2-6 for each remaining task
   Continue until ALL tasks are 'completed'

</workflow_for_multi_task_queries>

<critical_rules>
- ALWAYS include the FULL todo list in WriteTodos (all tasks, not just changes)
- ONLY ONE task should be 'in_progress' at a time
- MUST call WriteTodos to update status AFTER receiving each tool result
- Mark 'completed' based on actual tool success/failure
- For SINGLE task queries, skip WriteTodos and use tools directly
</critical_rules>
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

#TODO add response format for make it more reliable, because I have changed the model from llama3.2:3b to qwen3:8b output format is changed totally, response_format: This adds a node before END. This will call and LLM.with_structured_output and the output will be formatted to match the given schema and returned in the 'structured_response' state key.

class VectorPressAgent:
    """Handles Agent's processing and response generation"""

    def __init__(self, llm):
        """Initialize agent with model and build graph."""
        self.pruning_llm = pruning_llm_config.get_llm()
        self.llm = llm

        # Initialize tools using the Tools class
        self.tools = Tools()

        tools_validation = [
            TavilySearchSchema,
            #TheGuardianApiSchema,
            #NewYorkTimesApiSchema,
            #TechnologyRSSFeedSchema,
            #SportsRSSFeedSchema,
            WriteTodos,
            ReadTodos,
        ]
        self.structured_llm = self.llm.bind_tools(tools=tools_validation)

        self.state: AgentState = AgentState(
            context_window=[SystemMessage(content=INSTRUCTIONS)],
            query="",
            meta_data=[],
            todos=[],
            files="",  #TODO we need to offload the results into here
        )
        self.app = self._build_graph()

    def _llm_call(self, state: AgentState) -> AgentState:
        """LLM call that handles both initial user input and continuation after tools"""

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

                if tool_name == "TavilySearchSchema":
                    state.context_window.append(ToolMessage(content=raw_tool_result,
                                                            name=tool_name,
                                                            tool_call_id=tool_call["id"]))
                elif tool_name == "WriteTodos":
                    print(f"📝 Updated TODO list:\n{raw_tool_result}")
                    state.context_window.append(ToolMessage(content=raw_tool_result,
                                                            name=tool_name,
                                                            tool_call_id=tool_call["id"]))
                elif tool_name == "ReadTodos":
                    print(f"📋 Current TODO list:\n{raw_tool_result}")
                    state.context_window.append(ToolMessage(content=raw_tool_result,
                                                            name=tool_name,
                                                            tool_call_id=tool_call["id"]))
                else:
                    print(f"⚠️ Unknown tool call: {tool_name}")


            except Exception as e:
                logger.warning(f"{tool_name} execution error: {e}")
                continue

        return state

    def _build_graph(self):
        """Build and return the LangGraph pipeline (internal method)."""
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
    os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')

    # Configure logging to show in terminal
    logging.basicConfig(
        level=logging.INFO,  # Show INFO, WARNING, ERROR
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    #gpt-oss:120b-cloud
    config = ModelConfig(model="gpt-oss:120b-cloud", model_provider_url=settings.OLLAMA_HOST, reasoning=False, use_cloud=True)
    llm = config.get_llm()
    agent = VectorPressAgent(llm)

    agent.ask(query="Fetch news about AI developments, Ukraine war, and NBA results?")
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
if __name__ == '__main__':
    main()
