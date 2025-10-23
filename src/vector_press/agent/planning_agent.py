#from src.vector_press.agent.tools importtodo
import json
import logging

from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, SystemMessage

from vector_press.agent.news_api_client import NewYorkTimesAPIClient
from vector_press.agent.states import PlanningState
from vector_press.agent.tools import (TavilySearchSchema,
                                        TheGuardianApiSchema,
                                        WriteTodos,
                                        ReadTodos,
                                        Tools,)

from vector_press.model_config import ModelConfig
from config import settings


INSTRUCTIONS = """You are a smart Planner assistant. Your one and only mission is create TODOs and track them. 
You need to decompose the user's query into separate plans and you will 

<examples>
    explain what way to separate query into smaller parts
</examples>

<available_tools>
    1- TavilySearchSchema: You can use that tool for general web searches
    2- TheGuardianApiSchema: This is safe source to fetch news
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

logger = logging.getLogger(__name__)

tools_validation = [
    TavilySearchSchema,
    TheGuardianApiSchema,
    WriteTodos,
    ReadTodos,
]

config = ModelConfig(model="gpt-oss:120b-cloud", model_provider_url=settings.OLLAMA_HOST, use_cloud=True)

class PlanningAgent:

    def __init__(self, tools_instance=None):
        self.llm = config.get_llm()
        self.structured_llm = self.llm.bind_tools(tools=tools_validation)
        self.tools = tools_instance  # Use passed instance to avoid circular instantiation

        self.state : PlanningState = PlanningState(context_window = [SystemMessage(content=INSTRUCTIONS)],
                                                    todos = [])
        print('ossuruk') #check state's latest situation

        #TODO add here self.invoke().... because when we call that tool it will invoke immediately
        #self._build_graph(state)

    def _llm_call(self, state:PlanningState) -> PlanningState:
        """LLM call, makes tool call"""

        response = self.structured_llm.invoke(state.context_window)
        state.context_window.append(response)

        return state


    def _tools_call(self, state:PlanningState) -> PlanningState:
        for tool_call in state.context_window[-1].tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("args", {})

            try:
                response = self.tools.execute_tool(tool_name, args, state)
                state.context_window.append(ToolMessage(content=response,
                                                        name=tool_name,
                                                        tool_id=tool_call["id"]))

            except Exception as e:
                logger.warning(f"{tool_name} execution error: {e}")
                continue

        return state

    def _build_graph(self, state:PlanningState):
        """Build and return the LangGraph pipeline ."""

        graph = StateGraph(state)

        graph.add_node(node="llm_call", action=self._llm_call)
        graph.add_node(node="tools_call", action=self._tools_call)

        graph.add_edge(start_key=self._llm_call, end_key=self._tools_call)
        graph.add_edge(start_key=self._tools_call, end_key=self._llm_call)
        graph.add_conditional_edges(source=self._llm_call,
                                    path=self._should_continue,
                                    path_map={'continue': 'tools_call', 'end': END})

        graph.compile()
        #app = graph.invoke(self.state)
        return graph


    def execute(self, state:PlanningState):
        """Execute tool call"""
        self.state.context_window.append(state)

        graph = self._build_graph(state)
        app = graph.invoke(state)
        return app




    @staticmethod
    def _should_continue(state:PlanningState) -> bool:
        last_message = state.context_window[-1]
        if last_message.tool_calls:
            return 'continue'
        else:
            return 'end'




def write_todos(state, validation):
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        validation: List of Todo items with content and status
        state: current state of agent
    Returns:
    """
    todos_list = validation.todos
    state.todos = todos_list

    todos_dict = [{"content": todo.content, "status": todo.status} for todo in todos_list]
    message = f"Updated TODO List:\n\n{json.dumps(todos_dict, indent=2)}"

    return message # Todo burada elde ettigimiz yeni todo lari return edip daha sonra update edilmis state i tekrar model e verecegiz ilk sirada yer alan pending icin tavily web search yapacak
# TODO daha sonrasinda oradan gelen sonucu reflection ile kontorl edip okay verirsek readtodo toolunu cagirip completed deyip sonra yine write todos olacak sanirim


def read_todos(state, validation):
    """Read the current TODO list from agent state.

    Args:
        state: AgentState object
        validation: ReadTodos pydantic model (not used, kept for signature consistency)

    Returns:
        str: Formatted TODO list or message if empty
    """

    todos = validation.todos

    if not todos:
        return "No TODOs currently in the list"

    result = "Current TODO list:\n"
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo.status, "❓")
        result += f"{i}. {emoji} {todo.content} ({todo.status})\n"
    print('ossuruk')
    return result.strip()