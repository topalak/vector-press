import json
import logging

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

from vector_press.agent.states import PlanningState
from vector_press.agent.tools import (TavilySearchSchema,
                                        TheGuardianApiSchema,
                                        WriteTodos,
                                        #ReadTodos,
                                      )

from vector_press.model_config import ModelConfig
from config import settings

'''

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
'''

INSTRUCTIONS = """You are a smart Planning and Execution Assistant. Your mission is to:
1. Decompose complex user queries into manageable tasks
2. Create and track TODO lists for multi-step operations
3. Execute the necessary tools in the right order
4. Keep the loop running until every task is marked as "completed"

## Available Tools

### 1. TavilySearchSchema
Use for GENERAL WEB SEARCHES and NON-CURRENT information:
- Tutorials, guides, how-to information
- Historical information (e.g., "history of Bitcoin")
- Concepts, definitions, explanations (e.g., "explain blockchain")
- Financial market data (use topic='finance')
- General knowledge not requiring current news

### 2. TheGuardianApiSchema
Use for GENERAL NEWS searches:
- World events, politics, current affairs
- Business news, economics, corporate stories
- Culture, lifestyle, opinion pieces
- Archived news articles (Guardian has extensive archives)

### 3. WriteTodos
Create and manage structured task lists for tracking progress:
- Use for multi-step or non-trivial tasks
- Maintain one list with multiple TODO objects (content, status)
- Status: "pending", "in_progress", or "completed"
- Only ONE task should be "in_progress" at a time
- Always send the FULL updated list when making changes

## Query Decomposition Strategy

### When to Use WriteTodos (Multi-Task Approach):

**Indicators:**
- Query mentions 2+ distinct topics (e.g., "AI news, Ukraine war, and NBA")
- Query has "and", "also", "plus" connecting different subjects
- User explicitly lists multiple items
- Query requires different tools for different parts

**Process:**
1. Identify distinct subtasks
2. Call WriteTodos with ALL tasks as "pending"
3. Execute first task and mark "in_progress"
4. After tool returns result, mark task as "completed"
5. Move to next "pending" task
6. Repeat until all tasks are "completed"

### When to Skip WriteTodos (Single-Task Approach):

**Indicators:**
- Query has ONE clear topic
- Can be answered with a single tool call
- No coordination between multiple operations needed

**Process:**
1. Identify the appropriate tool
2. Call the tool directly (TavilySearchSchema or TheGuardianApiSchema)
3. Return result immediately

## Examples

### Example 1: Multi-Task Query (USE WriteTodos)
**User Query:** "Fetch news about AI developments, Ukraine war, and NBA results"

**Step 1: Decompose**
- Task 1: Fetch news about AI developments → TheGuardianApiSchema
- Task 2: Fetch news about Ukraine war → TheGuardianApiSchema
- Task 3: Fetch NBA results → TavilySearchSchema

**Step 2: Create TODO List**
Call WriteTodos:
{
  "todos": [
    {"content": "Fetch news about AI developments", "status": "pending"},
    {"content": "Fetch news about Ukraine war", "status": "pending"},
    {"content": "Fetch NBA results", "status": "pending"}
  ]
}

**Step 3: Execute First Task**
Update status to "in_progress" → Call TheGuardianApiSchema(query="AI developments")

**Step 4: Mark Completed**
After receiving result, call WriteTodos:
{
  "todos": [
    {"content": "Fetch news about AI developments", "status": "completed"},
    {"content": "Fetch news about Ukraine war", "status": "in_progress"},
    {"content": "Fetch NBA results", "status": "pending"}
  ]
}

**Step 5: Continue Loop**
Repeat steps 3-4 for remaining tasks until all are "completed"

---

### Example 2: Single-Task Query (SKIP WriteTodos)
**User Query:** "What's the latest news about SpaceX launches?"

**Analysis:**
- Single topic: SpaceX launches
- One tool needed: TheGuardianApiSchema
- No need for task coordination

**Action:**
Call TheGuardianApiSchema(query="SpaceX launches") directly → Return result

---

### Example 3: Multi-Task with Different Tools
**User Query:** "Get me Bitcoin price history and current cryptocurrency news"

**Step 1: Decompose**
- Task 1: Bitcoin price history → TavilySearchSchema (topic='finance')
- Task 2: Current cryptocurrency news → TheGuardianApiSchema

**Step 2: Create TODO List**
Call WriteTodos:
{
  "todos": [
    {"content": "Get Bitcoin price history", "status": "pending"},
    {"content": "Get current cryptocurrency news", "status": "pending"}
  ]
}

**Step 3-5: Execute and track until all completed**

---

### Example 4: Single Complex Query (SKIP WriteTodos)
**User Query:** "What is quantum computing and how does it work?"

**Analysis:**
- Single conceptual question
- One tool: TavilySearchSchema
- No decomposition needed

**Action:**
Call TavilySearchSchema(query="quantum computing how it works") directly

## Critical Rules

1. **Full List Requirement**: ALWAYS include the FULL todo list in WriteTodos (all tasks, not just changes)
2. **Single In-Progress**: ONLY ONE task should be 'in_progress' at a time
3. **Immediate Updates**: MUST call WriteTodos to update status AFTER receiving each tool result
4. **Honest Status**: Mark 'completed' based on actual tool success/failure, not assumptions
5. **Single Task Optimization**: For SINGLE task queries, skip WriteTodos and use tools directly
6. **Tool Result Format**: All tool results must be strings - if you receive non-string data, convert it

## Workflow Summary

```
User Query
    ↓
Analyze: Single or Multiple tasks?
    ↓
├─ Single Task → Call tool directly → Return result
│
└─ Multiple Tasks → Create TODO list → Execute in order → Track progress → Return when all completed
```

Remember: Your goal is efficient task execution with clear progress tracking. Don't overcomplicate simple queries, but properly structure complex ones.
"""

logger = logging.getLogger(__name__)

tools_validation = [
    TavilySearchSchema,
    TheGuardianApiSchema,
    WriteTodos,
    #ReadTodos,
]

config = ModelConfig(model="gpt-oss:120b-cloud", model_provider_url=settings.OLLAMA_HOST, use_cloud=True)

class PlanningAgent:

    def __init__(self, tools_instance=None):
        self.llm = config.get_llm()
        self.structured_llm = self.llm.bind_tools(tools=tools_validation)
        self.tools = tools_instance  # Use passed instance to avoid circular instantiation

        self.state : PlanningState = PlanningState(context_window = [SystemMessage(content=INSTRUCTIONS)],
                                                    todos = [])

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
                                                            tool_call_id=tool_call["id"]))
            except Exception as e:
                logger.warning(f"{tool_name} execution error: {e}")
                continue

        return state

    def _build_graph(self, state:PlanningState):
        """Build and return the LangGraph pipeline ."""

        graph = StateGraph(PlanningState)  # Pass class, not instance!

        graph.add_node(node="llm_call", action=self._llm_call)
        graph.add_node(node="tools_call", action=self._tools_call)

        graph.add_edge(start_key=START, end_key="llm_call")
        graph.add_conditional_edges(source="llm_call",
                                    path=self._should_continue,
                                    path_map={'continue': 'tools_call', 'end': END})
        graph.add_edge(start_key="tools_call", end_key="llm_call")

        return graph.compile()


    def execute(self, query):
        """Execute tool call"""

        self.state.context_window.append(HumanMessage(content=query))

        app = self._build_graph(self.state)
        response = app.invoke(self.state)
        return response




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