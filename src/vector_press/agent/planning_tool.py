#from src.vector_press.agent.tools importtodo
import json

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