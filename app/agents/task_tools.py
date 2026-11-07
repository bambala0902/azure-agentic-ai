from langchain_core.tools import tool
from langgraph.types import interrupt

from app.models import TaskCreate

@tool
def create_task(
    title: str,
    description :str,
    priority : str,
) -> dict:
    """Create a task after receiving human approval."""

    # Pause the graph before performing the task creation.
    approval = interrupt(
        {
            "type": "approval_required",
            "tool": "create_task",
            "message": "The assistant wants to create this task.",
            "arguments":{
                "title": title,
                "description": description,
                "priority": priority,
            },
        },
    )

    # Resume here after the user makes a decision.
    if not approval.get("approved", False):
        return{
            "status": "rejected",
            "message": "The task was rejected by the user.",
            "title": None,
        }

    from app.main import create_task as save_task
    task = TaskCreate(title=title, description=description, priority=priority)
    result = save_task(task)
    return result.model_dump()

@tool
def list_tasks() -> list[dict]:
    """ Retrieve all existing tasks."""
    from app.main import list_tasks as get_tasks
    return [task.model_dump() for task in get_tasks()]

TASK_TOOLS = [create_task, list_tasks]
    


