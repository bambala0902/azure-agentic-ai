from langchain_core.tools import tool
from app.models import TaskCreate

@tool
def create_task(
    title: str,
    description :str,
    priority : str,
) -> dict:
    """Create a task with the given title, description, and priority."""
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
    


