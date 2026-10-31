from fastapi import FastAPI, HTTPException, status
from app.models import TaskCreate, TaskResponse, TaskSuggestionRequest
from app.llm import(
    LLMNotconfigured,
    LLMProviderError,
    TaskExplanation,
    Tasksuggestion,
    explain_task,
    suggest_task,
) 
app = FastAPI(
    title="Azure Agentic AI",
    description="My hands-on AI engineering learning project",
    version="0.1.0",
)

# Temporary in-memory storage for this lesson.
tasks: dict[int, TaskResponse] = {}
next_task_id = 1

@app.get("/")
def root():
    return {
        "message": "Welcome my Azure Agentic AI project!",
        "lesson": 1,
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "azure-agentic-ai",
    }

@app.post("/tasks", response_model=TaskResponse, status_code= status.HTTP_201_CREATED)
def create_task(task: TaskCreate) -> TaskResponse:
    global next_task_id
    created_task = TaskResponse(
        id = next_task_id,
        title = task.title,
        description = task.description,
        priority = task.priority,
        completed = False,
    )
    tasks[next_task_id] = created_task
    next_task_id += 1

    return created_task


@app.get("/tasks", response_model=list[TaskResponse])
def list_tasks() -> list[TaskResponse]:
    return list(tasks.values())


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: int) -> TaskResponse:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")
    return task

@app.post("/tasks/suggest", response_model=Tasksuggestion)
def suggest_task_endpoint(body : TaskSuggestionRequest) -> Tasksuggestion:
    try:
        return suggest_task(body.request)

    except LLMNotconfigured as exc:
        raise HTTPException(status_code = 503, detail="AI service not configured") from exc

    except LLMProviderError as exc:
        detail = "AI service could not generate a task suggestion"
        cause = exc.__cause__
        if cause is not None:
            detail = f"{detail}. {type(cause).__name__}: {cause}"
        raise HTTPException(status_code=502, detail=detail) from exc


@app.post("/tasks/explain", response_model=TaskExplanation)
def explain_task_endpoint(body: TaskExplanationRequest) -> TaskExplanation:
    try:
        return explain_task(body.request)

    except LLMNotconfigured as exc:
        raise HTTPException(status_code=503, detail="AI service not configured") from exc

    except LLMProviderError as exc:
        detail = "AI service could not explain the task"
        cause = exc.__cause__
        if cause is not None:
            detail = f"{detail}. {type(cause).__name__}: {cause}"
        raise HTTPException(status_code=502, detail=detail) from exc
