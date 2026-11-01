from fastapi import FastAPI, HTTPException, status
from app.models import TaskCreate, TaskResponse, TaskSuggestionRequest, TaskExplanationRequest
from app.llm import(
    LLMNotconfigured,
    LLMProviderError,
    TaskExplanation,
    Tasksuggestion,
    explain_task,
    suggest_task,
) 
import json
import ollama
from app.tools.task_tools import TASK_TOOLS
import os
from app.models import AssistantRequest

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


def execute_tool(tool_name: str, arguments: dict):
    """
    Execute only explicitly registered application tools.
    Never execute arbitrary Python code from the LLM.
    """

    if tool_name == "create_task":
        task = TaskCreate.model_validate(arguments)
        return create_task(task).model_dump()

    if tool_name == "list_tasks":
        return [task.model_dump() for task in list_tasks()]

    raise ValueError(f"Tool not allowed: {tool_name}")


@app.post("/assistant/chat")
def assistant_chat(body: AssistantRequest):
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    client = ollama.Client(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        timeout=60,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a task management assistant. "
                "Use create_task when the user explicitly asks "
                "to create a task. Use list_tasks when the user "
                "asks to see existing tasks. Do not claim an action "
                "was completed unless the tool confirms success."
            ),
        },
        {"role": "user", "content": body.message},
    ]

    try:
        # First model call: decide whether to use a tool.
        response = client.chat(
            model=model,
            messages=messages,
            tools=TASK_TOOLS,
        )

        tool_calls = response.message.tool_calls or []

        if not tool_calls:
            return {
                "response": response.message.content,
                "tool_executed": False,
            }

        # Execute at most one tool per request in this lesson.
        if len(tool_calls) > 1:
            raise HTTPException(
                status_code=400,
                detail="Multiple tool calls are not supported yet.",
            )

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        if not isinstance(arguments, dict):
            raise ValueError("Invalid tool arguments")

        result = execute_tool(tool_name, arguments)

        # Send the tool result back to the model.
        messages.append(response.message)

        messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(result),
            }
        )

        # Second model call: generate the final response.
        final_response = client.chat(
            model=model,
            messages=messages,
        )

        return {
            "response": final_response.message.content,
            "tool_executed": True,
            "tool_name": tool_name,
            "tool_result": result,
        }

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="The model returned invalid tool arguments.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI assistant could not complete the request.",
        ) from exc
