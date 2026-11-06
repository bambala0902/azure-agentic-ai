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
from app.agents.graph import agent_graph

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
    try:
        result = agent_graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": body.message,
                    }
                ]
            },
            config = {"recursion_limit": 10}
        )
        messages = result["messages"]
        final_message = messages[-1]
        content = (
            final_message.content
            if hasattr(final_message, "content")
            else final_message["content"]
        )

        return {
            "response": content,
            "messages_count": len(messages),
            "workflow" : "langgraph",
        }

    except Exception as exc:
        raise HTTPException(
            status_code = 502,
            detail = "The LangGraph agent could not process the request"
        ) from exc
