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
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from contextlib import asynccontextmanager
from app.agents.graph import build_agent
from langgraph.types import Command
from app.models import AssistantResumeRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    connection = sqlite3.connect(
    "agent.memory.db",
    check_same_thread = False,
    )
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    app.state.agent_graph = build_agent(checkpointer)
    yield
    connection.close()

"""Important: agent_memory.db is created locally. The connection stays open while FastAPI is running and is closed during application shutdown.
This is a local learning setup; production deployments need appropriate database lifecycle and concurrency management."""

app = FastAPI(
    title="Azure Agentic AI",
    description="My hands-on AI engineering learning project",
    version="0.1.0",
    lifespan=lifespan,
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


def message_content(message):
    if hasattr(message, "content"):
        return message.content
    return message["content"]


def response_text(messages):
    """Prefer a rejected tool result over the model's later summary."""
    for message in reversed(messages):
        if getattr(message, "name", None) != "create_task":
            continue
        raw = message_content(message)
        if not isinstance(raw, str):
            break
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            break
        if payload.get("status") == "rejected":
            return payload.get("message") or "The task was rejected by the user."
        break
    return message_content(messages[-1])


@app.post("/assistant/chat")
def assistant_chat(body: AssistantRequest):
    try:
        result = app.state.agent_graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": body.message,
                    }
                ]
            },
            config = {
                "configurable":{"thread_id": body.thread_id},
                "recursion_limit": 10}
        )
        # Check whether LangGraph paused for approval.
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            approval_request = interrupts[0].value
            return {
                "status": "approval_required",
                "thread_id": body.thread_id,
                "approval": approval_request,
                "workflow" : "langgraph",
            }

        messages = result["messages"]
        content = response_text(messages)

        return {
            "status": "completed",
            "response": content,
            "messages_count": len(messages),
            "thread_id": body.thread_id,
            "workflow" : "langgraph",
        }

    except Exception as exc:
        raise HTTPException(
            status_code = 502,
            detail = "The LangGraph agent could not process the request"
        ) from exc

@app.post("/assistant/resume")
def assistant_resume(body: AssistantResumeRequest):
    config = {
        "configurable": {"thread_id": body.thread_id},
        "recursion_limit": 10,
    }
    try:
        snapshot = app.state.agent_graph.get_state(config)
        if not snapshot.interrupts:
            raise HTTPException(
                status_code=409,
                detail="This thread is not waiting for approval.",
            )

        result = app.state.agent_graph.invoke(
            Command(
                resume={
                    "approved": body.approved,
                }
            ),
            config=config,
        )
        # A workflow can pause again if it encounters another interrupt.
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            approval_request = interrupts[0].value
            return {
                "status": "approval_required",
                "thread_id": body.thread_id,
                "approval": approval_request,
                "workflow" : "langgraph",
            }
        messages = result["messages"]
        content = response_text(messages)
        return {
            "status": "completed",
            "response": content,
            "messages_count": len(messages),
            "thread_id": body.thread_id,     
            "workflow" : "langgraph",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code = 502,
            detail = "The LangGraph workflow could not be resumed."
        ) from exc