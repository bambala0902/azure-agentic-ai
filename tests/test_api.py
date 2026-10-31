
import pytest
from fastapi.testclient import TestClient

from app.main import app, tasks
import app.main as main_module
from app.llm import(
    LLMNotconfigured,
    LLMProviderError,
    Tasksuggestion,
) 
client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_tasks():
    tasks.clear()
    main_module.next_task_id = 1
    yield
    tasks.clear()
    main_module.next_task_id = 1


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["application"] == "azure-agentic-ai"

def test_create_task():
    response = client.post(
        "/tasks",
        json={
            "title": "Learn LangGraph",
            "description": "Build an agent",
            "priority": "high",
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["id"] == 1
    assert data["title"] == "Learn LangGraph"
    assert data["priority"] == "high"
    assert data["completed"] is False


def test_list_tasks():
    client.post("/tasks", json={"title": "Build an API"})
    client.post("/tasks", json={"title": "Write tests"})

    response = client.get("/tasks")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_task():
    created = client.post(
        "/tasks",
        json={"title": "Read documentation"},
    )
    task_id = created.json()["id"]

    response = client.get(f"/tasks/{task_id}")

    assert response.status_code == 200
    assert response.json()["title"] == "Read documentation"


def test_missing_task_returns_404():
    response = client.get("/tasks/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task 999 not found"


def test_invalid_title_is_rejected():
    response = client.post(
        "/tasks",
        json={"title": "OK"},
    )

    assert response.status_code == 422


def test_invalid_priority_is_rejected():
    response = client.post(
        "/tasks",
        json={
            "title": "Build an agent",
            "priority": "urgent",
        },
    )

    assert response.status_code == 422


def test_task_title_is_required():
    response = client.post(
        "/tasks",
        json={"description": "A task without a title"},
    )

    assert response.status_code == 422


def test_suggest_task_success(monkeypatch):
    def fake_suggest_task(user_request):
        assert "Azure architecture" in user_request
        return Tasksuggestion(
            title="Review Azure architecture",
            description="Review the current cloud design",
            priority="high",
        )

    monkeypatch.setattr(
        "app.main.suggest_task",
        fake_suggest_task,
    )

    response = client.post(
        "/tasks/suggest",
        json={
            "request": "Create a high-priority task to review "
            "my Azure architecture"
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "title": "Review Azure architecture",
        "description": "Review the current cloud design",
        "priority": "high",
    }


def test_suggest_task_missing_configuration(monkeypatch):
    def fake_suggest_task(user_request):
        raise LLMNotconfigured("Missing configuration")

    monkeypatch.setattr(
        "app.main.suggest_task",
        fake_suggest_task,
    )

    response = client.post(
        "/tasks/suggest",
        json={"request": "Review my cloud architecture"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "AI service not configured"
    )


def test_suggest_task_provider_failure(monkeypatch):
    def fake_suggest_task(user_request):
        raise LLMProviderError("Internal provider details")

    monkeypatch.setattr(
        "app.main.suggest_task",
        fake_suggest_task,
    )

    response = client.post(
        "/tasks/suggest",
        json={"request": "Review my cloud architecture"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "AI service could not generate a task suggestion"
    )


def test_suggest_task_rejects_short_request():
    response = client.post(
        "/tasks/suggest",
        json={"request": "OK"},
    )

    assert response.status_code == 422
