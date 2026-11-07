
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app


def test_assistant_chat_returns_approval_request():
    with TestClient(app) as client:
        mock_graph = MagicMock()

        mock_graph.invoke.return_value = {
            "__interrupt__": [
                MagicMock(
                    value={
                        "type": "approval_required",
                        "tool": "create_task",
                        "arguments": {
                            "title": "Learn LangGraph",
                            "description": "",
                            "priority": "high",
                        },
                    }
                )
            ]
        }

        app.state.agent_graph = mock_graph

        response = client.post(
            "/assistant/chat",
            json={
                "message": "Create a task to learn LangGraph",
                "thread_id": "test-approval-001",
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "approval_required"
        assert response.json()["approval"]["tool"] == "create_task"


def test_assistant_resume_approves_workflow():
    with TestClient(app) as client:
        mock_graph = MagicMock()

        mock_graph.invoke.return_value = {
            "messages": [
                MagicMock(content="Task created successfully")
            ]
        }

        app.state.agent_graph = mock_graph

        response = client.post(
            "/assistant/resume",
            json={
                "thread_id": "test-approval-002",
                "approved": True,
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"

        command = mock_graph.invoke.call_args.args[0]
        assert command.resume == {"approved": True}


def test_assistant_resume_rejects_workflow():
    with TestClient(app) as client:
        mock_graph = MagicMock()

        mock_graph.invoke.return_value = {
            "messages": [
                MagicMock(content="Task creation was rejected")
            ]
        }

        app.state.agent_graph = mock_graph

        response = client.post(
            "/assistant/resume",
            json={
                "thread_id": "test-approval-003",
                "approved": False,
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"

        command = mock_graph.invoke.call_args.args[0]
        assert command.resume == {"approved": False}
