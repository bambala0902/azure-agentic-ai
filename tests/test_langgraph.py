
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_langgraph_chat_endpoint():
    fake_result = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "I can help manage your tasks.",
            },
        ]
    }

    with patch(
        "app.main.agent_graph.invoke",
        return_value=fake_result,
    ) as mock_invoke:

        response = client.post(
            "/assistant/chat",
            json={"message": "Hello"},
        )

    assert response.status_code == 200
    assert response.json()["response"] == (
        "I can help manage your tasks."
    )
    assert response.json()["workflow"] == "langgraph"

    mock_invoke.assert_called_once()


def test_langgraph_agent_failure():
    with patch(
        "app.main.agent_graph.invoke",
        side_effect=RuntimeError("Model unavailable"),
    ):

        response = client.post(
            "/assistant/chat",
            json={"message": "Create a task"},
        )

    assert response.status_code == 502


def test_langgraph_rejects_short_message():
    response = client.post(
        "/assistant/chat",
        json={"message": "OK"},
    )

    assert response.status_code == 422
