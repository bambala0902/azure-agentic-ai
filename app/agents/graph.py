import os
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph.prebuilt import ToolNode, tools_condition
from app.agents.task_tools import TASK_TOOLS


def build_agent(checkpointer):
    model = ChatOllama(
        model= os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        base_url = os.getenv(
            "OLLAMA_HOST", "http://localhost:11434"
        ),
        temperature = 0,
    )

    #Give the model access to our approved Python tools.
    model_with_tools = model.bind_tools(TASK_TOOLS)

    def agent_node(state: MessagesState):
        system_message = {
            "role": "system",
            "content": (
                "You are a task management assistant. "
                "Use create_task when the user explicitly requests "
                "task creation. Use list_tasks when the user requests "
                "existing tasks. Do not claim an action succeeded "
                "unless a tool confirms it. "
                "If the tool result says the task was rejected, "
                "say that the task was not created. "
                "Do not repeat its title or priority as if it exists."
            ),
        }

        messages = [system_message] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    #Define the workflow nodes.
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TASK_TOOLS))

    # Define workflow connections.
    workflow.add_edge(START, "agent")

    # Decide whether to call a tool or finish.
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # After executing a tool, return to the agent.
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


