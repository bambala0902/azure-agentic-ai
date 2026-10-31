from typing import Literal
from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    title: str = Field(
        min_length=3,
        max_length=100,
        description="A short description of the task",
    )
    description: str = Field(
        default='',
        max_length=500,
        description="Optional task details",
    )
    priority: Literal["low", "medium", "high"] = "medium"


class TaskResponse(BaseModel):
    id: int
    title: str
    description: str
    priority: Literal["low", "medium", "high"]
    completed:bool = False

class TaskSuggestionRequest(BaseModel):
    request: str = Field(
        min_length=3,
        max_length=500,
        description="Describe the task you want the AI to suggest.",
    )
class TaskExplanationRequest(BaseModel):
    request: str = Field(
        min_length=3,
        max_length=500,
        description="Explain the task you want the AI to explain.",
    )

