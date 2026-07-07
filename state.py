from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Annotated
from langgraph.graph.message import AnyMessage, add_messages

class Task(BaseModel):
    id: str
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    assigned_team: Optional[str] = None  # e.g., "Research", "Execution"
    dependencies: List[str] = Field(default_factory=list)

class WorkingMemory(BaseModel):
    user_id: str
    session_id: str
    facts: List[str] = Field(default_factory=list, description="Extracted facts about the user/session.")
    tasks: List[Task] = Field(default_factory=list, description="Decomposed tasks in the current run.")
    current_task_id: Optional[str] = Field(None, description="The ID of the task currently being processed.")
    context_variables: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value store for session variables.")

class GraphState(BaseModel):
    """The state of our multi-agent graph, managed as a Pydantic model."""
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    working_memory: WorkingMemory
    next_step: Optional[str] = Field("supervisor", description="The next agent/node to execute.")
