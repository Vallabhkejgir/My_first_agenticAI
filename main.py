import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage

from agent import compiled_graph
from state import GraphState, WorkingMemory

app = FastAPI(
    title="Hybrid Multi-Agent Framework API",
    description="FastAPI backend hosting a centralized Supervisor-to-Worker stateful graph agent with working memory and LangGraph checkpointing.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input query or command.")
    thread_id: str = Field(..., description="Unique thread identifier to persist conversation state.")
    user_id: str = Field("user_123", description="The identifier of the user.")
    session_id: str = Field("session_abc", description="The active session identifier.")

class TaskResponse(BaseModel):
    id: str
    title: str
    description: str
    status: str
    assigned_team: Optional[str]

class WorkingMemoryResponse(BaseModel):
    user_id: str
    session_id: str
    facts: List[str]
    tasks: List[TaskResponse]
    current_task_id: Optional[str]
    context_variables: Dict[str, Any]

class MessageResponse(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    thread_id: str
    messages: List[MessageResponse]
    working_memory: WorkingMemoryResponse
    next_step: str

@app.post("/chat", response_model=ChatResponse, summary="Send a message to the multi-agent graph.")
async def chat_endpoint(request: ChatRequest):
    """
    Executes the Multi-Agent LangGraph loop. 
    It persists session state using standard LangGraph checkpointers, allowing conversational context
    and active task progress to be maintained across turns.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        # Retrieve existing state to check if we are starting fresh or appending
        state_snapshot = compiled_graph.get_state(config)
        
        # Prepare graph input
        new_message = HumanMessage(content=request.message)
        
        if not state_snapshot.values:
            # First turn: Initialize working memory and the message thread
            initial_memory = WorkingMemory(
                user_id=request.user_id,
                session_id=request.session_id,
                facts=[],
                tasks=[],
                current_task_id=None,
                context_variables={}
            )
            initial_state = {
                "messages": [new_message],
                "working_memory": initial_memory,
                "next_step": "supervisor"
            }
            # Run graph from scratch
            result = compiled_graph.invoke(initial_state, config=config)
        else:
            # Subsequent turn: pass only the new message, LangGraph handles merging
            input_update = {
                "messages": [new_message],
                "next_step": "supervisor" # Ensure we kickstart back to the supervisor
            }
            result = compiled_graph.invoke(input_update, config=config)
            
        # Parse result to API friendly response
        messages_out = []
        for msg in result.get("messages", []):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages_out.append(MessageResponse(role=role, content=str(msg.content)))
            
        wm_data = result.get("working_memory")
        working_memory_out = WorkingMemoryResponse(
            user_id=wm_data.user_id,
            session_id=wm_data.session_id,
            facts=wm_data.facts,
            tasks=[
                TaskResponse(
                    id=t.id,
                    title=t.title,
                    description=t.description,
                    status=t.status,
                    assigned_team=t.assigned_team
                ) for t in wm_data.tasks
            ],
            current_task_id=wm_data.current_task_id,
            context_variables=wm_data.context_variables
        )
        
        return ChatResponse(
            thread_id=request.thread_id,
            messages=messages_out,
            working_memory=working_memory_out,
            next_step=result.get("next_step", "end")
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@app.get("/state/{thread_id}", response_model=Optional[ChatResponse], summary="Retrieve the complete active state for a thread.")
async def get_state_endpoint(thread_id: str):
    """
    Retrieves the serialized state snapshot stored in the memory checkpointer for a given thread_id.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = compiled_graph.get_state(config)
    
    if not state_snapshot.values:
        raise HTTPException(status_code=404, detail=f"No active session state found for thread_id: {thread_id}")
        
    result = state_snapshot.values
    
    messages_out = []
    for msg in result.get("messages", []):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages_out.append(MessageResponse(role=role, content=str(msg.content)))
        
    wm_data = result.get("working_memory")
    working_memory_out = WorkingMemoryResponse(
        user_id=wm_data.user_id,
        session_id=wm_data.session_id,
        facts=wm_data.facts,
        tasks=[
            TaskResponse(
                id=t.id,
                title=t.title,
                description=t.description,
                status=t.status,
                assigned_team=t.assigned_team
            ) for t in wm_data.tasks
        ],
        current_task_id=wm_data.current_task_id,
        context_variables=wm_data.context_variables
    )
    
    return ChatResponse(
        thread_id=thread_id,
        messages=messages_out,
        working_memory=working_memory_out,
        next_step=result.get("next_step", "end")
    )

@app.post("/reset/{thread_id}", summary="Reset the state checkpointer for a given thread.")
async def reset_state_endpoint(thread_id: str):
    """
    Resets/clears the saved graph state checkpoint for a thread by writing an empty/initial state update.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # We can 'reset' by writing an empty dictionary or state update
    # In LangGraph, we update state to none or create a new state
    # A simple way to reset in-memory memory checkpointer for a thread is to update the state with empty fields
    try:
        compiled_graph.update_state(
            config,
            {
                "messages": [],
                "working_memory": WorkingMemory(
                    user_id="default",
                    session_id="default",
                    facts=[],
                    tasks=[],
                    current_task_id=None,
                    context_variables={}
                ),
                "next_step": "supervisor"
            }
        )
        return {"status": "success", "message": f"State for thread {thread_id} reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset state: {str(e)}")

from fastapi.responses import HTMLResponse
import os

@app.get("/", response_class=HTMLResponse, summary="Serve the centralized dashboard UI.")
async def root_endpoint():
    """
    Serves the beautiful, unified frontend dashboard single-file HTML.
    """
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        raise HTTPException(status_code=404, detail="Unified index.html file not found in the root directory.")

@app.get("/memory/global/{user_id}", summary="Retrieve the global episodic and semantic memory contents for a user.")
async def get_global_memory_endpoint(user_id: str):
    """
    Exposes the background global databases (Qdrant episodic summaries and LocalGraph nodes/edges) 
    for visual inspection on the dashboard UI.
    """
    from memory_manager import MemoryManagerAgent, graph_store
    try:
        # Retrieve memories by conducting a broad semantic retrieval
        past_vector_facts = MemoryManagerAgent.retrieve_memories(user_id=user_id, query="house blueprints lasagna cooking calculating fibonacci")
        
        # Pull all semantic relationship string visualizations directly from our Graph DB
        semantic_relationships = []
        for edge in graph_store.edges:
            src = edge["source"]
            tgt = edge["target"]
            etype = edge["type"]
            src_display = graph_store.nodes[src]["properties"].get("name", src.capitalize())
            tgt_display = graph_store.nodes[tgt]["properties"].get("name", tgt.capitalize())
            semantic_relationships.append(f"{src_display} -[{etype}]-> {tgt_display}")
            
        return {
            "user_id": user_id,
            "vector_memories": past_vector_facts,
            "graph_relations": semantic_relationships
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query global memories: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
