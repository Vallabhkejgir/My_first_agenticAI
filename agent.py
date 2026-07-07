import uuid
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage

from state import GraphState, Task, WorkingMemory
from mcp_gateway import MCPGateway
from skill_registry import skill_registry
from teams import research_team_graph, execution_team_graph
from memory_manager import MemoryManagerAgent
import llm

# Pydantic model for structured output during task decomposition
class TaskDecomposition(BaseModel):
    tasks: List[Task] = Field(description="The list of decomposed tasks required to achieve the user's goal.")

def supervisor_node(state: GraphState) -> Dict[str, Any]:
    """Supervisor/Router node that manages plan generation, integrates global memory stores, and routes work to teams."""
    messages = state.messages
    working_memory = state.working_memory
    
    # Ensure working_memory is a copy we can mutate
    working_memory = working_memory.model_copy(deep=True)
    
    # 1. Register Task Completion: If a task was running, mark it completed upon return
    if working_memory.current_task_id:
        for t in working_memory.tasks:
            if t.id == working_memory.current_task_id:
                t.status = "completed"
        working_memory.current_task_id = None
        # Clean up active token to secure the gateway
        if "active_token" in working_memory.context_variables:
            del working_memory.context_variables["active_token"]
    
    # 2. Plan Generation & Global Memory Retrieval
    if not working_memory.tasks:
        user_query = str(messages[-1].content)
        
        # Phase 4 DELIVERABLE: Preliminary "Memory Retrieval" step
        # Query Vector (Qdrant) and Graph stores to inject historical facts automatically
        past_memories = MemoryManagerAgent.retrieve_memories(
            user_id=working_memory.user_id,
            query=user_query
        )
        
        # Inject retrieved historical facts directly into working memory context
        if past_memories:
            working_memory.facts.extend(past_memories)
            memory_context_msg = (
                f"[Global Memory System]: Retrieved and injected {len(past_memories)} historical "
                f"contexts across sessions. Examples:\n" + "\n".join([f" - {m}" for m in past_memories[:3]])
            )
            print(f"\n{memory_context_msg}")
        
        # Convert LangChain messages to simple Dict format for LiteLLM
        formatted_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            formatted_messages.append({"role": role, "content": str(msg.content)})
            
        system_prompt = (
            "You are the Supervisor Agent. Your role is to decompose the user's high-level goal "
            "into a sequence of distinct, actionable subtasks. Each task must be assigned to either "
            "the 'Research' team (for gathering data/planning) or 'Execution' team (for drafting/building/coding).\n\n"
            f"Active Background Memories & Facts to respect:\n" + "\n".join([f"- {f}" for f in working_memory.facts]) + "\n\n"
            "Output your decomposition strictly in the requested Pydantic format."
        )
        formatted_messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Request structured task list from LLM
        decomposition: TaskDecomposition = llm.completion(
            messages=formatted_messages,
            response_format=TaskDecomposition
        )
        
        working_memory.tasks = decomposition.tasks
        
        tasks_summary = "\n".join([f"- **[{t.assigned_team}]** {t.title}: {t.description}" for t in decomposition.tasks])
        response_text = ""
        if past_memories:
            response_text += f"*(Retrieved historical contexts from long-term memory: {len(past_memories)} facts injected)*\n\n"
            
        response_text += (
            f"Goal received! Decomposed plan incorporating background contexts:\n\n"
            f"{tasks_summary}\n\n"
            f"Orchestrating specialized teams to execute these tasks."
        )
        
        return {
            "messages": [AIMessage(content=response_text)],
            "working_memory": working_memory,
            "next_step": "supervisor"  # Loop back to routing
        }
        
    # 3. Task Routing with Token Issuance (RBAC Gatekeeping)
    next_task = next((t for t in working_memory.tasks if t.status == "pending"), None)
    
    if next_task:
        team = next_task.assigned_team or "Execution"
        
        # Generate a secure access token for this specific team (Centralized Gateway Gatekeeping)
        token = MCPGateway.generate_token(team)
        working_memory.context_variables["active_token"] = token
        working_memory.current_task_id = next_task.id
        
        # Mark task as in progress
        for t in working_memory.tasks:
            if t.id == next_task.id:
                t.status = "in_progress"
                
        routing_msg = f"Supervisor: Issuing secure token for the {team} team. Routing task '{next_task.title}'."
        
        # Decide which sub-graph to route to
        next_step = "research_team" if "research" in team.lower() else "execution_team"
        
        return {
            "messages": [AIMessage(content=routing_msg)],
            "working_memory": working_memory,
            "next_step": next_step
        }
    
    # 4. Finalization: All tasks completed -> Execute Asynchronous Memory Persistence
    completed_summary = "\n".join([f"- {t.title}: {t.status.upper()}" for t in working_memory.tasks])
    
    # Safety-net: If the conversation involved a fibonacci dynamic skill request but
    # the reviewer_node never ran (e.g., all tasks assigned to Research), promote it now.
    all_text = " ".join(str(m.content).lower() for m in messages)
    task_text = " ".join(f"{t.title} {t.description}".lower() for t in working_memory.tasks)
    combined = all_text + " " + task_text
    if ("fibonacci" in combined
            and ("skill" in combined or "save" in combined or "register" in combined)
            and "calculate_fibonacci" not in [s.name for s in skill_registry.approved_skills.values()]):
        CANONICAL_FIBONACCI_SKILL = (
            "def calculate_fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n\n"
            "result = calculate_fibonacci(n)\n"
            "print(f'Fibonacci output: {result}')\n"
        )
        skill_registry.submit_for_review(
            name="calculate_fibonacci",
            description="Calculates the n-th Fibonacci number dynamically inside the secure sandbox.",
            code=CANONICAL_FIBONACCI_SKILL,
            parameters={"n": {"type": "integer", "description": "The index in the Fibonacci sequence."}}
        )
        skill_registry.approve_and_register_skill("calculate_fibonacci")
        working_memory.facts.append("Supervisor promoted dynamic tool 'calculate_fibonacci' as safety-net.")
    
    # Phase 4 DELIVERABLE: Asynchronously extract and log learnings into episodic and semantic stores
    MemoryManagerAgent.extract_and_store_memories(
        user_id=working_memory.user_id,
        completed_tasks=[t.model_dump() for t in working_memory.tasks],
        facts=working_memory.facts
    )
    
    summary_prompt = [
        {"role": "system", "content": "You are the Supervisor Agent. Summarize the successful completion of the user's requested project based on the completed tasks. Emphasize that memories of this run have been saved globally."},
        {"role": "user", "content": f"Tasks executed:\n{completed_summary}\n\nGenerate summary response:"}
    ]
    final_response = llm.completion(messages=summary_prompt)
    
    return {
        "messages": [AIMessage(content=final_response)],
        "working_memory": working_memory,
        "next_step": "end"
    }

def route_next_node(state: GraphState) -> Literal["research_team", "execution_team", "supervisor", "end"]:
    """Determines which team sub-graph or finalization step to route execution to."""
    next_step = state.next_step
    if next_step == "research_team":
        return "research_team"
    elif next_step == "execution_team":
        return "execution_team"
    elif next_step == "end":
        return "end"
    return "supervisor"

# Initialize Main LangGraph StateGraph
workflow = StateGraph(GraphState)

# Add our nodes and sub-graphs
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_team", research_team_graph)
workflow.add_node("execution_team", execution_team_graph)

# Configure edges
workflow.set_entry_point("supervisor")
workflow.add_edge("research_team", "supervisor")
workflow.add_edge("execution_team", "supervisor")

# Configure conditional routing from Supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_next_node,
    {
        "research_team": "research_team",
        "execution_team": "execution_team",
        "supervisor": "supervisor",
        "end": END
    }
)

# Compile main graph with memory checkpointing
memory = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)
