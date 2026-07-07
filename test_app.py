import os
import sys
from fastapi.testclient import TestClient

# Import the FastAPI app, memory, and skill registry
try:
    from main import app
    from mcp_gateway import MCPGateway
    from skill_registry import skill_registry
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

# Initialize the TestClient
client = TestClient(app)

def test_workflow():
    print("==================================================")
    print(" RUNNING INTEGRATION & PHASE 5 DYNAMIC SKILL TESTS")
    print("==================================================")
    
    thread_id = "dynamic_sandbox_thread_abc"
    user_id = "developer_alice"
    session_id = "sandbox_session_1"
    
    # Clean up residual test files
    for f in ["blueprints.txt", "findings.md"]:
        if os.path.exists(f):
            os.remove(f)
            
    # --- TEST 1: Gateway Token-Based RBAC Boundary Verification ---
    print("\n[Test 1] Checking Sandbox & RBAC Boundaries...")
    research_token = MCPGateway.generate_token("Research")
    execution_token = MCPGateway.generate_token("Execution")
    
    # Verify Research team cannot access sandbox runner
    unauthorized_res = MCPGateway.execute_tool(
        "execute_sandbox_code", 
        {"code": "print('exploit')"}, 
        research_token
    )
    print(f" - Research Sandbox Rejection response: {unauthorized_res}")
    assert "Security Error" in unauthorized_res, "Research team should be blocked from running sandboxed code!"
    
    # Verify Execution team is allowed to run sandboxed code
    sandbox_res = MCPGateway.execute_tool(
        "execute_sandbox_code", 
        {"code": "print(2 + 2)"}, 
        execution_token
    )
    print(f" - Execution Sandbox Test response: {sandbox_res.strip()}")
    assert "4" in sandbox_res, "Execution team sandbox run failed."
    print("Gateway Sandbox RBAC Boundary: SECURED & VERIFIED")

    # --- TEST 2: Run Custom Calculator Task (Coder Sandbox VM execution) ---
    print("\n[Test 2] Sending computational request: 'calculate the 10th fibonacci and save as a skill.'")
    request_data = {
        "message": "Please calculate the 10th fibonacci sequence number and save it as a dynamic skill.",
        "thread_id": thread_id,
        "user_id": user_id,
        "session_id": session_id
    }
    
    # Send message, executing the entire LangGraph sub-graph swarm to completion
    response = client.post("/chat", json=request_data)
    assert response.status_code == 200
    data = response.json()
    
    print("\nSwarm Response summary:")
    for msg in data["messages"][-2:]:
        print(f" -> {msg['role'].upper()}: {msg['content'][:150]}...")
        
    print("\nSwarm Working Memory Facts captured during run:")
    for fact in data["working_memory"]["facts"]:
        print(f" - Fact: {fact}")
        
    # --- TEST 3: Skill Verification & Promotion Assertion ---
    print("\n[Test 3] Verifying Reviewer Approval Loop & Skill Promotion...")
    
    # Verify that the 'calculate_fibonacci' skill was approved and registered globally!
    approved_schemas = MCPGateway.get_available_tools("Execution")
    approved_names = [t["function"]["name"] for t in approved_schemas]
    print(f"Available Tools list for Execution team: {approved_names}")
    
    assert "calculate_fibonacci" in approved_names, "Dynamic skill 'calculate_fibonacci' was not promoted or registered globally!"
    print("Verification Loop and Skill Promotion: SUCCESS (Skill dynamically registered as permanent tool!)")

    # --- TEST 4: Executing the Brand-New Dynamic Skill ---
    print("\n[Test 4] Calling the newly registered Dynamic Skill via Gateway...")
    
    # We call the brand-new tool we just dynamically registered, calculating the 6-th Fibonacci number!
    # The gateway should route it to skill_registry.execute_dynamic_skill which injects n=6 and runs it in the sandbox!
    tool_arguments = {"n": 6}
    dynamic_skill_result = MCPGateway.execute_tool(
        "calculate_fibonacci",
        tool_arguments,
        execution_token
    )
    
    print(f"Result of calculate_fibonacci(n=6):\n---\n{dynamic_skill_result.strip()}\n---")
    
    # The 6th Fibonacci number is 8 (0, 1, 1, 2, 3, 5, 8)
    assert "8" in dynamic_skill_result, f"Expected 8, got output:\n{dynamic_skill_result}"
    print("Dynamic Skill Execution & secure argument binding: SUCCESS")

    # Clean up test files
    for f in ["blueprints.txt", "findings.md"]:
        if os.path.exists(f):
            os.remove(f)
    print("\nTest files cleaned up.")

    print("\n==================================================")
    print(" ALL PHASE 5 DYNAMIC SKILL TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")

if __name__ == "__main__":
    try:
        test_workflow()
    except AssertionError as e:
        print(f"\n❌ TEST FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
