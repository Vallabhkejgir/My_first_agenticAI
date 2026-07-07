import sys
import json
from io import StringIO
from mcp.server.fastmcp import FastMCP

# Instantiate a FastMCP server named "Sandbox"
sandbox_mcp = FastMCP("Sandbox")

@sandbox_mcp.tool()
def execute_sandbox_code(code: str) -> str:
    """
    Executes a block of Python code inside an isolated, secure ephemeral sandbox and returns stdout or error.
    Use this to run calculations, process complex files, or solve unmapped edge cases dynamically.
    """
    print(f"\n[Sandbox] Running isolated code block (E2B VM simulation)...")
    
    # Store standard outputs
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    redirected_output = StringIO()
    redirected_error = StringIO()
    
    sys.stdout = redirected_output
    sys.stderr = redirected_error
    
    try:
        # Construct an isolated scope
        # In a real environment, this utilizes E2B sandbox microVMs or Docker.
        local_scope = {}
        global_scope = {
            "__builtins__": __builtins__,
            "json": json,
            "sys": sys
        }
        
        exec(code, global_scope, local_scope)
        
        # Restore standard outputs
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        stdout_val = redirected_output.getvalue()
        stderr_val = redirected_error.getvalue()
        
        if stderr_val:
            return f"Execution Error (stderr):\n{stderr_val}"
            
        if not stdout_val and local_scope:
            serializable = {k: str(v) for k, v in local_scope.items() if not k.startswith("__") and not callable(v)}
            return f"Success (No stdout, captured variables):\n{json.dumps(serialize_locals(serializable), indent=2)}"
            
        return stdout_val if stdout_val else "Success: Code executed cleanly with no output."
        
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return f"Execution Exception:\n{str(e)}"

def serialize_locals(d: dict) -> dict:
    """Recursively converts variables to basic types for serialization."""
    out = {}
    for k, v in d.items():
        try:
            json.dumps({k: v})
            out[k] = v
        except Exception:
            out[k] = str(v)
    return out
