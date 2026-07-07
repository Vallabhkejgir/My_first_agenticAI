import os
from mcp.server.fastmcp import FastMCP

# Instantiate a FastMCP server named "Filesystem"
filesystem_mcp = FastMCP("Filesystem")

WORKSPACE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def _safe_path(path: str) -> str:
    """Helper to resolve paths safely within the workspace directory."""
    # Prevent directory traversal attacks
    resolved = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not resolved.startswith(WORKSPACE_DIR):
        raise ValueError(f"Access denied: path '{path}' is outside the workspace.")
    return resolved

@filesystem_mcp.tool()
def read_file(path: str) -> str:
    """Reads the contents of a specified file. Use this to examine system files or workspace configurations."""
    try:
        safe_p = _safe_path(path)
        if not os.path.exists(safe_p):
            return f"Error: File '{path}' does not exist."
        if os.path.isdir(safe_p):
            return f"Error: '{path}' is a directory, not a file."
            
        with open(safe_p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@filesystem_mcp.tool()
def write_file(path: str, content: str) -> str:
    """Writes content to a specified file. Creates parent directories if missing. Use this to write scripts or documentation."""
    try:
        safe_p = _safe_path(path)
        os.makedirs(os.path.dirname(safe_p), exist_ok=True)
        
        with open(safe_p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Content written to '{path}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"

@filesystem_mcp.tool()
def list_directory(path: str = ".") -> str:
    """Lists the files and subdirectories inside a specified directory. Defaults to the current workspace root."""
    try:
        safe_p = _safe_path(path)
        if not os.path.exists(safe_p):
            return f"Error: Path '{path}' does not exist."
        if not os.path.isdir(safe_p):
            return f"Error: '{path}' is a file, not a directory."
            
        items = os.listdir(safe_p)
        if not items:
            return f"Directory '{path}' is empty."
            
        result = []
        for item in items:
            full_p = os.path.join(safe_p, item)
            type_str = "[DIR]" if os.path.isdir(full_p) else "[FILE]"
            result.append(f"{type_str} {item}")
            
        return "\n".join(result)
    except Exception as e:
        return f"Error listing directory: {str(e)}"
