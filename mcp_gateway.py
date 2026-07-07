import uuid
from typing import List, Dict, Any, Optional
from mcp_servers.filesystem import read_file, write_file, list_directory
from mcp_servers.web_search import web_search
from mcp_servers.sandbox import execute_sandbox_code
from skill_registry import skill_registry

# Explicitly define tool schemas for LLM routing (OpenAI compatible format)
TOOL_SCHEMAS = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the internet for information on a given query. Use this to find facts, latest news, or verify documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "The search query (e.g., 'latest python langgraph releases')"
                    },
                    "max_results": {
                        "type": "integer", 
                        "description": "Maximum number of search results to return (default 3, max 5).",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a specified file. Use this to examine system files or workspace configurations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string", 
                        "description": "The path to the file relative to the workspace root."
                    }
                },
                "required": ["path"]
            }
        }
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a specified file. Creates parent directories if missing. Use this to write scripts or documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string", 
                        "description": "The path where the file will be saved relative to the workspace root."
                    },
                    "content": {
                        "type": "string", 
                        "description": "The full string content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists the files and subdirectories inside a specified directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string", 
                        "description": "The folder path relative to the workspace root. Use '.' for the root directory.",
                        "default": "."
                    }
                }
            }
        }
    },
    "execute_sandbox_code": {
        "type": "function",
        "function": {
            "name": "execute_sandbox_code",
            "description": "Executes a block of Python code inside an isolated, secure ephemeral sandbox and returns stdout or error. Use this to run calculations, process complex files, or solve unmapped edge cases dynamically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string", 
                        "description": "The complete Python script string to execute. Example: 'import math; print(math.sqrt(16))'"
                    }
                },
                "required": ["code"]
            }
        }
    }
}

# Module-level registry to store active access tokens and map them to agent roles
ACTIVE_TOKENS: Dict[str, str] = {}

class MCPGateway:
    """
    Centralized MCP Gateway that manages primitive servers, integrates the Centralized Skill Registry,
    and enforces token-based Role-Based Access Control (RBAC).
    """
    
    @staticmethod
    def generate_token(role: str) -> str:
        """
        Generates a secure, short-lived access token for a specific agent role.
        """
        role_clean = (role or "").lower()
        token = f"mcp_token_{role_clean}_{str(uuid.uuid4())[:8]}"
        ACTIVE_TOKENS[token] = role_clean
        return token
    
    @staticmethod
    def get_role_from_token(token: str) -> Optional[str]:
        """
        Resolves the corresponding agent role mapped to an active access token.
        """
        return ACTIVE_TOKENS.get(token)

    @staticmethod
    def get_available_tools(agent_role: str) -> List[Dict[str, Any]]:
        """
        Retrieves schemas of authorized tools depending on the role (RBAC).
        Includes standard static primitives and dynamically registered dynamic skills.
        """
        role = (agent_role or "").lower()
        tools = []
        
        if "research" in role:
            tools.append(TOOL_SCHEMAS["web_search"])
        elif "execution" in role or "development" in role or "coder" in role:
            tools.extend([
                TOOL_SCHEMAS["read_file"],
                TOOL_SCHEMAS["write_file"],
                TOOL_SCHEMAS["list_directory"],
                TOOL_SCHEMAS["execute_sandbox_code"]  # Added E2B execution Sandbox
            ])
            # Load dynamically approved dynamic tools from the Skill Registry
            tools.extend(skill_registry.get_approved_tools_schemas())
            
        elif "supervisor" in role:
            # Supervisor role gets access to all schemas (static and dynamic) for oversight
            tools.extend(list(TOOL_SCHEMAS.values()))
            tools.extend(skill_registry.get_approved_tools_schemas())
            
        return tools

    @staticmethod
    def execute_tool(name: str, arguments: Dict[str, Any], token: str) -> str:
        """
        Executes a registered tool securely after validating the provided access token,
        checking role authorization, and routing static or dynamic execution accordingly.
        """
        role = ACTIVE_TOKENS.get(token)
        if not role:
            return f"Security Error: Invalid, expired, or missing access token for tool '{name}'."

        # Validate RBAC permissions
        allowed_tool_names = [t["function"]["name"] for t in MCPGateway.get_available_tools(role)]
        if name not in allowed_tool_names:
            return f"Security Error: Token role '{role}' is not authorized to execute the tool '{name}'."

        try:
            # 1. Handle Static Primitives
            if name == "web_search":
                query = arguments.get("query")
                max_results = arguments.get("max_results", 3)
                if not query:
                    return "Error: Missing required argument 'query'."
                return web_search(query=query, max_results=max_results)
                
            elif name == "read_file":
                path = arguments.get("path")
                if not path:
                    return "Error: Missing required argument 'path'."
                return read_file(path=path)
                
            elif name == "write_file":
                path = arguments.get("path")
                content = arguments.get("content")
                if not path or content is None:
                    return "Error: Missing required argument 'path' or 'content'."
                return write_file(path=path, content=content)
                
            elif name == "list_directory":
                path = arguments.get("path", ".")
                return list_directory(path=path)
                
            elif name == "execute_sandbox_code":
                code = arguments.get("code")
                if not code:
                    return "Error: Missing required argument 'code'."
                return execute_sandbox_code(code=code)
                
            # 2. Handle Globally Registered Dynamic Skills
            else:
                approved_dynamic_tools = [t["function"]["name"] for t in skill_registry.get_approved_tools_schemas()]
                if name in approved_dynamic_tools:
                    print(f"\n[MCP Gateway] Routing execution to Dynamic Skill Registry: '{name}'")
                    return skill_registry.execute_dynamic_skill(name, arguments)
                else:
                    return f"Error: Tool '{name}' is not registered in the Centralized MCP Gateway."
                
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"
