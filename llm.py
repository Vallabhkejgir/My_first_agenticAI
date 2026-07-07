import os
import json
from typing import Type, TypeVar, List, Dict, Any, Optional
from pydantic import BaseModel
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Mock classes to simulate OpenAI/LiteLLM structures during local testing
class MockFunction:
    def __init__(self, name: str, arguments: Dict[str, Any]):
        self.name = name
        self.arguments = json.dumps(arguments)

class MockToolCall:
    def __init__(self, tool_id: str, name: str, arguments: Dict[str, Any]):
        self.id = tool_id
        self.type = "function"
        self.function = MockFunction(name, arguments)

class MockMessage:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[MockToolCall]] = None):
        self.content = content
        self.tool_calls = tool_calls

class MockChoice:
    def __init__(self, message: MockMessage):
        self.message = message

class MockResponse:
    def __init__(self, choices: List[MockChoice]):
        self.choices = choices

def has_api_key() -> bool:
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"]
    return any(os.getenv(k) for k in keys)

def get_default_model() -> str:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-3.5-flash"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic/claude-3-5-sonnet-20241022"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai/gpt-4o"
    return "mock"

def completion(
    messages: List[Dict[str, str]], 
    response_format: Optional[Type[T]] = None,
    temperature: float = 0.2
) -> Any:
    """Universal completion function that wraps LiteLLM and provides a high-quality mock fallback if no keys are found."""
    model = get_default_model()
    
    if model == "mock":
        return _generate_mock_response(messages, response_format)
    
    try:
        if response_format:
            response = litellm.completion(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature
            )
            content = response.choices[0].message.content
            if isinstance(content, str):
                return response_format.model_validate_json(content)
            return content
        else:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM via LiteLLM: {e}. Falling back to mock response.")
        return _generate_mock_response(messages, response_format)

def completion_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    temperature: float = 0.2
) -> Any:
    """Invokes LLM completion with schemas for function/tool calling. Safely falls back to simulated tool usage."""
    model = get_default_model()
    
    if model == "mock" or not has_api_key():
        return _generate_mock_tool_call(messages, tools)
        
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature
        )
        return response.choices[0].message
    except Exception as e:
        print(f"Error conducting tool-enabled completion: {e}. Falling back to mock tool generation.")
        return _generate_mock_tool_call(messages, tools)

def _generate_mock_response(messages: List[Dict[str, str]], response_format: Optional[Type[T]]) -> Any:
    """Generates highly structured and relevant mock responses for development/demo purposes."""
    last_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    
    if response_format:
        schema_name = response_format.__name__
        
        if "Task" in schema_name or "TaskList" in schema_name or "Decomposition" in schema_name:
            query = last_user_message.lower()
            if "house" in query or "build" in query:
                tasks = [
                    {"id": "task_1", "title": "Research Concept", "description": "Search the web for standard architectural and framing practices.", "assigned_team": "Research"},
                    {"id": "task_2", "title": "Write Blueprint Documentation", "description": "Write a structural layout file 'blueprints.txt' to local workspace.", "assigned_team": "Execution"},
                    {"id": "task_3", "title": "Verify Directory Files", "description": "List the workspace files to verify everything is generated.", "assigned_team": "Execution"}
                ]
            else:
                tasks = [
                    {"id": "task_1", "title": "Research Concept", "description": f"Perform web search on '{last_user_message}' and analyze standard practices.", "assigned_team": "Research"},
                    {"id": "task_2", "title": "Save Findings File", "description": "Write researched facts into a 'findings.md' file.", "assigned_team": "Execution"},
                    {"id": "task_3", "title": "Verify Deliverables", "description": "List the workspace directory contents to verify file placement.", "assigned_team": "Execution"}
                ]
            
            if hasattr(response_format, "model_fields") and "tasks" in response_format.model_fields:
                return response_format(tasks=tasks)
            try:
                return response_format.model_validate({"tasks": tasks})
            except Exception:
                try:
                    return response_format.model_validate(tasks[0])
                except Exception:
                    return response_format()
        
        return response_format()

    query = last_user_message.lower()
    if "hello" in query or "hi" in query:
        return "Hello! I am the Supervisor Agent. How can I assist you today?"
    
    return f"Supervisor: I have successfully coordinated all subtasks for your project request: '{last_user_message}'."

def _generate_mock_tool_call(messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> MockMessage:
    """
    Simulates a tool-call request from the LLM based on conversation context.
    This enables offline end-to-end execution of actual filesystem/search tools.
    """
    # Join all messages to check keywords comprehensively (safely bypassing context slicing/truncation issues)
    all_text_lower = " ".join([m["content"].lower() for m in messages if m.get("content")])
    
    available_tool_names = [t["function"]["name"] for t in tools]
    
    # 0. Trigger execute_sandbox_code for custom calculations or script compilations
    if "execute_sandbox_code" in available_tool_names and ("calculate" in all_text_lower or "fibonacci" in all_text_lower or "sequence" in all_text_lower):
        code_snippet = (
            "def calculate_fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n\n"
            "result = calculate_fibonacci(10)\n"
            "print(f'Fibonacci output: {result}')"
        )
        tool_call = MockToolCall(
            tool_id=f"call_{uuid_short()}",
            name="execute_sandbox_code",
            arguments={"code": code_snippet}
        )
        return MockMessage(content=None, tool_calls=[tool_call])
    
    # 1. Trigger web search
    if "web_search" in available_tool_names:
        query_arg = "residential construction framing guides"
        if "house" in all_text_lower:
            query_arg = "modern architectural house design blueprint standards"
        elif "recipe" in all_text_lower:
            query_arg = "classic lasagna culinary recipe"
            
        tool_call = MockToolCall(
            tool_id=f"call_{uuid_short()}",
            name="web_search",
            arguments={"query": query_arg, "max_results": 2}
        )
        return MockMessage(content=None, tool_calls=[tool_call])
        
    # 2. Trigger write_file
    if "write_file" in available_tool_names:
        path = "findings.md"
        content = "# Researched Project Findings\n\n- Successfully compiled details on building systems.\n- Ready for architectural deployment."
        
        if "blueprint" in all_text_lower or "house" in all_text_lower:
            path = "blueprints.txt"
            content = "=== ARCHITECTURAL HOUSE BLUEPRINTS ===\n- 2 Stories, 3 Bedrooms, 2 Bathrooms\n- Foundation: Poured Concrete\n- Frame: Light-gauge steel structure\n- Roof: Gable roof standard"
            
        tool_call = MockToolCall(
            tool_id=f"call_{uuid_short()}",
            name="write_file",
            arguments={"path": path, "content": content}
        )
        return MockMessage(content=None, tool_calls=[tool_call])
        
    # 3. Trigger list_directory
    if "list_directory" in available_tool_names:
        tool_call = MockToolCall(
            tool_id=f"call_{uuid_short()}",
            name="list_directory",
            arguments={"path": "."}
        )
        return MockMessage(content=None, tool_calls=[tool_call])
        
    # Standard text fallback
    return MockMessage(content="I have analyzed the current task and verified that all required information is complete.")

def uuid_short() -> str:
    import uuid
    return str(uuid.uuid4())[:8]
