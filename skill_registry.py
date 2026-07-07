import json
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class DynamicSkill(BaseModel):
    name: str = Field(..., description="Unique alphanumeric identifier for the tool skill.")
    description: str = Field(..., description="Detailed description explaining when and how the LLM should invoke the skill.")
    code: str = Field(..., description="The executable Python script block that solves the edge case.")
    parameters_schema: Dict[str, Any] = Field(default_factory=dict, description="OpenAI-compatible dictionary of properties for arguments.")

class CentralizedSkillRegistry:
    """
    Centralized Database Pipeline for dynamic skill registration.
    Approved snippets are promoted to standard tool options.
    """
    def __init__(self):
        self.pending_skills: Dict[str, DynamicSkill] = {}
        self.approved_skills: Dict[str, DynamicSkill] = {}

    def submit_for_review(self, name: str, description: str, code: str, parameters: Dict[str, Any]) -> None:
        """Saves a newly compiled coding solution into the temporary pending registry."""
        skill = DynamicSkill(
            name=name,
            description=description,
            code=code,
            parameters_schema=parameters
        )
        self.pending_skills[name.lower().strip()] = skill
        print(f"\n[Skill Registry] Skill '{name}' submitted for review.")

    def approve_and_register_skill(self, name: str) -> bool:
        """Verifies, approves, and registers a pending skill globally."""
        key = name.lower().strip()
        if key not in self.pending_skills:
            print(f"[Skill Registry] Promotion failed: No pending skill named '{name}' found.")
            return False
            
        skill = self.pending_skills.pop(key)
        self.approved_skills[key] = skill
        print(f"[Skill Registry] Success! Skill '{skill.name}' approved and registered globally as a permanent tool.")
        return True

    def get_approved_tools_schemas(self) -> List[Dict[str, Any]]:
        """Converts approved dynamic skills into standard OpenAI-compatible function calling schemas."""
        schemas = []
        for skill in self.approved_skills.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": skill.name,
                    "description": skill.description,
                    "parameters": {
                        "type": "object",
                        "properties": skill.parameters_schema,
                        "required": list(skill.parameters_schema.keys())
                    }
                }
            })
        return schemas

    def execute_dynamic_skill(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Executes an approved dynamic skill securely in the sandbox environment.
        It binds the incoming arguments to global parameters and runs the script.
        """
        key = name.lower().strip()
        skill = self.approved_skills.get(key)
        if not skill:
            return f"Error: Dynamic skill '{name}' is not registered or approved."
            
        # Dynamically inject incoming arguments as python variables at the top of the execution code
        variable_bindings = []
        for arg_name, arg_val in arguments.items():
            if isinstance(arg_val, str):
                # Safely escape string values
                escaped = arg_val.replace('"', '\\"').replace('\n', '\\n')
                variable_bindings.append(f'{arg_name} = "{escaped}"')
            else:
                variable_bindings.append(f'{arg_name} = {json.dumps(arg_val)}')
                
        execution_preamble = "\n".join(variable_bindings) + "\n\n"
        complete_payload = execution_preamble + skill.code
        
        # Invoke sandbox runner
        from mcp_servers.sandbox import execute_sandbox_code
        return execute_sandbox_code(complete_payload)

# Singleton global registry representing global dynamic skill catalog
skill_registry = CentralizedSkillRegistry()
