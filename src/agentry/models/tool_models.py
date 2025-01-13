from typing import Dict, Any, Literal, Union
from pydantic import BaseModel, Field

class TextContent(BaseModel):
    """Model representing a text block in an assistant's message"""
    type: Literal["text"] = "text"
    custom_content: str
    partial: bool = False

class ToolUse(BaseModel):
    """Model representing a tool use block in an assistant's message"""
    type: Literal["tool_use"] = "tool_use"
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the tool call")
    partial: bool = False

    class Config:
        allow_population_by_field_name = True

AssistantMessageContent = Union[TextContent, ToolUse]

class ToolExecutionResult(BaseModel):   
    """Model representing the result of executing a tool"""
    success: bool
    result: str

# Predefined sets of valid tool and parameter names
# These could be expanded based on the available tools
TOOL_NAMES = {
    "read_file",
    "write_to_file",
    "execute_command",
    "replace_in_file",
    "search_files",
    "list_files",
    "list_code_definition_names",
    "use_mcp_tool",
    "access_mcp_resource",
    "ask_followup_question",
    "attempt_completion",
    # Test-specific tool names
    "custom_read_file",
    "sample_write_to_file"
}

PARAM_NAMES = {
    "path",
    "custom_content",  # Changed from content to custom_content
    "command",
    "requires_approval",
    "diff",
    "regex",
    "file_pattern",
    "recursive",
    "server_name",
    "tool_name",
    "arguments",
    "uri",
    "question",
    "result"
}
