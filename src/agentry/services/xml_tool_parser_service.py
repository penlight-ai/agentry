from typing import List, Optional
from agentry.models.tool_models import (
    AssistantMessageContent,
    TextContent,
    ToolUse,
    TOOL_NAMES,
    PARAM_NAMES,
)


class XMLToolParserService:
    def parse_tool_calls(self, message: str) -> List[ToolUse]:
        """Extract and parse all tool calls from an AI message.
        
        Args:
            message (str): The assistant's message containing XML tool calls
            
        Returns:
            List[ToolUse]: List of parsed tool calls
        """
        content_blocks = self.parse_message(message)
        return [block for block in content_blocks if isinstance(block, ToolUse)]

    def parse_message(self, message: str) -> List[AssistantMessageContent]:
        """Parse an assistant message into a list of content blocks (text and tool uses).
        
        Args:
            message (str): The assistant's message containing text and/or XML tool calls
            
        Returns:
            List[AssistantMessageContent]: List of parsed content blocks
        """
        content_blocks: List[AssistantMessageContent] = []
        current_text: Optional[TextContent] = None
        current_tool: Optional[ToolUse] = None
        current_param: Optional[str] = None
        
        # Track positions for extracting content
        text_start = 0
        tool_start = 0
        param_start = 0
        
        # Accumulate characters for pattern matching
        accumulator = ""
        
        for i, char in enumerate(message):
            accumulator += char
            
            # Handle parameter parsing within a tool
            if current_tool and current_param:
                param_value = accumulator[param_start:]
                param_close_tag = f"</{current_param}>"
                
                if param_value.endswith(param_close_tag):
                    # Extract parameter value without the closing tag
                    value = param_value[:-len(param_close_tag)].strip()
                    current_tool.parameters[current_param] = value
                    current_param = None
                continue
            
            # Handle tool parsing
            if current_tool:
                tool_content = accumulator[tool_start:]
                tool_close_tag = f"</{current_tool.name}>"
                
                if tool_content.endswith(tool_close_tag):
                    # Tool is complete
                    current_tool.partial = False
                    content_blocks.append(current_tool)
                    current_tool = None
                    # Reset accumulator to allow finding next tool
                    accumulator = ""
                    continue
                
                # Check for new parameter start
                param_match = self._find_param_start(tool_content)
                if param_match:
                    current_param = param_match
                    param_start = len(accumulator)
                
                # Handle content parameters
                if not current_param:
                    for param_name in ["custom_content", "content"]:
                        if param_name in current_tool.parameters:
                            self._handle_content_parameter(current_tool, tool_content, param_name)
                continue
            
            # Check for new tool start
            tool_match = self._find_tool_start(accumulator)
            if tool_match:
                # End current text block if exists
                if current_text:
                    current_text.partial = False
                    # Remove partial tool tag from text
                    text = current_text.custom_content[:-len(f"<{tool_match}>")].strip()
                    if text:  # Only append if there's actual content
                        current_text.custom_content = text
                        content_blocks.append(current_text)
                    current_text = None
                
                # Start new tool block
                current_tool = ToolUse(name=tool_match, parameters={})
                tool_start = len(accumulator) - len(f"<{tool_match}>")
                continue
            
            # Handle text content
            if current_text is None:
                text_start = i
            current_text = TextContent(
                custom_content=accumulator[text_start:].strip(),
                partial=True
            )
        
        # Handle incomplete blocks at end of message
        if current_tool:
            # Add any incomplete parameter
            if current_param:
                current_tool.parameters[current_param] = accumulator[param_start:].strip()
            content_blocks.append(current_tool)
        elif current_text and current_text.custom_content.strip():
            current_text.partial = False
            content_blocks.append(current_text)
        
        return content_blocks

    def _find_tool_start(self, text: str) -> Optional[str]:
        """Find if text ends with a valid tool opening tag.
        
        Args:
            text (str): The text to search in
            
        Returns:
            Optional[str]: The tool name if found, None otherwise
        """
        for tool_name in TOOL_NAMES:
            if text.endswith(f"<{tool_name}>"):
                return tool_name
        return None

    def _find_param_start(self, text: str) -> Optional[str]:
        """Find if text ends with a valid parameter opening tag.
        
        Args:
            text (str): The text to search in
            
        Returns:
            Optional[str]: The parameter name if found, None otherwise
        """
        for param_name in PARAM_NAMES:
            if text.endswith(f"<{param_name}>"):
                return param_name
        return None

    def _handle_content_parameter(self, tool: ToolUse, content: str, param_name: str) -> None:
        """Handle parameters that may contain XML-like content.
        
        Args:
            tool (ToolUse): The tool use being parsed
            content (str): The current tool content being parsed
            param_name (str): The name of the parameter to handle
        """
        if content.endswith(f"</{param_name}>"):
            start_tag = f"<{param_name}>"
            end_tag = f"</{param_name}>"
            content_start = content.find(start_tag) + len(start_tag)
            content_end = content.rfind(end_tag)
            
            if content_start != -1 and content_end != -1 and content_end > content_start:
                tool.parameters[param_name] = content[content_start:content_end].strip()
