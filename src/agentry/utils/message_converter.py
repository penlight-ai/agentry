from typing import List
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from agentry.models.chat_models import Message

def convert_messages(messages: List[Message]):
    """Convert API messages to LangChain message format"""
    message_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }

    converted_messages = []
    for msg in messages:
        if isinstance(msg.content, list):
            # Handle structured content (combine all text parts)
            content = " ".join(
                part["text"] for part in msg.content if part["type"] == "text"
            )
        else:
            content = msg.content

        message_class = message_map.get(msg.role)
        if message_class:
            converted_messages.append(message_class(content=content))

    return converted_messages
