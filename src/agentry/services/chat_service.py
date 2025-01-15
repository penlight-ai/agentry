import json
import time
import logging
import asyncio
from os import getenv
from typing import List, AsyncGenerator, Dict, Any
import typing
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from agentry.models.model_providers import OpenAiCompatibleApiConfig

from agentry.models.chat_models import Message
from ..utils.message_converter import convert_messages
from .token_service import TokenUsageCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler


# Set debug level for detailed streaming logs
stream_logger = logging.getLogger(__name__ + ".stream")
stream_logger.setLevel(logging.DEBUG)

from agentry.agents.general.standard_agent import TokenUsage
from langfuse.callback import CallbackHandler


def tokens_to_api_dict(token_usage: TokenUsage) -> Dict[str, int]:
    return {
        "prompt_tokens": token_usage.prompt_tokens,
        "completion_tokens": token_usage.completion_tokens,
        "total_tokens": token_usage.total_tokens,
    }


class FullChatModel:
    def __init__(
        self,
        chat_model: BaseChatModel,
        token_usage_handler: TokenUsageCallbackHandler,
        model_name: str,
    ):
        self.model = chat_model
        self.token_usage_handler = token_usage_handler
        self.model_name = model_name

    def get_tokens(self) -> TokenUsage:
        return self.token_usage_handler.get_tokens()


class ChatService:
    def make_typical_chat_model(
        self,
        model_name: str,
        api_config: OpenAiCompatibleApiConfig,
        callbacks: typing.Optional[typing.List[BaseCallbackHandler]] = None,
    ) -> FullChatModel:
        token_usage_handler = TokenUsageCallbackHandler()
        merged_callbacks = [
            token_usage_handler,
        ]
        if callbacks:
            merged_callbacks = merged_callbacks + callbacks
        chat_model = ChatOpenAI(
            api_key=SecretStr(api_config.api_key or ""),
            base_url=api_config.url_base,
            model=model_name,
            streaming=True,
            callbacks=merged_callbacks,
        )
        return FullChatModel(
            chat_model=chat_model,
            token_usage_handler=token_usage_handler,
            model_name=model_name,
        )

    async def generate_stream(
        self, messages: List[Message], chat_model: FullChatModel
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response in OpenAI format with accurate token counting.

        The streaming response follows OpenAI's format with:
        1. Initial chunk containing assistant role
        2. Content chunks for each piece of the response
        3. Final chunks with completion status and token usage
        """
        stream_logger.debug("Starting stream generation")
        converted_messages = convert_messages(messages)
        stream_logger.debug(f"Converted messages: {converted_messages}")

        try:
            stream_logger.debug("Creating new model instance with token counter")
            stream_model = chat_model
            chat_id = f"chatcmpl-{int(time.time())}"

            stream_logger.debug("Preparing initial assistant role chunk")
            initial_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chat_model.model_name,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            stream_logger.debug("Sending initial chunk")
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            stream_logger.debug("Starting content streaming")
            final_content = []
            async for chunk in stream_model.model.astream(converted_messages):
                # Save content for final token counting
                final_content.append(chunk.content)

                # Stream chunk in OpenAI format
                response_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": chat_model.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"
                await asyncio.sleep(0.01)  # Small delay to ensure proper streaming

            stream_logger.debug("Stream completed, finalizing token calculation")
            # Update final token counts from streaming
            final_tokens_as_dict = tokens_to_api_dict(chat_model.get_tokens())
            stream_logger.debug(f"Final token counts: {final_tokens_as_dict}")

            stream_logger.debug("Preparing completion status chunk")
            completion_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chat_model.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(completion_chunk)}\n\n"

            stream_logger.debug("Preparing token usage chunk")
            usage_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": chat_model.model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "usage": final_tokens_as_dict,
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            stream_logger.info("Stream ended successfully")

        except ValueError as e:
            stream_logger.error(f"Model validation error: {str(e)}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            stream_logger.error(f"Unexpected error: {str(e)}")
            error_chunk = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    async def create_chat_completion(
        self, messages: List[Message], chat_model: FullChatModel
    ) -> Dict[str, Any]:
        """Handle non-streaming response with token counting"""
        converted_messages = convert_messages(messages)

        # Create new model instance for clean token counting
        response = chat_model.model.invoke(converted_messages)

        token_usage = chat_model.token_usage_handler.get_tokens()
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_model.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": tokens_to_api_dict(token_usage),
        }
