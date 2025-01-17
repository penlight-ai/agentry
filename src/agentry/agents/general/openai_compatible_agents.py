from agentry.models.chat import TokenUsage, ChatMessage
import typing
import asyncio
from agentry.agents.general.standard_agent import StandardAgent
from agentry.models.model_providers import OpenAiCompatibleApiConfig
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from agentry.services.chat_service import ChatService


class OpenAiCompatibleTokenCalculatingAgent(StandardAgent):
    def __init__(
        self,
        model: str,
        api_config: OpenAiCompatibleApiConfig,
        callbacks: typing.Optional[typing.List[BaseCallbackHandler]] = None,
    ):
        self.model = model
        self.token_usage_for_last_reply = TokenUsage()
        self.callbacks = callbacks
        self.api_config = api_config

    def get_latest_reply_token_usage(self) -> typing.Optional[TokenUsage]:
        return self.token_usage_for_last_reply

    async def reply(
        self, messages: typing.List[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        # Convert messages to the format required by OpenRouter
        self.token_usage_for_last_reply = TokenUsage()
        messages_in_langchain = [
            AIMessage(content=message.text_content) for message in messages
        ]

        # Create a new model instance for streaming
        chat_service = ChatService()
        chat_model = chat_service.make_typical_chat_model(
            model_name=self.model,
            callbacks=self.callbacks,
            api_config=self.api_config,
        )

        # Generate streaming response
        async for chunk in chat_model.model.astream(messages_in_langchain):
            # Update token usage from callback handler
            yield ChatMessage(text_content=str(chunk.content))
            self.token_usage_for_last_reply = chat_model.get_tokens()
            await asyncio.sleep(0.01)
        self.token_usage_for_last_reply = chat_model.get_tokens()
