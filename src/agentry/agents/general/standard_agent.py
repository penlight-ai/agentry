from agentry.models.chat import ChatMessage, TokenUsage
import typing
from abc import ABC, abstractmethod

class StandardAgent(ABC):
    @abstractmethod
    def reply(self, messages: typing.List[ChatMessage]) -> typing.AsyncGenerator[ChatMessage, None]:
        raise NotImplementedError
    
    def get_latest_reply_token_usage(self) -> typing.Optional[TokenUsage]:
        """Optional method that agents can implement to provide token usage information
        for the most recent reply() call.
        
        This method should return the token usage that has accumulated since the start
        of the latest reply() method call.
        
        Returns:
            TokenUsage if the agent implements token counting, None otherwise.
        """
        return None

    def setup(self):
        pass
