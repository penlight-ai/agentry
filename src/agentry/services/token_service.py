import logging
from typing import Dict, Any, List
from langchain.callbacks.base import BaseCallbackHandler
from aiser.agent.agent import TokenUsage

# Set debug level for detailed token counting logs
token_logger = logging.getLogger(__name__)
token_logger.setLevel(logging.DEBUG)

class TokenUsageCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for tracking token usage in both streaming and non-streaming modes.
    Designed to provide OpenAI-compatible token counting.
    """
    def __init__(self):
        # Initialize token counters following OpenAI's format
        # self.tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tokens = TokenUsage()
        super().__init__()

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """
        Called when LLM starts processing. Captures prompt tokens from LangChain.
        """
        token_logger.debug("Starting new request")
        token_logger.debug(f"LangChain serialized data: {serialized}")
        token_logger.debug(f"Prompts: {prompts}")
        token_logger.debug(f"Additional kwargs: {kwargs}")
        
        self.tokens = TokenUsage()
        
        # Log detailed information about the invocation parameters
        if "invocation_params" in kwargs:
            token_logger.debug(f"Invocation params structure: {kwargs['invocation_params']}")
            if isinstance(kwargs['invocation_params'], dict):
                for key, value in kwargs['invocation_params'].items():
                    token_logger.debug(f"Invocation param {key}: {value}")
        
        # Try to extract token information from all possible sources
        token_sources = {
            "kwargs_invocation_params": kwargs.get("invocation_params", {}).get("prompt_tokens"),
            "serialized_params": serialized.get("invocation_params", {}).get("prompt_tokens") if serialized else None,
            "token_usage": kwargs.get("invocation_params", {}).get("token_usage", {}).get("prompt_tokens"),
            "metadata": getattr(kwargs.get("metadata", {}), "prompt_tokens", None),
        }
        token_logger.debug(f"Available token sources: {token_sources}")
        
        # Use the first available token count
        for source, tokens in token_sources.items():
            if tokens is not None:
                self.tokens.prompt_tokens = tokens
                token_logger.debug(f"Using token count from {source}: {tokens}")
                break
        else:
            token_logger.debug("No token information found in standard sources")

    def on_llm_end(self, response, **kwargs):
        """
        Called when LLM completes processing. Updates token counts using LangChain's counts.
        Handles both streaming and non-streaming responses.
        """
        token_logger.debug("Request completed, finalizing counts")
        token_logger.debug(f"Response type: {type(response)}")
        token_logger.debug(f"Response full data: {response}")
        
        # For streaming responses, token info is in the generation's message
        if hasattr(response, "generations") and response.generations:
            generation = response.generations[0][0]  # Get first generation
            if hasattr(generation, "message") and hasattr(generation.message, "usage_metadata"):
                token_usage = generation.message.usage_metadata
                if token_usage:
                    token_logger.debug(f"Token usage from generation message metadata: {token_usage}")
                    self.tokens.prompt_tokens = int(token_usage.get("input_tokens", 0))
                    self.tokens.completion_tokens = int(token_usage.get("output_tokens", 0))
                    token_logger.debug(f"Updated token state: {self.tokens}")
        
        # For non-streaming responses, token info is in llm_output
        elif hasattr(response, "llm_output") and response.llm_output:
            token_logger.debug(f"LLM output content: {response.llm_output}")
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                token_logger.debug(f"Token usage from LLM output: {token_usage}")
                self.tokens.prompt_tokens = int(token_usage.get("prompt_tokens", 0))
                self.tokens.completion_tokens = int(token_usage.get("completion_tokens", 0))
        
        token_logger.debug(f"Current token state: {self.tokens}")
        token_logger.debug(f"Final token counts: {self.get_tokens()}")

    def on_llm_token(self, token: str, **kwargs):
        """
        Called for each token in streaming mode. Tracks completion tokens.
        """
        # Log detailed information about each token
        token_logger.debug(f"Token received - Raw token: {token}")
        token_logger.debug(f"Token kwargs: {kwargs}")
        
        # Track token counts
        self.tokens.completion_tokens += 1
        
        # Log progress with detailed token state
        if self.tokens.completion_tokens % 5 == 0:
            token_logger.debug(
                f"Streaming progress - Current token state: {self.tokens}\n"
                f"Token info - Current token: '{token}', "
                f"Token number: {self.tokens.completion_tokens}"
            )

    def get_tokens(self) -> TokenUsage:
        """
        Returns current token counts, ensuring all values are integers.
        """
        return self.tokens

