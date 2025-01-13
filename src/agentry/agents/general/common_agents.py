from src.config.config import AiModelProviderApiKeys, LangfuseApiInfos
from agentry.agents.general.pure_open_router_llm_agent import PureOpenRouterLlmAgent


class DefaultProjectOpenRouterLlmAgent(PureOpenRouterLlmAgent):
    def __init__(self, model_name: str):
        super().__init__(
            model=model_name,
            agent_id=model_name,
            langfuse_key_info=LangfuseApiInfos.default,
            api_config=AiModelProviderApiKeys.default_openrouter,
        )
