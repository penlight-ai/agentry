from agentry.agents.general.long_term_memory_agent import LongTermMemoryAgent
from agentry.models.logging import LangfuseKeyInfo
from agentry.models.model_providers import OpenAiCompatibleApiConfig
import os 
from dotenv import load_dotenv

load_dotenv()

openrouter_base_url = "https://openrouter.ai/api/v1"

def get_env_var(var_name: str) -> str:
    var = os.getenv(var_name)
    if var is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return var

open_router_api_config = OpenAiCompatibleApiConfig(
    url_base=openrouter_base_url,
    api_key=get_env_var("DEFAULT_OPENROUTER_API_KEY"),
)

agent = LongTermMemoryAgent(
    model="anthropic/claude-3.5-sonnet:beta",
    langfuse_key_info=LangfuseKeyInfo(
        public_key=get_env_var("PERSONAL_CATHY_LANGFUSE_PUBLIC_KEY"),
        secret_key=get_env_var("PERSONAL_CATHY_LANGFUSE_SECRET_KEY"),
    ),
    api_config=open_router_api_config,
)
agent.setup()
compiled_graph = agent.get_graph()