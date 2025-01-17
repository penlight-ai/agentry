from pydantic import BaseModel

class ModelNames(BaseModel):
    openai_gpt_4o_mini: str
    anthropic_claude_3_5_sonnet_beta: str
    google_gemini_2_0_flash_thinking_exp_free: str
    google_gemini_flash_1_5: str
    deepseek_deepseek_v3_chat: str


open_router_model_names = ModelNames(
    openai_gpt_4o_mini="openai/gpt-4o-mini",
    anthropic_claude_3_5_sonnet_beta="anthropic/claude-3.5-sonnet:beta",
    google_gemini_2_0_flash_thinking_exp_free="google/gemini-2.0-flash-thinking-exp:free",
    google_gemini_flash_1_5="google/gemini-flash-1.5",
    deepseek_deepseek_v3_chat="deepseek/deepseek-chat",
)
    
default_router_model_names = open_router_model_names