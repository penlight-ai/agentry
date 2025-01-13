from agentry.agents.general.pure_open_router_llm_agent import PureOpenRouterLlmAgent
from aiser.models import ChatMessage
import typing

cline_preamble = """You are Cline, a highly skilled software engineer with extensive knowledge in many programming languages, frameworks, design patterns, and best practices"""


class ClineBasedAgent(PureOpenRouterLlmAgent):
    modified_cline_agent_signature = "END OF MODIFIED CLINE SYSTEM PROMPT"
    def modify_cline_system_prompt(self, starting_system_prompt: str) -> str:
        return starting_system_prompt
    
    async def reply(
        self, messages: typing.List[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        for msg in messages:
            if msg.text_content.startswith(cline_preamble):
                if not self.cline_system_prompt_is_already_modfied(msg):
                    msg.text_content = self.modify_cline_system_prompt(msg.text_content)
                break;
            
        async for chunk in super().reply(messages):
            yield chunk

    def cline_system_prompt_is_already_modfied(self, msg: ChatMessage) -> bool:
        return msg.text_content.endswith(self.modified_cline_agent_signature)
