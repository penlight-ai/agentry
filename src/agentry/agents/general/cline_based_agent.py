from agentry.agents.general.long_term_memory_agent import LongTermMemoryAgent
from agentry.memory.memory import SimpleMemory, MemoryData
from agentry.models.chat import ChatMessage
import typing
from agentry.models.logging import LangfuseKeyInfo
from agentry.models.model_providers import OpenAiCompatibleApiConfig

cline_preamble = """You are Cline, a highly skilled software engineer with extensive knowledge in many programming languages, frameworks, design patterns, and best practices"""


class ClineBasedAgent(LongTermMemoryAgent):
    def __init__(
        self,
        name: str,
        model: str,
        langfuse_key_info: LangfuseKeyInfo,
        api_config: OpenAiCompatibleApiConfig,
    ):
        super().__init__(
            model=model,
            langfuse_key_info=langfuse_key_info,
            api_config=api_config,
        )
        self.name = name

    modified_cline_agent_signature = "END OF MODIFIED CLINE SYSTEM PROMPT"
    have_added_cline_system_prompt_to_procedural_memory = False

    def modify_cline_system_prompt(self, starting_system_prompt: str) -> str:
        return starting_system_prompt

    async def reply(
        self, messages: typing.List[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        for msg in messages:
            if msg.text_content.startswith(cline_preamble):
                if not self.cline_system_prompt_is_already_modfied(msg):
                    msg.text_content = self.modify_cline_system_prompt(msg.text_content)
                if not self.have_added_cline_system_prompt_to_procedural_memory:
                    self.add_procedural_memories(
                        [
                            SimpleMemory(
                                memory_data=MemoryData(
                                    title="Cline System Prompt",
                                    description="Cline System Prompt",
                                    content=msg.text_content,
                                    order_factor=-1,
                                )
                            )
                        ]
                    )
                    self.have_added_cline_system_prompt_to_procedural_memory = True
                break

        async for chunk in super().reply(messages):
            yield chunk

    def cline_system_prompt_is_already_modfied(self, msg: ChatMessage) -> bool:
        return msg.text_content.endswith(self.modified_cline_agent_signature)
