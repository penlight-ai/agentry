from agentry.agents.general.long_term_memory_agent import LongTermMemoryAgent
from agentry.memory.memory import Memory, MemoryData, MemoryManager
from agentry.models.chat import ChatMessage
import typing
from langchain_core.callbacks import BaseCallbackHandler
from agentry.models.model_providers import OpenAiCompatibleApiConfig
from pathlib import Path

cline_preamble = """You are Cline, a highly skilled software engineer with extensive knowledge in many programming languages, frameworks, design patterns, and best practices"""


class ClineFileMemoryManager(MemoryManager):
    def __init__(self, memories_directory: Path):
        self.memories_directory = memories_directory
        self.memories = []

    def get_memory(self, id: str) -> typing.Optional[Memory]:
        memory_file_path = self.memories_directory / f"{id}.md"
        if not memory_file_path.exists():
            return None
        
        try:
            with open(memory_file_path, "r") as file:
                content = file.read()
                import re

                # Extract title, description, and content using regex
                title_match = re.search(r"# Title: (.*?)(?:\n|$)", content)
                desc_match = re.search(r"Description: (.*?)(?:\n|$)", content)
                content_match = re.search(r"---\n([\s\S]*)", content)

                # Check if all regex matches are found
                if not (title_match and desc_match and content_match):
                    return None

                # Extract data from matches
                title = title_match.group(1) if title_match else ""
                description = desc_match.group(1) if desc_match else ""
                content = content_match.group(1).strip() if content_match else ""

                return Memory(
                    data=MemoryData(
                        id=id,
                        title=title,
                        description=description,
                        content=content
                    )
                )
        except Exception:
            return None

    def create_memory(self, memory: Memory) -> Memory:
        # Ensure memories directory exists
        self.memories_directory.mkdir(parents=True, exist_ok=True)
        
        memory_data = memory.get_data()
        file_body = f"""# Title: {memory_data.title}
Description: {memory_data.description}
---
{memory_data.content}"""
        
        file_path = self.memories_directory / f"{memory_data.id}.md"
        with open(file_path, "w") as file:
            file.write(file_body)
        return memory

    def update_memory(self, memory: Memory) -> Memory:
        created_memory = self.create_memory(memory)
        return created_memory


class ClineBasedAgent(LongTermMemoryAgent):
    def __init__(
        self,
        name: str,
        model: str,
        api_config: OpenAiCompatibleApiConfig,
        callbacks: typing.Optional[typing.List[BaseCallbackHandler]] = None,
        memory_manager: typing.Optional[MemoryManager] = None,
    ):
        super().__init__(
            model=model,
            callbacks=callbacks,
            api_config=api_config,
            memory_manager=memory_manager,
        )
        self.name = name

    modified_cline_agent_signature = "END OF MODIFIED CLINE SYSTEM PROMPT"
    have_added_cline_system_prompt_to_procedural_memory = False

    def modify_cline_system_prompt(self, starting_system_prompt: str) -> str:
        return starting_system_prompt

    def is_cline_system_prompt(self, msg: ChatMessage) -> bool:
        return msg.text_content.startswith(cline_preamble) or msg.text_content.startswith('\n' + cline_preamble)
    
    async def reply(
        self, messages: typing.List[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        for msg in messages:
            if self.is_cline_system_prompt(msg):
                if not self.cline_system_prompt_is_already_modfied(msg):
                    msg.text_content = self.modify_cline_system_prompt(msg.text_content)
                if not self.have_added_cline_system_prompt_to_procedural_memory:
                    self.add_procedural_memories(
                        [
                            Memory(
                                data=MemoryData(
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

        messages_without_cline_system_prompt = [
            msg for msg in messages if not self.is_cline_system_prompt(msg)
        ]
        async for chunk in super().reply(messages_without_cline_system_prompt):
            yield chunk

    def cline_system_prompt_is_already_modfied(self, msg: ChatMessage) -> bool:
        return msg.text_content.endswith(self.modified_cline_agent_signature)
