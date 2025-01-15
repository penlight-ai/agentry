from pathlib import Path
from .prompt_providers import PromptProvider, MarkdownPromptProvider
from typing import Optional
import typing as t


class DefaultPromptProviders:
    class MarkdownTrustedContext(MarkdownPromptProvider):
        def __init__(
            self,
            file_path: Path,
            file_description: Optional[str] = None,
            include_file_path_in_context: bool = False,
        ):
            super().__init__(file_path=file_path)
            self.include_file_path_in_context = include_file_path_in_context
            self.file_description = file_description

        markdown_trusted_context_tag = "markdown_trusted_context"
        file_path_tag = "file_path"
        file_description_tag = "file_description"

        def get_as_text(self) -> str:
            raw_text = super().get_as_text()
            body_text = raw_text
            if self.include_file_path_in_context:
                file_path_text = f"""<{self.file_path_tag}>{str(self.file_path.absolute())}</{self.file_path_tag}>"""
                body_text = f"""{file_path_text}
{body_text}"""
            if self.file_description:
                file_description_text = f"""<{self.file_description_tag}>{self.file_description}</{self.file_description_tag}>"""
                body_text = f"""{file_description_text}
{body_text}"""
            r = f"""<{self.markdown_trusted_context_tag}>
{body_text}
</{self.markdown_trusted_context_tag}>"""
            return r

    class MarkdownTrustedContextWithFilePath(MarkdownTrustedContext):
        def __init__(self, file_path: Path, file_description: Optional[str] = None):
            super().__init__(
                file_path=file_path,
                file_description=file_description,
                include_file_path_in_context=True,
            )

    class AgentPromptTrustedContext(MarkdownTrustedContextWithFilePath):
        def __init__(self, file_path: Path):
            super().__init__(
                file_path=file_path,
                file_description=(
                    "This file contains agent-specific rules that should be followed "
                    "whenever this agent is called. These rules are distinct from "
                    "project-specific rules that apply to all agents working within "
                    "the project."
                )
            )

    class Section(PromptProvider):
        section_tag = "section"
        section_title_tag = "section_title"
        section_content_tag = "section_content"

        def __init__(self, title: str, content: str):
            self.title = title
            self.content = content

        def get_as_text(self) -> str:
            return f"""<{self.section_tag}>
<{self.section_title_tag}>{self.title}</{self.section_title_tag}>
<{self.section_content_tag}>
{self.content}
</{self.section_content_tag}>
</{self.section_tag}>"""

    class TrustedContextGroup(PromptProvider):
        trusted_context_group_tag = "trusted_context_group"

        def __init__(self, context_providers: t.Sequence[PromptProvider]):
            self.context_providers = context_providers

        def get_as_text(self) -> str:
            context_providers_text = "\n".join(
                [provider.get_as_text() for provider in self.context_providers]
            )
            return f"""<{self.trusted_context_group_tag}>
The following is a collection of trusted context entries ("trusted_context"s) of different types that 
you must carefully consider and follow.
Each entry contains either specific instructions, relevant background information, or guidelines that are 
essential for your tasks. These entries form your core knowledge base and operational framework.

Please process and adhere to each entry with equal importance:
{context_providers_text}
</{self.trusted_context_group_tag}>"""
