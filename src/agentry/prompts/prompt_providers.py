from abc import ABC, abstractmethod
from pathlib import Path


class PromptProvider(ABC):
    @abstractmethod
    def get_as_text(self) -> str:
        raise NotImplementedError("Subclass must implement get_as_text")


class MarkdownPromptProvider(PromptProvider):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def get_as_text(self) -> str:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Prompt file not found at: {self.file_path}")
        return self.file_path.read_text(encoding="utf-8")

