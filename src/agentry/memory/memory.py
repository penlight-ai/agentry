from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import typing
import os
import uuid

class MemoryData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    content: str
    is_active: bool = True
    order_factor: int = 0

    @staticmethod
    def make_empty() -> "MemoryData":
        return MemoryData(id="", title="", description="", content="")


class Memory(ABC):
    @abstractmethod
    def get_data(self) -> MemoryData:
        raise NotImplementedError

    @abstractmethod
    def update(self, data: MemoryData) -> MemoryData:
        raise NotImplementedError

    @abstractmethod
    def delete(self) -> None:
        raise NotImplementedError

class SimpleMemory(Memory):
    def __init__(self, memory_data: MemoryData):
        self.memory_data = memory_data

    def get_data(self) -> MemoryData:
        return self.memory_data 

    def update(self, data: MemoryData) -> MemoryData:
        self.memory_data = data
        return self.memory_data

    def delete(self) -> None:
        self.memory_data = MemoryData.make_empty()

# class FileMemory(Memory):
#     def __init__(self, file_path: str, memory_data: MemoryData):
#         self.file_path = file_path

#     def get_data(self) -> MemoryData:
#         with open(self.file_path, "r") as file:
#             return MemoryData()


class MemoryManager(ABC):
    @abstractmethod
    def get_memories(self) -> typing.Sequence[Memory]:
        pass

    @abstractmethod
    def add_memory(self, memory: MemoryData) -> None:
        pass
