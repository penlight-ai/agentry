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


class Memory:
    def __init__(self, data: MemoryData):
        self.data = data

    def get_data(self) -> MemoryData:
        return self.data 
    


class MemoryManager(ABC):
    @abstractmethod
    def get_memory(self, id: str) -> typing.Optional[Memory]:
        raise NotImplementedError

    @abstractmethod
    def create_memory(self, memory: Memory) -> Memory:
        raise NotImplementedError

    @abstractmethod
    def update_memory(self, memory: Memory) -> Memory:
        raise NotImplementedError

    def get_or_otherwise_create_memory(self, memory_to_create: Memory) -> Memory:
        fetched_memory = self.get_memory(memory_to_create.get_data().id)
        if fetched_memory:
            return fetched_memory
        return self.create_memory(memory_to_create)

class SimpleMemoryManager(MemoryManager):
    def __init__(self):
        self.memories: typing.Dict[str, Memory] = {}

    def get_memory(self, id: str) -> typing.Optional[Memory]:
        memory = self.memories.get(id)
        if memory is None:
            return None
        return memory

    def create_memory(self, memory: Memory) -> Memory:
        self.memories[memory.get_data().id] = memory
        return memory

    def update_memory(self, memory: Memory) -> Memory:
        self.memories[memory.get_data().id] = memory
        return memory
