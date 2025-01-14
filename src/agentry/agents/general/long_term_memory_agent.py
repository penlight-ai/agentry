
from agentry.models.chat import TokenUsage, ChatMessage
import typing
import asyncio
from agentry.agents.general.standard_agent import StandardAgent
from agentry.models.model_providers import OpenAiCompatibleApiConfig
from agentry.models.logging import LangfuseKeyInfo
from agentry.services.chat_service import ChatService
from langchain_core.messages import AIMessage

from pydantic import BaseModel
from agentry.utils.langgraph import LanggraphNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
import typing
from abc import ABC, abstractmethod

class MemoryData(BaseModel):
    id: str
    title: str
    description: str
    content: str
    is_active: bool = True


class Memory(ABC):
    @abstractmethod
    def get_data(self) -> typing.List[MemoryData]:
        pass

    @abstractmethod
    def update(self, data: MemoryData) -> MemoryData:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass


class MemoryManager(ABC):
    @abstractmethod
    def get_memories(self) -> typing.List[Memory]:
        pass

    @abstractmethod
    def add_memory(self, memory: MemoryData) -> None:
        pass


class CustomAgentState(BaseModel):
    messages: typing.List[BaseMessage]
    main_subagent_input_messages: typing.List[BaseMessage]
    main_subagent_output_messages: typing.List[BaseMessage]
    active_procedural_memories: typing.List[MemoryData]
    active_semantic_memories: typing.List[MemoryData]
    active_episodic_memories: typing.List[MemoryData]


class FeedContextNode(LanggraphNode):
    def run(self, state: CustomAgentState) -> CustomAgentState:
        state.messages.append(AIMessage(content="-- FeedContextNode"))
        print("FeedContextNode")
        return state


class GenerateAnswerNode(LanggraphNode):
    def run(self, state: CustomAgentState) -> CustomAgentState:
        state.messages.append(AIMessage(content="-- GenerateAnswerNode"))
        print("GenerateAnswerNode")
        return state


class SummarizeNode(LanggraphNode):
    def run(self, state: CustomAgentState) -> CustomAgentState:
        state.messages.append(AIMessage(content="-- SummarizeNode"))
        print("SummarizeNode")
        return state



class LongTermMemoryAgent(StandardAgent):
    def __init__(
        self,
        model: str,
        langfuse_key_info: LangfuseKeyInfo,
        api_config: OpenAiCompatibleApiConfig,
    ):
        self.model = model
        self.token_usage_for_last_reply = TokenUsage()
        self.langfuse_key_info = langfuse_key_info
        self.api_config = api_config

    def setup(self):
        super().setup()
        self.compiled_graph = self.build_langgraph_graph()

    
    @staticmethod
    def build_langgraph_graph() -> CompiledStateGraph:
        graph = StateGraph(CustomAgentState)
        feed_context_node = FeedContextNode()
        generate_answer_node = GenerateAnswerNode()
        summarize_node = SummarizeNode()

        graph.add_node(**feed_context_node.get_as_kwargs())
        graph.add_node(**generate_answer_node.get_as_kwargs())
        graph.add_node(**summarize_node.get_as_kwargs())

        graph.add_edge(START, feed_context_node.get_name())
        graph.add_edge(feed_context_node.get_name(), generate_answer_node.get_name())
        graph.add_edge(generate_answer_node.get_name(), summarize_node.get_name())
        graph.add_edge(summarize_node.get_name(), END)

        compiled_graph = graph.compile()
        print(compiled_graph.get_graph().draw_ascii())

        return compiled_graph


    def get_latest_reply_token_usage(self) -> typing.Optional[TokenUsage]:
        return TokenUsage(
            prompt_tokens=self.token_usage_for_last_reply.prompt_tokens,
            completion_tokens=self.token_usage_for_last_reply.completion_tokens,
        )

    def use_graph(self):
        initial_state = CustomAgentState(
            messages=[
                HumanMessage(content="initial message"),
            ],
            main_subagent_input_messages=[],
            main_subagent_output_messages=[],
            active_procedural_memories=[],
            active_semantic_memories=[],
            active_episodic_memories=[],
        )
        r = self.compiled_graph.invoke(initial_state)
        r_state = CustomAgentState(**r)
        for message in r_state.messages:
            message.pretty_print()

    async def reply(
        self, messages: typing.List[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        # Convert messages to the format required by OpenRouter
        self.token_usage_for_last_reply = TokenUsage()
        messages_in_langchain = [
            AIMessage(content=message.text_content) for message in messages
        ]

        # Create a new model instance for streaming
        chat_service = ChatService()
        chat_model = chat_service.make_typical_chat_model(
            model_name=self.model,
            langfuse_key_info=self.langfuse_key_info,
            api_config=self.api_config,
        )

        # Generate streaming response
        async for chunk in chat_model.model.astream(messages_in_langchain):
            # Update token usage from callback handler
            yield ChatMessage(text_content=str(chunk.content))
            self.token_usage_for_last_reply = chat_model.get_tokens()
            await asyncio.sleep(0.01)
        self.token_usage_for_last_reply = chat_model.get_tokens()
