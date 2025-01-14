from agentry.models.chat import TokenUsage, ChatMessage
import typing
import asyncio
from agentry.agents.general.standard_agent import StandardAgent
from agentry.models.model_providers import OpenAiCompatibleApiConfig
from agentry.models.logging import LangfuseKeyInfo
from agentry.services.chat_service import ChatService
from langchain_core.messages import AIMessage
from agentry.services.chat_service import FullChatModel

from pydantic import BaseModel
from agentry.utils.langgraph import LanggraphNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
import typing
from abc import ABC, abstractmethod
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI


class MemoryData(BaseModel):
    id: str
    title: str
    description: str
    content: str
    is_active: bool = True


class Memory(ABC):
    @abstractmethod
    def get_data(self) -> typing.Sequence[MemoryData]:
        pass

    @abstractmethod
    def update(self, data: MemoryData) -> MemoryData:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass


class MemoryManager(ABC):
    @abstractmethod
    def get_memories(self) -> typing.Sequence[Memory]:
        pass

    @abstractmethod
    def add_memory(self, memory: MemoryData) -> None:
        pass


class CustomAgentState(BaseModel):
    messages: typing.List[BaseMessage] = []
    main_subagent_input_messages: typing.List[BaseMessage] = []
    main_subagent_output_messages: typing.List[BaseMessage] = []
    active_procedural_memories: typing.List[MemoryData] = []
    active_semantic_memories: typing.List[MemoryData] = []
    active_episodic_memories: typing.List[MemoryData] = []


class FeedContextNode(LanggraphNode):
    def run(self, state: CustomAgentState) -> CustomAgentState:
        state.messages.append(AIMessage(content="-- FeedContextNode"))
        print("FeedContextNode")
        return state


class GenerateAnswerNode(LanggraphNode):
    def __init__(
        self,
        chat_model: FullChatModel,
    ):
        self.chat_model = chat_model

    def run(self, state: CustomAgentState, config: RunnableConfig) -> CustomAgentState:
        state.messages.append(AIMessage(content="-- GenerateAnswerNode"))
        print("GenerateAnswerNode")
        self.token_usage_for_last_reply = TokenUsage()
        messages_in_langchain = [
            AIMessage(content=message.content)
            for message in state.main_subagent_input_messages
        ]
        response_message = self.chat_model.model.invoke(
            messages_in_langchain, config=config
        )
        state.main_subagent_output_messages.append(response_message)
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
        # Create a new model instance for streaming
        chat_service = ChatService()
        self.chat_model = chat_service.make_typical_chat_model(
            model_name=self.model,
            langfuse_key_info=self.langfuse_key_info,
            api_config=self.api_config,
        )
        self.compiled_graph = self.build_langgraph_graph(chat_model=self.chat_model)

    def build_langgraph_graph(self, chat_model: FullChatModel) -> CompiledStateGraph:
        graph = StateGraph(CustomAgentState)
        feed_context_node = FeedContextNode()
        generate_answer_node = GenerateAnswerNode(
            chat_model=chat_model,
        )
        summarize_node = SummarizeNode()

        graph.add_node(**feed_context_node.get_as_kwargs())
        graph.add_node(
            node=generate_answer_node.get_name(), action=generate_answer_node.run
        )
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

    def get_graph(self) -> CompiledStateGraph:
        return self.compiled_graph

    async def use_graph(
        self, latest_messages: typing.Sequence[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        input_messages: typing.Sequence[BaseMessage] = [
            AIMessage(content=latest_message.text_content)
            for latest_message in latest_messages
        ]
        initial_state = CustomAgentState(
            messages=input_messages,
            main_subagent_input_messages=input_messages,
            main_subagent_output_messages=[],
            active_procedural_memories=[],
            active_semantic_memories=[],
            active_episodic_memories=[],
        )
        async for event in self.compiled_graph.astream_events(
            initial_state, version="v2"
        ):
            node_name = event["metadata"].get("langgraph_node", "")
            if (
                node_name != "GenerateAnswerNode"
                or event["event"] != "on_chat_model_stream"
            ):
                continue
            data = event["data"]
            content = data["chunk"].content
            yield ChatMessage(text_content=str(content))

    async def reply(
        self, messages: typing.Sequence[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        # Convert messages to the format required by OpenRouter
        self.token_usage_for_last_reply = TokenUsage()
        messages_in_langchain = [
            AIMessage(content=message.text_content) for message in messages
        ]

        async for msg in self.use_graph(latest_messages=messages):
            self.token_usage_for_last_reply = self.chat_model.get_tokens()
            yield msg
        self.token_usage_for_last_reply = self.chat_model.get_tokens()
