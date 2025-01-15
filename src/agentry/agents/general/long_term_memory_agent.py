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
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from agentry.memory.memory import MemoryData, Memory
from langchain_core.messages import trim_messages
from langchain_core.callbacks import BaseCallbackHandler


class CustomAgentState(BaseModel):
    messages_from_client: typing.Sequence[BaseMessage] = []
    main_subagent_input_messages: typing.Sequence[BaseMessage] = []
    main_subagent_output_messages: typing.Sequence[BaseMessage] = []
    procedural_memories: typing.Sequence[MemoryData] = []
    semantic_memories: typing.Sequence[MemoryData] = []
    episodic_memories: typing.Sequence[MemoryData] = []


class FeedContextNode(LanggraphNode):
    def __init__(self, max_tokens: int = 30000) -> None:
        super().__init__()
        self.max_tokens = max_tokens

    def _procedural_memory_to_system_message(self, memory: MemoryData) -> SystemMessage:
        return SystemMessage(
            content=f"Procedural Memory: {memory.title}\n{memory.description}\n{memory.content}"
        )

    def run(self, state: CustomAgentState) -> CustomAgentState:
        print("FeedContextNode")
        sorted_memories = sorted(
            state.procedural_memories, key=lambda x: x.order_factor
        )
        system_messages = [
            self._procedural_memory_to_system_message(memory)
            for memory in sorted_memories
        ]
        trimmed_messages = trim_messages(
            state.messages_from_client,
            max_tokens=self.max_tokens,
            token_counter=ChatOpenAI(model="gpt-4o"),
        )
        state.main_subagent_input_messages = system_messages + list(trimmed_messages)
        return state


class GenerateAnswerNode(LanggraphNode):
    def __init__(
        self,
        chat_model: FullChatModel,
    ):
        self.chat_model = chat_model

    def run(self, state: CustomAgentState, config: RunnableConfig) -> CustomAgentState:
        print("GenerateAnswerNode")
        self.token_usage_for_last_reply = TokenUsage()
        messages_in_langchain = [
            AIMessage(content=message.content)
            for message in state.main_subagent_input_messages
        ]
        response_message = self.chat_model.model.invoke(
            messages_in_langchain, config=config
        )
        state.main_subagent_output_messages = list(
            state.main_subagent_output_messages
        ) + [response_message]
        return state


class SummarizeNode(LanggraphNode):
    def run(self, state: CustomAgentState) -> CustomAgentState:
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
        # This thread_id is hardcoded temporarily. In this future, threads should at least
        # be separated by user and probably also by user session.
        self.thread_id = "1"
        self.pre_reply_state = CustomAgentState()
        self.procedural_memories: typing.List[Memory] = []
        self.semantic_memories: typing.List[Memory] = []

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

    def _make_graph_config(self) -> RunnableConfig:
        callbacks: typing.List[BaseCallbackHandler] = []
        if self.chat_model.tracing_handler:
            callbacks.append(self.chat_model.tracing_handler)
        if self.chat_model.token_usage_handler:
            callbacks.append(self.chat_model.token_usage_handler)
        return {
            "configurable": {
                "thread_id": self.thread_id,
            },
            "callbacks": callbacks,
        }

    def _check_is_called_after_setup(self, method_name: str):
        if not self.compiled_graph:
            raise Exception(f"{method_name} should only be called after setup.")

    def get_state(self):
        self._check_is_called_after_setup(method_name=self.get_state.__name__)
        return self.compiled_graph.get_state(
            config=self._make_graph_config(), subgraphs=True
        )

    # def update_pre_reply_state(self, state: CustomAgentState):
    #     return state

    def add_procedural_memories(self, memories: typing.List[Memory]):
        self.procedural_memories = self.procedural_memories + memories

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

        compiled_graph = graph.compile(checkpointer=MemorySaver())
        print(compiled_graph.get_graph().draw_ascii())

        return compiled_graph

    def get_latest_reply_token_usage(self) -> typing.Optional[TokenUsage]:
        return TokenUsage(
            prompt_tokens=self.token_usage_for_last_reply.prompt_tokens,
            completion_tokens=self.token_usage_for_last_reply.completion_tokens,
        )

    def get_graph(self) -> CompiledStateGraph:
        return self.compiled_graph

    def _get_pre_reply_state(
        self, input_messages: typing.Sequence[BaseMessage]
    ) -> CustomAgentState:
        procedural_memory_data_list: typing.List[MemoryData] = [
            memory.get_data() for memory in self.procedural_memories
        ]
        semantic_memory_data_list: typing.List[MemoryData] = [
            memory.get_data() for memory in self.semantic_memories
        ]
        return CustomAgentState(
            messages_from_client=input_messages,
            main_subagent_input_messages=input_messages,
            procedural_memories=procedural_memory_data_list,
            semantic_memories=semantic_memory_data_list,
        )

    async def use_graph(
        self, latest_messages: typing.Sequence[ChatMessage]
    ) -> typing.AsyncGenerator[ChatMessage, None]:
        input_messages: typing.Sequence[BaseMessage] = [
            AIMessage(content=latest_message.text_content)
            for latest_message in latest_messages
        ]
        pre_reply_state = self._get_pre_reply_state(input_messages)
        async for event in self.compiled_graph.astream_events(
            pre_reply_state, config=self._make_graph_config(), version="v2"
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
