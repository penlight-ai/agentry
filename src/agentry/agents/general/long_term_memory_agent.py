from agentry.models.chat import TokenUsage, ChatMessage
import typing
import asyncio
from agentry.agents.general.standard_agent import StandardAgent
from agentry.models.model_providers import OpenAiCompatibleApiConfig
from agentry.services.chat_service import ChatService
from langchain_core.messages import AIMessage
from agentry.services.chat_service import FullChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage
from pydantic import BaseModel
from agentry.utils.langgraph import LanggraphNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from agentry.memory.memory import MemoryData, Memory
from langchain_core.messages import trim_messages
from langchain_core.callbacks import BaseCallbackHandler
from agentry.memory.memory import SimpleMemoryManager, MemoryManager
from agentry.utils.model_names import default_router_model_names, ModelNames


class CustomAgentState(BaseModel):
    messages_from_client: typing.Sequence[BaseMessage] = []
    main_subagent_input_messages: typing.Sequence[BaseMessage] = []
    main_subagent_output_messages: typing.Sequence[BaseMessage] = []
    procedural_memories: typing.Sequence[MemoryData] = []
    semantic_memories: typing.Sequence[MemoryData] = []
    episodic_memories: typing.Sequence[MemoryData] = []


class LanggraphNodeWithLifeCycle(LanggraphNode):
    def on_about_to_run_graph(self, state: CustomAgentState) -> CustomAgentState:
        return state


class FeedContextNode(LanggraphNodeWithLifeCycle):
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
        system_messages.append(
            SystemMessage(
                content=f"""
The current episodic memory (symmary of previous events and messages) is: 
{chr(10).join(f"{memory.content}" for memory in state.episodic_memories)}
"""
            )
        )
        trimmed_messages = trim_messages(
            state.messages_from_client,
            max_tokens=self.max_tokens,
            token_counter=ChatOpenAI(model="gpt-4o"),
        )
        state.main_subagent_input_messages = system_messages + list(trimmed_messages)
        return state


class GenerateAnswerNode(LanggraphNodeWithLifeCycle):
    def __init__(
        self,
        chat_model: FullChatModel,
    ):
        self.chat_model = chat_model

    def run(self, state: CustomAgentState, config: RunnableConfig) -> CustomAgentState:
        print("GenerateAnswerNode")
        # return state
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


class SummarizeNode(LanggraphNodeWithLifeCycle):
    def __init__(
        self,
        memory_manager: MemoryManager,
        summarization_model: str,
        api_config: OpenAiCompatibleApiConfig,
        max_tokens: int = 100000,
        max_bullet_points: int = 20,
        summarization_frequency: int = 3,
    ):
        super().__init__()
        self.memory_manager = memory_manager
        self.summarization_model = summarization_model
        self.max_tokens = max_tokens
        self.max_bullet_points = max_bullet_points
        self.api_config = api_config
        self.summarization_frequency = summarization_frequency
        # Create a chat service and get a chat model
        chat_service = ChatService()
        self.chat_model = chat_service.make_typical_chat_model(
            model_name=self.summarization_model,
            api_config=self.api_config,
            callbacks=None,
        )

    def on_about_to_run_graph(self, state: CustomAgentState) -> CustomAgentState:
        # load the episodic memory
        episodic_memory = self.memory_manager.get_or_otherwise_create_memory(
            memory_to_create=Memory(
                data=MemoryData(
                    id="episodic_memory",
                    title="Episodic Memory",
                    description="A memory that contains a running summary of the conversation and events that have occurred.",
                    content="",
                )
            )
        )
        state.episodic_memories = [episodic_memory.data]
        return state

    def _create_summarization_prompt(
        self, previous_summary: str, messages: typing.Sequence[BaseMessage]
    ) -> str:
        typescript_code = """```typescript
  const cachedData = await redis.get(key);
  if (!cachedData) {
    const validated = validateUserPrefs(data);
    await redis.set(key, JSON.stringify(validated), 'EX', 3600);
  }
  ```"""

        sql_code = """```sql
  -- Before: SELECT * FROM orders o LEFT JOIN products p ... LEFT JOIN users u ...
  -- After:  SELECT * FROM orders o LEFT JOIN users u ... LEFT JOIN products p ...
  ```"""

        bash_code = """```bash
  setfacl -m u:1000:rwx /host/path/to/data
  setfacl -m d:u:1000:rwx /host/path/to/data
  ```"""

        return f"""You are a highly skilled AI tasked with creating concise, insightful summaries of conversations. Your goal is to create a summary that preserves the key information and insights while being much shorter than the original conversation.

Guidelines for creating summaries:
1. Create bullet points that include:
   - Who sent the message (User/Agent Name)
   - The main action or request
   - Any key insights, observations, or important outcomes
2. Focus on capturing:
   - Task objectives and requirements
   - Important decisions and changes
   - Problems encountered and their solutions
   - Specific error messages or code blocks when they might be relevant later
3. Keep the total number of bullet points under {self.max_bullet_points}
4. When approaching the bullet point limit, combine older related points into more concise summaries
5. Include detailed information (error messages, code blocks, specific commands) when they provide valuable context
6. Output ONLY bullet points, no headers or sections

Here are examples showing different summarization approaches:

Example 1

Takeaways from this example:
Shows how to handle overlapping messages between previous summary and new messages. Notice that new messages often include content that was already summarized - this is normal and expected. The agent should recognize that messages like "We're running as non-root for security" appear both in the previous summary and new messages, as we're always processing a sliding window of recent messages. Technical details like error messages, commands, and code blocks are included when they might be useful later. The bullet points vary in length - some are concise while others contain detailed technical information when warranted.

Input:
Previous summary:
• User requested Docker container setup for production environment
• Agent suggested using multi-stage builds for smaller image size
• User agreed but mentioned need for specific file permissions
• User reported container failing to start
• Agent identified permission issue with volume mount

New messages:
[user]: We're running as non-root for security
[assistant]: Yes, that's why we're seeing the permission errors. Let me help fix that.
[user]: The container keeps failing to start with the same error
[assistant]: Let me check the logs... I see a permission error when trying to write to /data/app.log
[user]: Yes, we're running as non-root for security
[assistant]: We'll need to adjust the user permissions. Here's how...

Expected Ideal Output:
• User requested Docker container setup for production environment
• Agent suggested using multi-stage builds for smaller image size
• User agreed but mentioned need for specific file permissions
• User reported container failing to start
• Agent identified permission issue in volume mount: "Error: EACCES: permission denied, open '/data/app.log'"
• User confirmed running container as non-root user
• Agent provided solution with specific command: `docker run --user $(id -u):$(id -g) ...`, fixing permission mismatch
• User successfully deployed with new command and requested security hardening
• Agent suggested implementing volume ACLs: {bash_code}

Example 2

Takeaways from this example:
Shows how to condense older messages when approaching the bullet point limit. Notice how multiple related technical points are combined into a single, comprehensive bullet point. The condensed version maintains the key technical decisions and implementation details while reducing the total number of points. The most recent points remain detailed as they represent current context.

Input:
Previous summary:
• User requested new REST API endpoint for user preferences
• Agent suggested using Express.js with TypeScript
• User approved but requested OpenAPI documentation
• Agent implemented initial endpoint with swagger-jsdoc
• User reported validation issues with nested objects
• Agent fixed validation using Joi schema

New messages:
[user]: The validation with Joi schema is working well
[assistant]: Great to hear! Would you like to add more validation rules?
[user]: Can we add caching to improve performance?
[assistant]: Yes, let's implement Redis caching
[user]: How will this affect the existing validation?
[assistant]: We'll cache post-validation. Here's the implementation...

Expected Ideal Output:
• User and Agent collaborated on REST API endpoint implementation, covering TypeScript, OpenAPI docs, and Joi validation
• User confirmed successful validation implementation with Joi schema
• User requested Redis caching for performance optimization
• Agent implemented caching layer after validation:
{typescript_code}

Example 3

Takeaways from this example:
Shows how to handle many messages by combining related points while preserving specific technical details that might be valuable later. Notice how the query optimization process, which involved many back-and-forth messages, is condensed into clear, actionable points. Performance metrics and specific SQL changes are preserved as they provide valuable context for future optimizations.

Input:
Previous summary:
• Team working on database performance optimization
• Initial analysis showed missing indexes on several tables
• Added indexes on customer_id and order_date columns
• Found missing index on user_id
• Added index, reducing query time to 5s

New messages:
[user]: The index on user_id helped a lot
[assistant]: Yes, the query is much faster now, but I see we can optimize it further
[user]: Can you help optimize this SQL query more?
[assistant]: Let's analyze the query structure first...
[user]: It's still taking about 5 seconds to run
[assistant]: I found the issue - the join order is suboptimal
[user]: Can you show the before and after?
[assistant]: Here's the comparison...

Expected Ideal Output:
• Team working on database performance optimization
• Added indexes on customer_id, order_date, and user_id columns
• User confirmed significant performance improvement from user_id index
• Team discovered and fixed suboptimal join order: "Changed LEFT JOIN users before products table, leveraging new index"
• Query now executes in 200ms with specific improvements:
{sql_code}

Notes: 
- In the examples above, the user and agent messages are short. In the real world, they will be longer.
---- End of Examples ----

Previous Summary:
{previous_summary}

Current Messages to Summarize:

--- START of messages to summarize ---

{chr(10).join(f"[{msg.type}]: {msg.content}" for msg in messages)}

--- END of messages to summarize ---

As a reminder, here is the previous summary:
{previous_summary}

Create a bullet-point summary following the guidelines above. Remember:
1. Include the previous summary's points, condensing older ones if needed
2. Stay under {self.max_bullet_points} total bullet points
3. Include specific details (errors, code, commands) when they might be valuable later
4. Output ONLY bullet points with no headers or sections
5. Make sure to keep the temportal and logical consistency of the summary. Meaning, the order of the events has to be correct.
If in doubt, just keep the previous summary and append some new bullet points to it. It's not necessary to rewrite the entire summary everytime.
6. The summary should change slowly with time. The previously summary should mostly be kept the same unless there a need to summarize 
because we have reached the total bullet point limit. Other notable exceptions include updating the previous summary because a new detail
has become more important or relevant according to the latest messages. 
7. Your end goal is to create a summary that is a good representation of the conversation and events that have occurred.
You will be called every few messages.
"""

    def run(self, state: CustomAgentState) -> CustomAgentState:
        print("SummarizeNode")
        # return state
        if len(state.messages_from_client) % self.summarization_frequency != 0:
            return state

        # Trim messages to avoid token overflow
        all_messages = list(state.messages_from_client)
        if state.main_subagent_output_messages:
            all_messages.extend(state.main_subagent_output_messages)

        trimmed_messages = trim_messages(
            all_messages,
            max_tokens=self.max_tokens,
            token_counter=ChatOpenAI(model="gpt-4o"),
        )

        # episodic_memory = self.memory_manager.get_or_otherwise_create_memory(
        #     memory_to_create=Memory(
        #         data=MemoryData(
        #             id="episodic_memory",
        #             title="Episodic Memory",
        #             description="A memory that contains a running summary of the conversation and events that have occurred.",
        #             content="",
        #         )
        #     )
        # )
        episodic_memory = Memory(state.episodic_memories[0])

        state.episodic_memories = [episodic_memory.data]
        # Create the prompt with previous summary and trimmed messages
        prompt = self._create_summarization_prompt(
            previous_summary=(
                state.episodic_memories[0].content
                if state.episodic_memories
                else "No previous summary available"
            ),
            messages=trimmed_messages,
        )

        # Prepare the conversation history for summarization
        messages_to_summarize = [SystemMessage(content=prompt)]

        # Get the summary
        summary_message = self.chat_model.model.invoke(messages_to_summarize)
        episodic_memory.data.content = str(summary_message.content)
        self.memory_manager.update_memory(episodic_memory)

        # Store the summary in the state
        # The invoke method returns an AIMessage, so we need to get its content
        # and ensure it's a string
        if isinstance(summary_message, AIMessage):
            summary_content = summary_message.content
        else:
            summary_content = str(summary_message)

        # Convert any complex content to string
        if isinstance(summary_content, (list, dict)):
            summary_content = str(summary_content)

        state.episodic_memories = [episodic_memory.data]

        return state


class LongTermMemoryAgent(StandardAgent):
    def __init__(
        self,
        model: str,
        api_config: OpenAiCompatibleApiConfig,
        summarization_model: typing.Optional[str] = None,
        memory_manager: typing.Optional[MemoryManager] = None,
        callbacks: typing.Optional[typing.List[BaseCallbackHandler]] = None,
        router_model_names: typing.Optional[ModelNames] = None,
    ):
        self.model = model
        self.router_model_names = router_model_names or default_router_model_names
        self.summarization_model = (
            summarization_model
            or self.router_model_names.anthropic_claude_3_5_sonnet_beta
        )
        self.token_usage_for_last_reply = TokenUsage()
        self.callbacks = callbacks
        self.api_config = api_config
        # This thread_id is hardcoded temporarily. In this future, threads should at least
        # be separated by user and probably also by user session.
        self.thread_id = "1"
        self.pre_reply_state = CustomAgentState()
        self.procedural_memories: typing.List[Memory] = []
        self.semantic_memories: typing.List[Memory] = []
        if memory_manager:
            self.memory_manager = memory_manager
        else:
            self.memory_manager = SimpleMemoryManager()

    def setup(self):
        super().setup()
        # Create a new model instance for streaming
        chat_service = ChatService()
        self.chat_model = chat_service.make_typical_chat_model(
            model_name=self.model,
            api_config=self.api_config,
            callbacks=self.callbacks,
        )
        self.feed_context_node = FeedContextNode()
        self.generate_answer_node = GenerateAnswerNode(
            chat_model=self.chat_model,
        )
        self.summarize_node = SummarizeNode(
            memory_manager=self.memory_manager,
            summarization_model=self.summarization_model,
            api_config=self.api_config,
        )
        self.compiled_graph = self.build_langgraph_graph(chat_model=self.chat_model)
        self.graph_nodes = [
            self.feed_context_node,
            self.generate_answer_node,
            self.summarize_node,
        ]

    def _make_graph_config(self) -> RunnableConfig:
        callbacks: typing.List[BaseCallbackHandler] = []
        if self.callbacks:
            callbacks.extend(self.callbacks)
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

        graph.add_node(**self.feed_context_node.get_as_kwargs())
        graph.add_node(
            node=self.generate_answer_node.get_name(),
            action=self.generate_answer_node.run,
        )
        graph.add_node(**self.summarize_node.get_as_kwargs())

        graph.add_edge(START, self.feed_context_node.get_name())
        graph.add_edge(
            self.feed_context_node.get_name(), self.generate_answer_node.get_name()
        )
        graph.add_edge(
            self.generate_answer_node.get_name(), self.summarize_node.get_name()
        )
        graph.add_edge(self.summarize_node.get_name(), END)

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
        for node in self.graph_nodes:
            node.on_about_to_run_graph(pre_reply_state)

        async for event in self.compiled_graph.astream_events(
            pre_reply_state, config=self._make_graph_config(), version="v2"
        ):
            node_name = event.get("metadata", {}).get("langgraph_node", "")
            if (
                node_name != self.generate_answer_node.get_name()
                or event["event"] != "on_chat_model_stream"
            ):
                continue
            data = event["data"]
            chunk = data.get("chunk")
            if isinstance(chunk, BaseMessage):
                content = chunk.content
            else:
                content = ""
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
