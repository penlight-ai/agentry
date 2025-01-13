from .default_prompt_providers import DefaultPromptProviders


class ClineRelatedPrompts:
    @staticmethod
    def get_end_of_cline_prompt() -> str:
        return "=== END OF SYSTEM PROMPT ==="

    @staticmethod
    def get_identity_prompt(agent_name: str) -> str:
        return f"""### Identity:
You are a modified version of Cline named {agent_name}."""

    @staticmethod
    def get_multi_agent_awareness_prompt() -> str:
        return DefaultPromptProviders.Section(
            title="Multi-Agent Context",
            content="""You are part of a conversation where multiple agents are interacting with the user.
Each agent's message will begin with:
```
#### Agent "{agent_name}":
```
This will be added automatically to the start of each agent message. 
There is no need for you to include it as part of your response.""").get_as_text()

    @staticmethod
    def get_interactive_terminal_tmux_commands_prompt() -> str:
        return """## Interactive Terminal Commands:
When dealing with large files or interactive commands (like less, git log, etc.), you must use tmux to properly handle the interactive session. This allows you to read large files efficiently and send keystrokes to navigate through content.

Here's an example using the 'less' command to read a large file:

1. Create a tmux session and get initial output (use your name as the session name):
```bash
tmux new-session -d -s {your_name} -x 120 -y 60 "less large_file.txt" && sleep 0.5 && tmux capture-pane -t {your_name} -p
```

2. Send keystrokes and get updated output (e.g., navigating through the file):
```bash
tmux send-keys -t {your_name} "G" && sleep 0.1 && tmux capture-pane -t {your_name} -p  # Go to end of file
tmux send-keys -t {your_name} "g" && sleep 0.1 && tmux capture-pane -t {your_name} -p  # Go to start of file
tmux send-keys -t {your_name} "/pattern" Enter && sleep 0.1 && tmux capture-pane -t {your_name} -p  # Search for pattern
tmux send-keys -t {your_name} "n" && sleep 0.1 && tmux capture-pane -t {your_name} -p  # Go to next match
```

3. Always kill the session when finished:
```bash
tmux kill-session -t {your_name}

Note: If you face issues during the process, you can reset the session to start over.
This usually happens when you lose track of the keystrokes.
```"""

    @staticmethod
    def get_standard_prompt(
        starting_system_prompt: str,
        agent_name: str,
        additional_context: str = "",
    ) -> str:
        return f"""
{starting_system_prompt} 

## Additional/Extended Instructions:
{ClineRelatedPrompts.get_identity_prompt(agent_name=agent_name)}
{ClineRelatedPrompts.get_multi_agent_awareness_prompt()}
{additional_context}
"""
