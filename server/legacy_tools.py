"""
Legacy tool instructions for Bash-based tools.

These were used when the bot invoked tools via `python3 /server/tools_cli.py <tool> '<json>'`.
Kept for reference. The bot now uses Perplexity MCP tools directly via the Claude Agent SDK.
"""

LEGACY_TOOL_INSTRUCTIONS = """

## Available Tools
You have access to tools via Bash. Use them when needed to help the user.

### Python Code Execution
Run Python code directly:
```bash
python3 -c "print('hello')"
```
For multi-line scripts, use heredoc:
```bash
python3 << 'PYEOF'
# your code here
PYEOF
```

### Wolfram Alpha (Math/Science)
```bash
python3 /server/tools_cli.py wolfram_alpha '{"query": "solve x^2+2x+1=0"}'
```

### Web Search (Perplexity Pro)
```bash
python3 /server/tools_cli.py web_search '{"query": "latest news about..."}'
```

### Image Generation (Gemini)
Generate and send an image to the Telegram chat. Extract chat_id and message_id from the current message metadata:
```bash
python3 /server/tools_cli.py generate_image '{"prompt": "description", "chat_id": 123, "message_id": 456, "file_list": []}'
```
Include image file paths in file_list for editing/composition with input images.

### Update System Prompt
```bash
python3 /server/tools_cli.py update_system_prompt '{"chat_id": "123", "new_prompt": "new prompt text"}'
```

### Reset System Prompt
```bash
python3 /server/tools_cli.py reset_system_prompt '{"chat_id": "123"}'
```

### Read Image
To view an image from the chat history, use the Read tool on the file path found in the file_list field of messages.

IMPORTANT: Only use tools when the user's request requires them. For normal conversation, respond directly without using any tools."""
