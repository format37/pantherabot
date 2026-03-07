# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pantherabot is a conversational Telegram bot named "Janet" powered by Claude (via `claude_agent_sdk`) with tool access. It's a FastAPI-based server that integrates with Telegram via a separate telegram bot server, providing conversational AI capabilities with tool calling, image generation/understanding, web search, and file handling.

The bot is part of a three-component architecture:
- Telebot server (separate repo: github.com/format37/telegram_bot)
- Panthera bot (this repo)
- LLM service: Claude subscription via `claude_agent_sdk` (flat-rate, not per-token)

## Development Commands

```bash
# Full rebuild and deploy (sources .env, rebuilds container)
./compose.sh

# Build and start
docker-compose up --build -d

# View logs
./logs.sh  # runs: sudo docker logs -f panthera

# Restart / stop
docker-compose restart
docker-compose down

# Health check
curl http://localhost:4221/test
```

The server runs on port 4221 via `network_mode: host`. Environment variables go in `.env` (copy from `.env.example`).

**Prerequisites:** Claude CLI must be authenticated on the host (`claude login`). The `~/.claude` directory is mounted into the container for auth.

## Architecture

### Three Source Files

**server.py** — FastAPI endpoints and Telegram message dispatch:
- `POST /message`: Main handler — validates access, saves to chat history, dispatches to LLM
- `POST /inline`: Inline query handler (photo/group selection)
- `GET /test`: Health check
- `call_llm_response()`: Formats LLM output as MarkdownV2, handles >4096 char responses by converting to .txt files
- `flush_media_group()`: Buffers Telegram albums (media groups) with 2s timeout before processing

**panthera.py** — Agent orchestration:
- `Panthera` class: Manages chat history, user sessions, system prompts, LLM invocation via `claude_agent_sdk`
- `_claude_agent_query()`: Core method — sends prompt to Claude with Bash and Read tools enabled
- Chat history is formatted as text and included in the prompt (no LangChain)

**tools_cli.py** — Standalone tool implementations callable via Bash:
- `wolfram_alpha`: Math/science queries via Wolfram|Alpha JSON API
- `web_search`: Perplexity Pro web search with citations
- `generate_image`: Gemini image generation + Telegram delivery
- `update_system_prompt` / `reset_system_prompt`: Per-chat prompt management
- Claude calls these via `python3 /server/tools_cli.py <tool> '<json_args>'`

### Primary Model Configuration

Configured in `config.json`:
```json
{
    "TOKEN": "TELEGRAM-BOT-TOKEN",
    "primary_model": "claude-opus-4-6",
    "token_limit": 50000
}
```

The `primary_model` is passed to `claude_agent_sdk` as the model parameter. The `token_limit` controls chat history pruning.

### How Tools Work

Claude uses `claude_agent_sdk` with `allowed_tools=["Bash", "Read"]`. Tool descriptions are appended to the system prompt (see `TOOL_INSTRUCTIONS` constant in `panthera.py`). When Claude needs a tool, it:

1. Uses **Bash** to run `python3 /server/tools_cli.py <tool_name> '<json_args>'`
2. Uses **Bash** to execute Python code directly (replaces the old `python_repl` tool)
3. Uses **Read** to view image files from chat history

### Adding a New Tool

1. Add an async function in `tools_cli.py`
2. Register it in the `TOOLS` dict at the bottom of `tools_cli.py`
3. Add usage documentation to `TOOL_INSTRUCTIONS` in `panthera.py`

### Chat History

Stored per-user in `data/users/{user_id}/chats/{chat_id}/` as JSON files named `{timestamp}_{message_id}.json`:
```json
{"type": "HumanMessage", "text": "...", "images": []}
```

Pruned automatically by `read_chat_history()`: oldest files removed when exceeding token limit or 2040 message count. Chat history is formatted as text and included in the Claude prompt.

### Message Flow

1. Telegram server forwards message to `/message` endpoint
2. `user_access()` validates authorization (checks `data/users.txt`, group membership)
3. Message metadata assembled and saved to chat history
4. LLM invoked if: private chat, `/*` or `/.` prefix in group, or reply to bot
5. `panthera.llm_request()` loads chat history, formats prompt, calls `claude_agent_sdk.query()`
6. Response formatted with MarkdownV2 and sent via Telegram

### Message Formatting (MarkdownV2)

The bot uses placeholder tokens to avoid conflicts with Telegram's MarkdownV2 escaping:
- `&&&` → `*` (bold), `%%%` → `_` (italic), `@@@` → `__` (underline)
- `~~~` → `~` (strikethrough), `||` → `||` (spoiler), ` ``` ` → ` ``` ` (code blocks)

These are replaced with UUIDs before `escape_markdown()`, then restored after.

### File Path Handling

Image paths from Telegram include user prefixes like `/6014837471:AAE5.../file.jpg`. Always clean with:
```python
re.sub(r'^/[^/]+:', '/', file_path)
```
This is done in `server.py` when extracting paths and in `tools_cli.py` when processing images.

### User Access Control

- `data/users.txt`: Authorized user IDs
- `data/admins.txt`: Admin user IDs (for `/add`, `/remove` commands)
- `data/granted_groups/{chat_id}.txt` / `data/denied_groups/{chat_id}.txt`: Cached group access decisions
- Group access: checks if any authorized user is a member of the group

### Telegram API

Uses a local Telegram bot server, not the official API:
```python
telebot.apihelper.API_URL = 'http://localhost:8081/bot{0}/{1}'
telebot.apihelper.FILE_URL = 'http://localhost:8081'
```

### Response Size Handling

Messages >4096 chars are converted to `.txt` files with heuristic-generated filenames (via `generate_filename()`) and sent as document attachments.

### System Prompts

Default prompt defined in `Panthera.get_system_prompt()`. Per-chat custom prompts stored in `data/custom_prompts/{chat_id}.txt`. Tool instructions (from `TOOL_INSTRUCTIONS` constant) are always appended to the system prompt.

## Key Conventions

- Group chat IDs are negative, user IDs are positive. `chat_id` and `user_id` are used interchangeably for private chats.
- Image file_ids are stored as empty files in `data/users/{chat_id}/images/` for inline query caching.
- `config.json` is the source of truth for model selection — never rely on user session `model` field.
- The `data/` directory is volume-mounted from the host; `config.json` is separately mounted.
- `~/.claude` is mounted for Claude CLI authentication.
- Colons in Telegram file paths require bind mounts on Linux (see README.md for `mount --bind` instructions).
