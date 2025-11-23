# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pantherabot is a conversational Telegram bot named "Janet" powered by LangChain agents with multiple AI models and tools. It's a FastAPI-based server that integrates with Telegram via a separate telegram bot server, providing conversational AI capabilities with tool calling, image generation/understanding, web search, and file handling.

The bot is part of a three-component architecture:
- Telebot server (separate repo: github.com/format37/telegram_bot)
- Panthera bot (this repo)
- LLM service (separate repo: github.com/format37/openai_proxy)

## Development Commands

### Docker Operations
```bash
# Build and start the container
docker-compose up --build -d

# View logs
./logs.sh  # or: docker logs -f panthera_gptaidbot

# Restart the service
docker-compose restart

# Stop the service
docker-compose down
```

### Local Development
The server runs on port 4221 inside the container, exposed via `network_mode: host`.

Test endpoint:
```bash
curl http://localhost:4221/test
```

## Environment Configuration

Copy `.env.example` to `.env` and configure:
- `TELEGRAM_BOT_TOKEN`: Telegram bot token
- `OPENAI_API_KEY`: Required for primary LLM (configurable model)
- `ANTHROPIC_API_KEY`: For Claude models
- `GEMINI_API_KEY`: For Gemini/NanoBanana image generation
- `BFL_API_KEY`: For Flux Pro image generation
- `PERPLEXITY_API_KEY`: For web search
- `WOLFRAM_ALPHA_APPID`: For math/science queries
- `SERPER_API_KEY`: For Google search (currently disabled)
- `LANGCHAIN_API_KEY`: For LangSmith tracing
- `BOT_USERNAME`: Bot's Telegram username

## Architecture

### Core Components

**server.py (FastAPI Server)**
- POST `/message`: Main message handler with user access control
- POST `/inline`: Inline query handler for photo/group selection
- GET `/test`: Health check endpoint
- Handles Telegram API communication via custom API URL (localhost:8081)

**panthera.py (Agent Logic)**
- `Panthera` class: Main bot orchestrator managing chat history, user sessions, system prompts
- `ChatAgent` class: LangChain agent executor with tool integrations

### Primary Model Configuration

The primary model is configured in `config.json`:
```json
{
    "TOKEN": "TELEGRAM-BOT-TOKEN",
    "primary_model": "gpt-5.1"
}
```

This model is used throughout the codebase and overrides any stored user session models. When modifying model selection logic, ensure the primary_model from config.json is respected.

### Chat History Management

Chat histories are stored per-user in `data/users/{user_id}/chats/{chat_id}/` as JSON files with format:
```json
{
    "type": "HumanMessage" | "AIMessage",
    "text": "message content"
}
```

Files are named `{timestamp}_{message_id}.json` and automatically pruned based on:
- Token limits (configurable via `token_limit` in user sessions)
- Message count (max 2040 messages)
- Oldest files removed first when limits exceeded

### User Access Control

User authorization is managed via:
- `data/users.txt`: Authorized user IDs
- `data/admins.txt`: Admin user IDs
- `data/granted_groups/{chat_id}.txt`: Whitelisted groups
- `data/denied_groups/{chat_id}.txt`: Blacklisted groups

Group access is determined by checking if authorized users are members.

### Available Tools

The LangChain agent has access to:
- `python_repl`: Execute Python code
- `wolfram_alpha`: Math/science queries (JSON API)
- `image_context_conversation`: Vision understanding (uses primary_model)
- `image_plotter_nanobanana`: Gemini image generation (supports multi-image input)
- `update_system_prompt` / `reset_system_prompt`: Per-chat prompt customization
- `web_search`: Perplexity Pro web search with citations

Commented out tools (can be re-enabled):
- `google_search_tool`: Google search via Serper
- `youtube_tool`: YouTube search
- `wikipedia_tool`: Wikipedia queries
- `image_plotter` (BFL): Flux Pro 1.1 Ultra image generation
- `image_plotter_openai`: OpenAI gpt-image-1 generation/editing
- `text_file_reader`: Read text/JSON files
- `ask_reasoning`: o1-preview reasoning expert

### Message Formatting

The bot uses Telegram MarkdownV2 with custom pre-processing:
- `&&&` → `*` (bold)
- `%%%` → `_` (italic)
- `@@@` → `__` (underline)
- `~~~` → `~` (strikethrough)
- `||` → `||` (spoiler)
- ` ``` ` → ` ``` ` (code blocks)

These are converted to unique tokens before escaping, then restored after escaping to prevent conflicts.

### File Path Handling

Image file paths from Telegram may include user prefixes like `/6014837471:AAE5.../file.jpg`. When processing files:
1. Strip the prefix: `re.sub(r'^/[^/]+:', '/', file_path)` (see panthera.py:687, 814)
2. Use the cleaned path for file operations

This is critical for image_context_conversation and image_plotter_nanobanana tools.

### Volume Mounting Notes

The project supports mounting external directories (e.g., Telegram media) on Linux systems where colons in paths are problematic:
```bash
sudo mount --bind "/user_id:token" "/mnt/token"
```
For persistence across reboots, add to `/etc/fstab`.

### Response Size Handling

Messages exceeding 4096 characters are:
1. Converted to `.txt` files with AI-generated filenames (via `generate_filename()`)
2. Sent as document attachments instead of text messages
3. Filename generation uses gpt-5-nano with structured output

### System Prompts

Default system prompt is in `panthera.py` (get_system_prompt method). Per-chat custom prompts are stored in `data/custom_prompts/{chat_id}.txt`.

The system prompt includes:
- Bot name (Janet)
- Model information (dynamically inserted)
- Current date determination instructions
- MarkdownV2 formatting examples

### Telegram API Configuration

The bot communicates with a local Telegram bot server:
```python
telebot.apihelper.API_URL = 'http://localhost:8081/bot{0}/{1}'
telebot.apihelper.FILE_URL = 'http://localhost:8081'
```

### Error Handling

The `llm_request` method uses tenacity for retry logic:
- 3 retry attempts with exponential backoff (2-10 seconds)
- Retries on: `httpx.RemoteProtocolError`, `httpx.ReadTimeout`, `httpx.ConnectTimeout`
- Errors are logged and saved to chat history

### Image Generation

**NanoBanana (Gemini) - Primary Image Tool**:
- Model: gemini-3-pro-image-preview
- Supports multiple input images
- Image size: 1K
- Includes Google Search tool integration
- Returns image data inline

**BFL (Flux Pro)**:
- Model: flux-pro-1.1-ultra
- Size: 1280x1280
- Raw mode option for less processed images
- Async task polling (5s intervals)

**OpenAI gpt-image-1**:
- Supports image editing with up to 10 input images
- Returns base64 encoded images

All image generators:
- Store file_ids in `data/users/{chat_id}/images/`
- Truncate prompts to 1000 chars for captions
- Use spoiler formatting for captions

## Common Patterns

### Adding a New Tool

1. Define Pydantic args schema (e.g., `class MyToolArgs(BaseModel)`)
2. Implement async tool function in `ChatAgent` class
3. Create `StructuredTool.from_function()` with coroutine
4. Append to `tools` list in `initialize_agent()`

### Message Flow

1. Telegram server forwards message to `/message` endpoint
2. `user_access()` validates authorization
3. Message saved to chat history with metadata
4. `call_llm_response()` invoked if conditions met (private chat, command prefix, or reply to bot)
5. `panthera.llm_request()` loads chat history, invokes agent
6. Response formatted with MarkdownV2 and sent

### Working with Chat IDs

Group chat IDs are negative (e.g., `-888407449`). User IDs are positive. The codebase often uses `chat_id` interchangeably with `user_id` for private chats.

## Testing Considerations

- The bot requires external services (Telegram server on port 8081)
- Mock external API calls when testing tool functions
- Test chat history pruning with various token limits
- Verify file path cleaning for image inputs
- Test message length handling (4096 char limit)

## Important Notes

- Never commit `.env` file (contains secrets)
- The primary_model in `config.json` overrides all user session models
- Chat history files are automatically managed and pruned
- Image file_ids are stored as empty files for inline query caching
- The bot uses a custom Telegram API server URL (not official api.telegram.org)
- Admin commands (`/add`, `/remove`) require admin.txt authorization
- Group access is cached in granted_groups/denied_groups for performance
