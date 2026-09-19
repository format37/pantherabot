# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pantherabot is a conversational Telegram bot named "Janet" powered by Claude (via `claude_agent_sdk`) with tool access. It's a FastAPI-based server that integrates with Telegram via a separate telegram bot server, providing conversational AI capabilities with tool calling, image generation/understanding, web search, and file handling.

The bot is part of a three-component architecture:
- Telebot server (separate repo: github.com/format37/telegram_bot), the relay that forwards updates here
- Panthera bot (this repo)
- LLM service: Claude subscription via `claude_agent_sdk` (flat-rate, not per-token)

## Development Commands

```bash
# Full rebuild and deploy (sources .env, rebuilds both containers)
./compose.sh

# View logs
./logs.sh  # runs: docker compose logs -f -t

# Health check
curl http://localhost:4221/test

# Tests (no Docker, no Telegram, no Claude): a venv with server/requirements.txt and pytest
python -m pytest tests -q
```

The server runs on port 4221 via `network_mode: host`. Environment variables go in `.env` (copy from `.env.example`).

**Prerequisites:** the bot has its own Claude config dir, `~/.claude-bot` on the host (mounted at `/home/appuser/.claude-bot`), and authenticates with `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` in `.env`. See README.md.

## Architecture

Two containers: `panthera_gptaidbot` (the head: FastAPI, the Claude CLI, every secret) and `panthera_sandbox` (the hands: runs the model's code, no network, no secrets), connected by a Unix socket in `./run`.

### Source Files

**server.py** — FastAPI endpoints and Telegram delivery:
- `POST /message`: validates access, saves the message to the history, and when addressed starts the answer in the background (`answer_in_background()`), so the relay's thread is free again at once
- `POST /edited_message`: rewrites an edited message's history record in place; never answers
- `POST /inline`: inline query handler (photo/group selection)
- `GET /test`: health check
- `human_record()` / `attached_files()`: the one builder for human history records (messages, albums, edits)
- `call_llm_response()`: runs an answer through `edits.answer()`, then saves it and sends its outbox and text (`deliver()`, in a worker thread)
- `flush_media_group()`: buffers Telegram albums (media groups) for `MEDIA_GROUP_WAIT_SECONDS` before saving and answering

**panthera.py** — history, prompts and the model call:
- `save_record()` / `rewrite_record()` / `find_human_records()` / `read_chat_history()`: chat history on disk
- `prepare_prompt()`: reads the history (synchronously) and returns the prompts and the context
- `generate()`: calls `_claude_agent_query()`, strips tool-name artifacts, retries an empty answer; saves nothing
- `_claude_agent_query()`: `claude_agent_sdk.query()` with no built-in tools and the in-process MCP server

**edits.py** — the edit rule: attempts, cancellation, the pre-send check, the regeneration cap, and the limit of `MAX_CONCURRENT_ANSWERS` answers at once.

**bot_tools.py** — the model's tools, as an in-process MCP server built per attempt (`create_bot_server()` / `build_tools()`), with chat_id and message_id in closures: `run_command` (sandbox), `view_image`, `send_file`, `generate_image`, `wolfram_alpha`, `web_search` / `deep_research` (Perplexity), `render_math`, `remember` / `forget` / `replace_memory`, `update_system_prompt` / `reset_system_prompt`.

**tools_cli.py** — implementations behind several tools (Gemini images, Wolfram|Alpha, formula PNGs, prompt files) and an operator CLI: `python3 tools_cli.py <tool> '<json_args>'`.

**research.py** — Perplexity through its API: `search()` (sonar-pro, synchronous, run in a thread) and `start()`, which runs a deep research job as a task of the event loop, independent of the answer that started it. When the job ends it calls `server.research_done()`: the report goes to the chat as a `.md` document, its text is filed as a history record from `deep_research (tool)` (suffix `research-{message_id}`, so an edit never touches it), and an answer to it starts as for any message. One job per message, `MAX_RESEARCH_JOBS` per process. Needs `PERPLEXITY_API_KEY`; without it the Perplexity MCP server (`PERPLEXITY_MCP_URL`) is attached instead.

**memory.py** — per-chat long-term notes in `data/users/{chat_id}/CLAUDE.md`, injected into the system prompt and kept across `/reset`.

**exec_service.py** — the sandbox's exec service (`panthera_sandbox`), reached over `./run/exec.sock`. A command is killed when it times out or when the head hangs up (its answer was cancelled).

### Primary Model Configuration

Configured in `config.json`:
```json
{
    "TOKEN": "TELEGRAM-BOT-TOKEN",
    "primary_model": "claude-fable-5-1",
    "token_limit": 50000
}
```

The `primary_model` is passed to `claude_agent_sdk` as the model parameter. The `token_limit` controls chat history pruning.

### How Tools Work

`_claude_agent_query()` passes `tools=[]`: the model has no Bash, Read or Write. Everything it can do is an MCP tool: the `bot` server from `bot_tools.py` (`mcp__bot__*`) and, only without `PERPLEXITY_API_KEY`, the Perplexity MCP server (`mcp__perplexity`, the whole server; with the key, search is the bot's own `web_search` / `deep_research`). `strict_mcp_config=True` and `setting_sources=[]` keep out anything written into the config dir. Code runs in the sandbox through `run_command`. Senders not in `data/users.txt` (guests in a granted group) get no tools at all. Tool usage is described in `TOOL_INSTRUCTIONS` in `panthera.py`, appended to the system prompt for authorized senders only.

Tools never send to the chat themselves. `send_file`, `generate_image` and `render_math` append a `bot_tools.Outgoing` (the bytes, taken when the tool runs) to the attempt's outbox, up to `MAX_OUTBOX_BYTES`. The outbox is sent just before the answer's text, and only for the attempt that is sent.

### Adding a New Tool

1. Define it with `@tool` inside `bot_tools.build_tools()` and add it to the list at the end of that function.
2. Add its name to `bot_tools.TOOL_NAMES` (it becomes `mcp__bot__<name>` in `allowed_tools`).
3. A tool that delivers a file appends an `Outgoing` to `outbox` rather than sending it.
4. Document it in `TOOL_INSTRUCTIONS` in `panthera.py`.

### Chat History

Stored per chat in `data/users/{chat_id}/chats/{chat_id}/`, one JSON file per record, named `{save-time}_{message_id}.json` (save-time is the container's local time, Pacific, down to microseconds). An answer is filed under the id of the message it answers.
```json
{"type": "HumanMessage", "text": "user_name: ...\nmessage_text: ...", "images": [],
 "message_id": 123, "raw_text": "...", "file_unique_ids": []}
```
- `text` is what the model sees; `raw_text` is the text or caption as sent. An edited record also has `edit_date` (UTC), shown as an `edit_date:` line in `text`. An album record adds `media_group_id` and `captions` (per item) and is filed under its first item. Records older than 2026-09-16 have only `type`, `text` and `images`; their id comes from the file name.
- **Order is file mtime.** A record is written once (`save_record()`), and only `rewrite_record()` changes it afterwards, keeping its mtime. Never sort by ctime: a `chown -R` or any rewrite changes it.
- `read_chat_history()` loads the newest records that fit the token limit and 2040 messages, deletes the rest, and returns the human records it loaded (the context).

### Message Flow

1. The relay forwards a message to `/message`.
2. `user_access()` validates authorization (checks `data/users.txt`, group membership).
3. Commands (`/add`, `/remove`, `/help`, `/reset`, `/memory`, `/forget`, `/start`, `response:`) are handled and not saved.
4. The human record is built (`human_record()`) and saved.
5. Janet answers if: private chat, `/*` or `/.` prefix in a group (on any caption of an album), or a reply to the bot. The request returns now; the answer runs as a background task.
6. `edits.answer()` runs attempts. Each attempt reads the history (`prepare_prompt()`, current message from its record) and generates (`generate()`), with files going to its outbox.
7. Before sending, the pre-send check discards an attempt whose context was edited meanwhile, and a new attempt starts.
8. The answer is saved as an `AIMessage`, and its outbox and text are sent as a Telegram rich message (fallbacks: a `.txt` document, then MarkdownV2).

**Edits** (the relay forwards them only with `"forward_edits": 1` in its `bots.json`):
1. `/edited_message` finds the message's human record by its file-name suffix and rewrites it in place with `edit_date`. An edit that changes neither the text/caption nor the files is ignored.
2. An edit never answers and never runs a command. A sent answer is final.
3. If an answer is being generated from a context that includes the message, the running attempt is cancelled (its CLI ends about 5 s later, its sandbox command at once). The answer is regenerated once edits stop for `DEBOUNCE_SECONDS` (waiting `MAX_SETTLE_SECONDS` at most), at most `MAX_REGENERATIONS` times; after that the next attempt is sent as is.
4. An album is found by the edited item's id or, for the other items, among the nine ids before it (the album is filed under its first item). An album still being collected just gets the new caption; a caption change that leaves the album's text the same only updates the record's `captions`.

### Message Formatting

Answers go out as Telegram rich messages (`sendRichMessage`, standard Markdown, up to 32768 chars). The MarkdownV2 fallback uses placeholder tokens to avoid conflicts with Telegram's escaping:
- `&&&` → `*` (bold), `%%%` → `_` (italic), `@@@` → `__` (underline)
- `~~~` → `~` (strikethrough), `||` → `||` (spoiler), ` ``` ` → ` ``` ` (code blocks)

These are replaced with UUIDs before `escape_markdown()`, then restored after.

### File Path Handling

Image paths from Telegram include user prefixes like `/6014837471:AAE5.../file.jpg`. Always clean with:
```python
re.sub(r'^/[^/]+:', '/', file_path)
```
This is done in `server.attached_files()`. Tools accept only paths under the Telegram file store or the chat's sandbox work dir (`bot_tools._resolve()`).

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

An answer that cannot go out as a rich message and is over 4096 chars is sent as a `.txt` document with a heuristic file name (`generate_filename()`).

### System Prompts

Default prompt defined in `Panthera.get_system_prompt()`. Per-chat custom prompts stored in `data/custom_prompts/{chat_id}.txt`. `FORMATTING_INSTRUCTIONS` and the chat's memory are always appended; `TOOL_INSTRUCTIONS` only for authorized senders.

## Key Conventions

- Group chat IDs are negative, user IDs are positive. `chat_id` and `user_id` are used interchangeably for private chats.
- Image file_ids are stored as empty files in `data/users/{chat_id}/images/` for inline query caching (written when a generated image is sent).
- `config.json` is the source of truth for model selection — never rely on user session `model` field.
- The `data/` directory is volume-mounted from the host; `config.json` is separately mounted.
- `server/panthera.py` and `server/server.py` use CRLF line endings; keep them.
- A new module in `server/` needs its own `COPY` line in `server/Dockerfile`.
- Nothing that takes seconds may run on the event loop (the Gemini call once froze every chat for 40 s): use `asyncio.to_thread`.
- `generate_image` reaches Gemini through **Vertex AI** when `VERTEX_SA_JSON_B64` is set (the production VPS: the Gemini Developer API refuses that host's address for every project, and no billing change fixes it), and through `GEMINI_API_KEY` otherwise. The default model is Nano Banana Pro, `gemini-3-pro-image` (about $0.24 per 4K image); `gemini-3.1-flash-image` is the cheaper, faster alternative, and `GEMINI_IMAGE_MODEL` in `.env` overrides the default on either door.
- Colons in Telegram file paths require bind mounts on Linux (see README.md for `mount --bind` instructions).
