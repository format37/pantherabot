import os
import logging
import json
import re
from pathlib import Path
import tiktoken
import time as py_time

from claude_agent_sdk import (
    query as claude_query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)

import memory

with open('config.json') as config_file:
    config = json.load(config_file)


# Leading tool-name artifacts like "[Bash]" — the model labelling its reply with
# the tool it used, or imitating a tool call as text instead of invoking it. Must
# never reach the chat or be saved to history, or the model learns the pattern
# from its own history (2026-08-30 incident). `\w+_\w+` catches the bare MCP tool
# names the model also uses, e.g. "[perplexity_sonar_pro]" (seen 2026-09-03).
TOOL_ARTIFACT_RE = re.compile(r'^\s*(?:\[(?:Bash|Read|mcp__\w+|\w+_\w+)\]\s*)+')


# Always included. Nothing here needs a tool, so guests (non-authorized senders in
# a granted group) get it too.
FORMATTING_INSTRUCTIONS = """

## Formatting
Your replies are delivered with Telegram rich message formatting (standard Markdown). These are available — use them when they improve clarity; plain text is perfectly fine:
- **bold**, *italic*, `inline code`, ~~strikethrough~~, ||spoiler||
- headings: # H1, ## H2, ### H3
- bullet lists (- item), numbered lists (1. item), task lists (- [ ] / - [x])
- > block quotations
- tables: | a | b | with a |:--|:--| separator row
- fenced code blocks with a language tag, e.g. ```python ... ```
- collapsible sections: <details><summary>Summary</summary> ...content... </details>
Use standard Markdown: single *asterisks* = italic, double **asterisks** = bold. Do NOT use any &&& / %%% / @@@ placeholder tokens.

## Math
Telegram now renders LaTeX natively in your replies. Write inline math as $...$ and display equations as $$...$$.
For example: $ax^2 + bx + c = 0$ and $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$."""


# Appended only for senders authorized in data/users.txt. A guest turn runs with
# tools=[] and no MCP servers, so describing tools to it would only invite the
# model to fake them as text.
TOOL_INSTRUCTIONS = """

## Image Generation
Generate images using the Gemini Nano Banana model (gemini-3.1-flash-image-preview).
Call it with: python3 /server/tools_cli.py generate_image '{"prompt": "<description>", "chat_id": <chat_id>, "message_id": <message_id>}'
Optionally include "file_list": ["<path>"] to pass input images for editing or composition.
Use this whenever the user asks to generate, create, or draw an image.

## Wolfram Alpha
Use Wolfram Alpha for math, science, unit conversions, equations, and factual lookups.
Call it with: python3 /server/tools_cli.py wolfram_alpha '{"query": "<your query>"}'

## Rasterized formulas
Telegram renders LaTeX natively, so you rarely need this. Only if you specifically need a
rasterized PNG of a formula, call:
python3 /server/tools_cli.py render_math '{"formula": "<LaTeX without $ delimiters>", "chat_id": <chat_id>, "message_id": <message_id>}'

## Images
When a message includes a file_list, use the Read tool to view each image before responding. The Read tool can access those files directly.

## Memory
When the user asks you to remember, save, or keep something in mind (запомни, сохрани, не забудь),
call: python3 /server/tools_cli.py remember '{"chat_id": <chat_id>, "note": "<one short self-contained fact>"}'
When they ask you to forget something (забудь):
python3 /server/tools_cli.py forget '{"chat_id": <chat_id>, "pattern": "<substring, or * for everything>"}'
When the memory is oversized and the user agrees to condense it:
python3 /server/tools_cli.py replace_memory '{"chat_id": <chat_id>, "content": "<the full new list, one note per line>"}'
Always pass the chat_id of the current conversation — any other value is refused.
Do not store secrets, passwords, or tokens. Saved notes appear under "## Memory" in your
instructions on every request; prefer them over older conversation history.

## Web Search
You have access to Perplexity web search tools. Use them when the user asks about recent events, current prices, news, or anything requiring up-to-date information.
Only use tools when the user's request requires them. For normal conversation, respond directly."""


class Panthera:

    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.config = json.load(open('./data/users/default.json', 'r'))
        # Force model from config.json
        self.config['model'] = config.get('primary_model', 'claude-fable-5-1')
        self.logger.info(f'Using model: {self.config["model"]}')
        # Override token_limit from config.json if present
        if 'token_limit' in config:
            self.config['token_limit'] = config['token_limit']
            self.logger.info(f'Token limit: {self.config["token_limit"]}')

        self.data_dir = './data/chats'
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.chat_history = []

    def is_reply_to_ai_message(self, message):
        if "reply_to_message" not in message:
            return False
        if "from" not in message["reply_to_message"]:
            return False
        if "is_bot" not in message["reply_to_message"]["from"]:
            return False
        if message["reply_to_message"]["from"]["is_bot"] == False:
            return False
        if "username" not in message["reply_to_message"]["from"]:
            return False
        if message["reply_to_message"]["from"]["username"] == os.environ.get('BOT_USERNAME', 'your_bot_name'):
            return True

        return False

    def save_user_session(self, user_id, session):
        self.logger.info(f'save_user_session: {user_id} with cmd: {session["last_cmd"]}')
        path = './data/users'
        user_path = os.path.join(path, f'{user_id}.json')
        json.dump(session, open(user_path, 'w'))

    def get_user_session(self, user_id):
        self.logger.info(f'get_user_session: {user_id}')
        path = './data/users'
        user_path = os.path.join(path, f'{user_id}.json')
        if not os.path.exists(user_path):
            default_path = os.path.join(path, 'default.json')
            session = json.load(open(default_path, 'r'))
            self.save_user_session(user_id, session)
        session = json.load(open(user_path, 'r'))
        return session

    def reset_chat(self, chat_id):
        self.logger.info(f'reset_chat: {chat_id}')
        chat_path = os.path.join('data', 'users', str(chat_id), 'chats', str(chat_id))
        Path(chat_path).mkdir(parents=True, exist_ok=True)
        # *.json only: everything else in this folder is not a history record.
        for f in os.listdir(chat_path):
            if not f.endswith('.json'):
                continue
            self.logger.info(f'remove file: {f}')
            os.remove(os.path.join(chat_path, f))

    def token_counter(self, text):
        model_for_tokens = self.config.get('model', 'gpt-4o')
        try:
            enc = tiktoken.encoding_for_model(model_for_tokens)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        return len(tokens)

    def save_to_chat_history(
        self,
        chat_id,
        message_text,
        message_id,
        type,
        message_date=None,
        name_of_user='AI',
        image_paths=None
    ):
        user_id = chat_id
        chat_log_path = os.path.join('data', 'users', str(user_id), 'chats', str(chat_id))
        os.makedirs(chat_log_path, exist_ok=True)
        if message_date is None:
            message_date = py_time.strftime('%Y-%m-%d-%H-%M-%S', py_time.localtime())
        log_file_name = f'{message_date}_{message_id}.json'
        with open(os.path.join(chat_log_path, log_file_name), 'w') as log_file:
            json.dump({
                "type": type,
                "text": f"{message_text}",
                "images": image_paths or []
            }, log_file)

    def get_message_file_list(self, bot, message):
        """Extract file paths from a Telegram message."""
        if 'photo' in message or 'document' in message:
            file_id = ''
            if 'photo' in message:
                photo = message['photo']
                self.logger.info(f"photo in message: {len(photo)}")
                if len(photo) > 0:
                    file_id = photo[-1]['file_id']
                    self.logger.info("file_id: "+str(file_id))
            elif 'document' in message:
                self.logger.info("document in message")
                document = message['document']
                if document['mime_type'].startswith('image/'):
                    file_id = document['file_id']
                    self.logger.info("file_id: "+str(file_id))
                elif document['mime_type'].startswith('text/') or \
                    document['mime_type'].startswith('application/json') or \
                    document['mime_type'].startswith('application/xml'):
                    file_id = document['file_id']
                    self.logger.info("file_id: "+str(file_id))
            if file_id != '':
                file_info = bot.get_file(file_id)
                file_path = file_info.file_path
                self.logger.info(f'file_path: {file_path}')
                return [file_path]
        return []

    def read_chat_history(self, chat_id: str):
        '''Reads the chat history from a folder with improved message limit handling.'''
        user_id = chat_id
        chat_log_path = os.path.join('data', 'users', str(user_id), 'chats', str(chat_id))
        if not os.path.exists(chat_log_path):
            return

        self.chat_history = []

        files = []
        for log_file in os.listdir(chat_log_path):
            # *.json only: a stray file here must not be parsed as a message,
            # nor pruned as an over-quota one.
            if not log_file.endswith('.json'):
                continue
            file_path = os.path.join(chat_log_path, log_file)
            try:
                files.append((file_path, os.path.getctime(file_path)))
            except Exception as e:
                self.logger.error(f'Error getting file creation time: {e}')
                continue

        files.sort(key=lambda x: x[1], reverse=True)

        message_count = 0
        token_count = 0
        MAX_MESSAGES = 2040
        MAX_TOKENS = self.config['token_limit'] if 'token_limit' in self.config else 4000

        for file_path, _ in files:
            if message_count >= MAX_MESSAGES:
                try:
                    os.remove(file_path)
                    self.logger.info(f'Removed old chat history file: {file_path}')
                except Exception as e:
                    self.logger.error(f'Error removing file: {e}')
                continue

            try:
                with open(file_path, 'r') as file:
                    message = json.load(file)

                    message_tokens = self.token_counter(message['text'])

                    if token_count + message_tokens > MAX_TOKENS:
                        try:
                            os.remove(file_path)
                            self.logger.info(f'Removed file exceeding token limit: {file_path}')
                        except Exception as e:
                            self.logger.error(f'Error removing file: {e}')
                        continue

                    if message['type'] == 'AIMessage':
                        self.chat_history.insert(0, {"role": "assistant", "content": message['text']})
                    elif message['type'] == 'HumanMessage':
                        self.chat_history.insert(0, {"role": "user", "content": message['text']})

                    message_count += 1
                    token_count += message_tokens

            except Exception as e:
                self.logger.error(f'Error reading chat history file {file_path}: {e}')
                try:
                    os.remove(file_path)
                    self.logger.error(f'Removed corrupted file: {file_path}')
                except Exception as remove_error:
                    self.logger.error(f'Error removing corrupted file: {remove_error}')

        self.logger.info(f'Loaded {message_count} messages with {token_count} tokens for chat {chat_id}')

    def get_first_name(self, message):
        if 'first_name' in message['chat']:
            first_name = message['from']['first_name']
        elif 'username' in message['from']:
            first_name = message['from']['username']
        elif 'id' in message['from']:
            first_name = message['from']['id']
        else:
            first_name = 'Unknown'
        return first_name

    def get_system_prompt(self, chat_id, tools_enabled=True):
        """Base prompt + formatting rules, plus tool docs and memory when allowed."""
        custom_prompt_path = f'./data/custom_prompts/{chat_id}.txt'
        if os.path.exists(custom_prompt_path):
            with open(custom_prompt_path, 'r') as f:
                base_prompt = f.read().strip()
        else:
            base_prompt = f"""Your name is Janet.
You are Artificial Intelligence and the participant in the multi-user or personal telegram chat.
Your model is {self.config['model']}.
You can determine the current date from the message_date field in the current message."""

        prompt = base_prompt + FORMATTING_INSTRUCTIONS
        if tools_enabled:
            prompt += TOOL_INSTRUCTIONS
        prompt += memory.render_for_prompt(chat_id)
        return prompt

    def format_chat_history(self):
        """Format chat history as text for inclusion in the prompt."""
        if not self.chat_history:
            return ""
        lines = []
        for msg in self.chat_history:
            if msg["role"] == "user":
                lines.append(f"[User]: {msg['content']}")
            elif msg["role"] == "assistant":
                lines.append(f"[Assistant]: {msg['content']}")
        return "\n".join(lines)

    async def _claude_agent_query(self, system_prompt, user_prompt, chat_id=None, tools_enabled=True):
        """Query Claude using the agent SDK with Perplexity MCP tools."""
        self.logger.info(f"Sending query to Claude agent SDK (tools_enabled={tools_enabled})...")

        stderr_lines = []

        def _stderr_callback(line: str) -> None:
            stderr_lines.append(line)
            self.logger.info(f"Claude CLI stderr: {line}")

        perplexity_url = os.environ.get("PERPLEXITY_MCP_URL", "")

        # Built-in tools: Bash runs tools_cli.py, Read views images from chat history.
        # `tools` is the base set the CLI exposes at all (--tools), `allowed_tools`
        # is what runs without a permission prompt. A non-authorized sender in a
        # granted group gets neither, and no MCP servers either.
        base_tools = ["Bash", "Read"] if tools_enabled else []
        allowed_tools = list(base_tools)
        mcp_servers = {}
        if tools_enabled and perplexity_url:
            mcp_servers["perplexity"] = {
                "type": "http",
                "url": perplexity_url,
            }
            allowed_tools.extend([
                "mcp__perplexity__perplexity_sonar",
                "mcp__perplexity__perplexity_sonar_pro",
                "mcp__perplexity__perplexity_sonar_deep_research",
            ])

        # Which chat this turn belongs to. tools_cli.py refuses any chat_id that
        # does not match, so a prompt injection cannot write another chat's memory
        # or send it an image. The Bash tool inherits this env.
        tool_env = {}
        if chat_id is not None:
            tool_env["PANTHERA_CHAT_ID"] = str(chat_id)

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=self.config['model'],
            max_turns=10,
            tools=base_tools,
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers,
            # Only the MCP servers passed above; ignore any .mcp.json the bot
            # could write into its own working directory.
            strict_mcp_config=True,
            # No settings.json from anywhere: a hook written into the config dir
            # by a chat user must never be honoured by the next query.
            # (`[]` -> `--setting-sources=`; only `None` broke older SDKs.)
            setting_sources=[],
            effort="high",  # Fable 5.1: thinking is always on; fixed budgets are rejected
            env=tool_env,
            stderr=_stderr_callback,
        )

        prompt_arg = user_prompt

        result_text = ""
        try:
            async for message in claude_query(prompt=prompt_arg, options=options):
                self.logger.info(f"SDK message type: {type(message).__name__}")
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result_text += block.text

            return result_text.strip()
        except Exception as e:
            if result_text.strip():
                self.logger.warning(f"CLI exited non-zero after successful response, returning result. Error: {e}")
                return result_text.strip()
            stderr_text = "\n".join(stderr_lines[-10:]) if stderr_lines else "no stderr captured"
            self.logger.error(f"Claude CLI failed. stderr:\n{stderr_text}")
            self.logger.error(f"Exception type: {type(e).__name__}, details: {e}")
            raise

    async def llm_request(self, chat_id, message_id, message_text, tools_enabled=True):
        self.logger.info(f'llm_request: {chat_id} (tools_enabled={tools_enabled})')

        # Read chat history
        self.read_chat_history(chat_id=chat_id)
        self.logger.info(f'invoking message_text: {message_text}')
        system_prompt = self.get_system_prompt(chat_id, tools_enabled=tools_enabled)

        # Build prompt with chat history context
        history_text = self.format_chat_history()
        user_prompt = f"chat_id: {chat_id}\nmessage_id: {message_id}\n\n"
        if history_text:
            user_prompt += f"Previous conversation:\n{history_text}\n\n"
        user_prompt += f"Current message:\n{message_text}"

        try:
            response = await self._claude_agent_query(
                system_prompt, user_prompt, chat_id=chat_id, tools_enabled=tools_enabled
            )
            self.logger.info(f'llm_request response: {response[:200]}...' if len(response) > 200 else f'llm_request response: {response}')

            # Handle list/dict responses
            if isinstance(response, list):
                if len(response) > 0:
                    response = response[0]
                else:
                    response = ''
                if isinstance(response, dict):
                    try:
                        response = response['text']
                    except:
                        response = str(response)

            cleaned = TOOL_ARTIFACT_RE.sub('', response).strip()
            if cleaned != response.strip():
                self.logger.warning(f'Stripped tool-name artifact from response: {response[:100]!r}')
            response = cleaned

            if not response:
                self.logger.warning('Empty response after artifact stripping, retrying with tool reminder')
                if tools_enabled:
                    retry_prompt = user_prompt + (
                        '\n\nReminder: tools are only invoked by actually calling the Bash tool. '
                        'Writing a tool name like [Bash] as text does nothing. '
                        'Complete the request now by invoking the tool, then reply with text.'
                    )
                else:
                    retry_prompt = user_prompt + (
                        '\n\nReminder: you have no tools in this conversation. '
                        'Answer directly, in plain text.'
                    )
                response = await self._claude_agent_query(
                    system_prompt, retry_prompt, chat_id=chat_id, tools_enabled=tools_enabled
                )
                response = TOOL_ARTIFACT_RE.sub('', response).strip()
                self.logger.info(f'retry response: {response[:200]}')

            self.save_to_chat_history(
                chat_id,
                response,
                message_id,
                'AIMessage'
            )

            return response

        except Exception as e:
            error_message = f"I encountered an error while processing your request. Please try again later."
            self.logger.error(f"Error in llm_request: {str(e)}", exc_info=True)

            self.save_to_chat_history(
                chat_id,
                error_message,
                message_id,
                'AIMessage'
            )

            return error_message

    async def generate_filename(self, content):
        """Generate a descriptive filename from content."""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        first_line = lines[0] if lines else "response"
        filename = first_line[:50]
        filename = re.sub(r'[^\w\s-]', '', filename)
        filename = re.sub(r'\s+', '_', filename).strip('_')
        if not filename:
            filename = "response"
        if len(filename) > 40:
            filename = filename[:40]
        return filename + ".txt"
