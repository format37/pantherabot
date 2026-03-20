import os
import logging
import json
import re
import base64
import mimetypes
from pathlib import Path
import tiktoken
import time as py_time

from claude_agent_sdk import (
    query as claude_query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)

with open('config.json') as config_file:
    config = json.load(config_file)


TOOL_INSTRUCTIONS = """

## Image Generation
Generate images using the Gemini Nano Banana model (gemini-3.1-flash-image-preview).
Call it with: python3 /server/tools_cli.py generate_image '{"prompt": "<description>", "chat_id": <chat_id>, "message_id": <message_id>}'
Optionally include "file_list": ["<path>"] to pass input images for editing or composition.
Use this whenever the user asks to generate, create, or draw an image.

## Wolfram Alpha
Use Wolfram Alpha for math, science, unit conversions, equations, and factual lookups.
Call it with: python3 /server/tools_cli.py wolfram_alpha '{"query": "<your query>"}'

## Render Math
Telegram does NOT render LaTeX. Never output LaTeX notation (no $...$, \\frac, \\int, etc.) in your text responses.
When a response contains a mathematical formula or equation, render it as an image instead:
python3 /server/tools_cli.py render_math '{"formula": "<LaTeX without $ delimiters>", "chat_id": <chat_id>, "message_id": <message_id>}'
After sending the image, write the surrounding explanation in plain text using Unicode math where helpful (e.g. ∫, ², ³, √, ≈, ±).

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
        self.config['model'] = config.get('primary_model', 'claude-opus-4-6')
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
        for f in os.listdir(chat_path):
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

    def get_system_prompt(self, chat_id):
        """Get system prompt with tool descriptions appended."""
        custom_prompt_path = f'./data/custom_prompts/{chat_id}.txt'
        if os.path.exists(custom_prompt_path):
            with open(custom_prompt_path, 'r') as f:
                base_prompt = f.read().strip()
        else:
            markdown_sample = """&&&bold text&&&
%%%italic text%%%
@@@underline@@@
~~~strikethrough~~~
||spoiler||
```
pre-formatted fixed-width code block
```"""
            base_prompt = f"""Your name is Janet.
You are Artificial Intelligence and the participant in the multi-user or personal telegram chat.
Your model is {self.config['model']}.
You can determine the current date from the message_date field in the current message.
For the formatting you can use the telegram MarkdownV2 format. For example: {markdown_sample}."""

        return base_prompt + TOOL_INSTRUCTIONS

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

    async def _claude_agent_query(self, system_prompt, user_prompt, image_paths=None):
        """Query Claude using the agent SDK with Perplexity MCP tools."""
        self.logger.info("Sending query to Claude agent SDK...")

        stderr_lines = []

        def _stderr_callback(line: str) -> None:
            stderr_lines.append(line)
            self.logger.info(f"Claude CLI stderr: {line}")

        perplexity_url = os.environ.get("PERPLEXITY_MCP_URL", "")

        # Build allowed_tools list: Bash+Read always enabled for tools_cli.py calls
        allowed_tools = ["Bash", "Read"]
        mcp_servers = {}
        if perplexity_url:
            mcp_servers["perplexity"] = {
                "type": "http",
                "url": perplexity_url,
            }
            allowed_tools.extend([
                "mcp__perplexity__perplexity_sonar",
                "mcp__perplexity__perplexity_sonar_pro",
                "mcp__perplexity__perplexity_sonar_deep_research",
            ])

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=self.config['model'],
            max_turns=10,
            allowed_tools=allowed_tools if allowed_tools else [],
            mcp_servers=mcp_servers if mcp_servers else None,
            setting_sources=["user"],
            max_thinking_tokens=10000,
            stderr=_stderr_callback,
        )

        # Build prompt: multimodal AsyncIterable when images present, plain string otherwise
        if image_paths:
            async def _multimodal_prompt():
                content = []
                for img_path in image_paths:
                    try:
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                        mime_type, _ = mimetypes.guess_type(img_path)
                        if not mime_type or not mime_type.startswith('image/'):
                            mime_type = 'image/jpeg'
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64.b64encode(img_bytes).decode(),
                            }
                        })
                        self.logger.info(f"Included image in context: {img_path} ({mime_type})")
                    except Exception as e:
                        self.logger.error(f"Failed to include image {img_path}: {e}")
                content.append({"type": "text", "text": user_prompt})
                yield {
                    "type": "user",
                    "session_id": "",
                    "message": {"role": "user", "content": content},
                    "parent_tool_use_id": None,
                }
            prompt_arg = _multimodal_prompt()
        else:
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

    async def llm_request(self, chat_id, message_id, message_text, image_paths=None):
        self.logger.info(f'llm_request: {chat_id}')

        # Read chat history
        self.read_chat_history(chat_id=chat_id)
        self.logger.info(f'invoking message_text: {message_text}')
        system_prompt = self.get_system_prompt(chat_id)

        # Build prompt with chat history context
        history_text = self.format_chat_history()
        user_prompt = f"chat_id: {chat_id}\nmessage_id: {message_id}\n\n"
        if history_text:
            user_prompt += f"Previous conversation:\n{history_text}\n\n"
        user_prompt += f"Current message:\n{message_text}"

        try:
            response = await self._claude_agent_query(system_prompt, user_prompt, image_paths=image_paths)
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
