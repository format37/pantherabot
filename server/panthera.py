import os
import logging
import json
import time
import glob
import re
from pathlib import Path
import tiktoken
import time as py_time
import telebot
from telebot.formatting import escape_markdown

from claude_agent_sdk import (
    query as claude_query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ResultMessage,
)

with open('config.json') as config_file:
    config = json.load(config_file)


TOOL_INSTRUCTIONS = """

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

    def get_message_type(self, user_session, text):
        if text == '/start':
            return 'cmd'
        elif text == '/configure':
            return 'cmd'
        elif text == '/reset':
            return 'cmd'
        with open('data/menu.json') as f:
            menu = json.load(f)
        for key, value in menu.items():
            if text == key:
                return 'button'
            for button in value['buttons']:
                if text == button['text']:
                    return 'button'
        return 'text'

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

    def default_bot_message(self, message, text):
        current_unix_timestamp = int(time.time())
        self.logger.info(f'default_bot_message: {message}')
        if 'first_name' in message['chat']:
            first_name = message['from']['first_name']
        else:
            first_name = message['from']['username']
        return {
        'message_id': int(message['message_id']) + 1,
        'from': {
                'id': 0,
                'is_bot': True,
                'first_name': 'assistant',
                'username': 'assistant',
                'language_code': 'en',
                'is_premium': False
            },
            'chat': {
                'id': message['chat']['id'],
                'first_name': first_name,
                'username': message['from']['username'],
                'type': 'private'
            },
            'date': current_unix_timestamp,
            'text': text
        }

    def add_evaluation_to_topic(self, session, topic_name, value=10):
        if "topics" not in session:
            session["topics"] = {}
        if topic_name not in session["topics"]:
            session["topics"][topic_name] = {"evaluations": []}
        date = int(time.time())
        evaluation_dict = {"date": date, "value": value}
        session["topics"][topic_name]["evaluations"].append(evaluation_dict)
        return session

    def crop_queue(self, chat_id):
        chat_path = os.path.join("data", "chats", str(chat_id))
        Path(chat_path).mkdir(parents=True, exist_ok=True)
        list_of_files = glob.glob(chat_path + "/*.json")
        list_of_files.sort(key=os.path.getctime, reverse=True)
        tokens = 0
        self.logger.info(f"list_of_files: \n{list_of_files}")
        for file in list_of_files:
            if tokens > self.config['token_limit']:
                self.logger.info(f"Removing file: {file}")
                os.remove(file)
                continue
            try:
                message = json.load(open(file, 'r'))
                text = message['text']
                tokens += self.token_counter(text)
                self.logger.info(f"file: {file} tokens: {tokens}")
                if tokens > self.config['token_limit']:
                    self.logger.info(f"Removing file: {file}")
                    os.remove(file)
            except Exception as e:
                self.logger.error(f"Error loading file: {file} error: {e}")
                os.remove(file)

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

    async def _claude_agent_query(self, system_prompt, user_prompt):
        """Query Claude using the agent SDK with Bash and Read tools."""
        self.logger.info("Sending query to Claude agent SDK...")

        stderr_lines = []

        def _stderr_callback(line: str) -> None:
            stderr_lines.append(line)
            self.logger.info(f"Claude CLI stderr: {line}")

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=self.config['model'],
            max_turns=10,
            allowed_tools=["Bash", "Read"],
            permission_mode="bypassPermissions",
            max_thinking_tokens=32768,
            stderr=_stderr_callback,
        )

        try:
            result_text = ""
            async for message in claude_query(prompt=user_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result_text += block.text

            return result_text.strip()
        except Exception as e:
            stderr_text = "\n".join(stderr_lines[-10:]) if stderr_lines else "no stderr captured"
            self.logger.error(f"Claude CLI failed. stderr:\n{stderr_text}")
            raise

    async def llm_request(self, chat_id, message_id, message_text, image_paths=None):
        self.logger.info(f'llm_request: {chat_id}')

        # Read chat history
        self.read_chat_history(chat_id=chat_id)
        self.logger.info(f'invoking message_text: {message_text}')
        system_prompt = self.get_system_prompt(chat_id)

        # Build prompt with chat history context
        history_text = self.format_chat_history()
        user_prompt = ""
        if history_text:
            user_prompt += f"Previous conversation:\n{history_text}\n\n"
        user_prompt += f"Current message:\n{message_text}"

        try:
            response = await self._claude_agent_query(system_prompt, user_prompt)
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
