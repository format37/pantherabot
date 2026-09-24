import os
import logging
import json
import re
import stat
from datetime import datetime
from pathlib import Path
import tiktoken

from claude_agent_sdk import (
    query as claude_query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)

import bot_tools
import memory
import research

with open('config.json') as config_file:
    config = json.load(config_file)


# Leading tool-name artifacts like "[Bash]" — the model labelling its reply with
# the tool it used, or imitating a tool call as text instead of invoking it. Must
# never reach the chat or be saved to history, or the model learns the pattern
# from its own history (2026-08-30 incident). `\w+_\w+` catches the bare MCP tool
# names the model also uses, e.g. "[perplexity_sonar_pro]" (seen 2026-09-03).
TOOL_ARTIFACT_RE = re.compile(r'^\s*(?:\[(?:Bash|Read|mcp__\w+|\w+_\w+)\]\s*)+')


def record_message_id(path, record):
    """The Telegram message_id a history record is filed under.

    Stored in the record since 2026-09-16; older records only have it as the
    file name's suffix, `{save-time}_{message_id}.json`.
    """
    value = record.get('message_id')
    if value is None:
        value = os.path.basename(path)[:-len('.json')].rpartition('_')[2]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def record_raw_text(record):
    """A human record's text or caption as sent (older records: parsed from `text`)."""
    if 'raw_text' in record:
        return record['raw_text']
    _, marker, raw = record.get('text', '').partition('\nmessage_text: ')
    return raw if marker else ''


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
# no tools at all, so describing tools to it would only invite the model to fake
# them as text.
TOOL_INSTRUCTIONS = """

## Tools
Everything you can do beyond talking is an MCP tool — invoke it. Writing a tool
name as text does nothing. You have no shell and no file access of your own; the
tools are the only way out of this process.

## Images
When a message includes a file_list, call view_image on each path before answering:
that is the only way to actually see a photo. Images produced in the sandbox can be
viewed the same way, and delivered to the chat with send_file.
Use generate_image whenever the user asks to generate, create, draw or edit an image.
Everything send_file, generate_image and render_math deliver reaches the chat together
with your reply, just before its text.

## Code execution
run_command runs bash in an isolated sandbox container: no network, no access to the
bot's data or credentials, and a working directory of its own that persists between
messages in this chat. python3 there has pandas, numpy, matplotlib and pillow — do not
try to install anything, there is no network. Write results to the working directory
and deliver them with send_file.

## Memory
When the user asks you to remember, save, or keep something in mind (запомни, сохрани,
не забудь), call remember with a short, self-contained note. Call forget when they ask
you to forget something (забудь). Do not store secrets, passwords, or tokens. Saved
notes appear under "## Memory" in your instructions on every request; prefer them over
older conversation history.

## Web Search
Use web_search when the user asks about recent events, current prices, news, or anything
requiring up-to-date information; it answers in seconds with sources. Put the sources in
your reply as links, so the user can check them.
deep_research is for an explicit request for research, a report, or a thorough comparison.
It takes 3-10 minutes and runs on its own: tell the user it has started and end your reply.
When it is done, its report is posted in this chat as a file, and you get a message from
"deep_research (tool)" with the report: present it then, in the language of the chat, with
the key findings and the links that back them. The user does not need to ask for it.
If the tools are named perplexity_* instead, the same applies: they are the same search.
Only use tools when the user's request requires them. For normal conversation, respond directly."""


class Panthera:

    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.config = json.load(open('./data/users/default.json', 'r'))
        # Force model from config.json
        self.config['model'] = config.get('primary_model', 'claude-opus-5-5')
        self.logger.info(f'Using model: {self.config["model"]}')
        # Override token_limit from config.json if present
        if 'token_limit' in config:
            self.config['token_limit'] = config['token_limit']
            self.logger.info(f'Token limit: {self.config["token_limit"]}')

        self._enc = None
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

    def _encoder(self):
        """Resolve the tokenizer once, or None if it is unavailable.

        `get_encoding` downloads the BPE file on a cold cache, so this can fail
        on a network blip. It must not raise: the only caller is history
        pruning, and an exception there used to be read as a corrupt file.
        """
        if self._enc is not None:
            return self._enc
        for resolve in (
            lambda: tiktoken.encoding_for_model(self.config.get('model', 'gpt-4o')),
            lambda: tiktoken.get_encoding("cl100k_base"),
        ):
            try:
                self._enc = resolve()
                return self._enc
            except Exception as e:
                last_error = e
        self.logger.warning(f'Tokenizer unavailable, estimating token counts: {last_error}')
        return None

    def token_counter(self, text):
        enc = self._encoder()
        if enc is None:
            # Roughly four characters per token — good enough to keep pruning
            # working, and far better than failing.
            return max(1, len(text) // 4)
        return len(enc.encode(text))

    def chat_log_path(self, chat_id):
        return os.path.join('data', 'users', str(chat_id), 'chats', str(chat_id))

    def save_record(self, chat_id, message_id, record, message_date=None):
        """Write a new history record and return its path.

        Its mtime is its place in the history (see read_chat_history).
        """
        chat_log_path = self.chat_log_path(chat_id)
        os.makedirs(chat_log_path, exist_ok=True)
        if message_date is not None:
            path = os.path.join(chat_log_path, f'{message_date}_{message_id}.json')
            with open(path, 'w') as log_file:
                json.dump(record, log_file)
            return path
        # An answer is filed under the message it answers. With whole seconds
        # in the name, an answer written in the same second as its question
        # replaced the question's record, so the time goes down to microseconds
        # and the file must be new.
        while True:
            stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            path = os.path.join(chat_log_path, f'{stamp}_{message_id}.json')
            try:
                with open(path, 'x') as log_file:
                    json.dump(record, log_file)
                return path
            except FileExistsError:
                continue

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
        self.save_record(chat_id, message_id, {
            "type": type,
            "text": f"{message_text}",
            "images": image_paths or []
        }, message_date)

    def rewrite_record(self, path, record):
        """Replace a record's content without moving it in the history.

        The new content goes to a temp file that takes over the old file's mode
        and times before it is renamed over it, so a reader sees either the old
        record or the new one, always at the old position.
        """
        st = os.stat(path)
        tmp = f'{path}.{os.getpid()}.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(record, f)
            os.chmod(tmp, stat.S_IMODE(st.st_mode))
            os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def find_human_records(self, chat_id, message_ids):
        """{message_id: [(path, record)]} of the human records filed under these ids.

        Only the files whose name ends in one of the ids are read.
        """
        chat_log_path = self.chat_log_path(chat_id)
        if not os.path.isdir(chat_log_path):
            return {}
        wanted = {str(message_id) for message_id in message_ids}
        found = {}
        for name in sorted(os.listdir(chat_log_path)):
            if not name.endswith('.json'):
                continue
            suffix = name[:-len('.json')].rpartition('_')[2]
            if suffix not in wanted:
                continue
            path = os.path.join(chat_log_path, name)
            try:
                with open(path) as f:
                    record = json.load(f)
            except Exception:
                continue
            if not isinstance(record, dict) or record.get('type') != 'HumanMessage':
                continue
            found.setdefault(int(suffix), []).append((path, record))
        return found

    def attached_file(self, message):
        """(file_id, file_unique_id) of the file the bot reads from a message, or None."""
        if 'photo' in message:
            photo = message['photo']
            self.logger.info(f"photo in message: {len(photo)}")
            if len(photo) > 0:
                return photo[-1]['file_id'], photo[-1].get('file_unique_id')
        elif 'document' in message:
            self.logger.info("document in message")
            document = message['document']
            mime_type = document.get('mime_type', '')
            if mime_type.startswith('image/') or \
                mime_type.startswith('text/') or \
                mime_type.startswith('application/json') or \
                mime_type.startswith('application/xml'):
                return document['file_id'], document.get('file_unique_id')
        return None

    def get_message_file_list(self, bot, message):
        """Extract file paths from a Telegram message."""
        attached = self.attached_file(message)
        if attached is None:
            return []
        self.logger.info("file_id: "+str(attached[0]))
        file_info = bot.get_file(attached[0])
        file_path = file_info.file_path
        self.logger.info(f'file_path: {file_path}')
        return [file_path]

    def read_chat_history(self, chat_id: str):
        '''Load the newest records that fit the limits into self.chat_history.

        History order is file mtime. A record is written once, and only
        rewrite_record() changes it afterwards, keeping the mtime. ctime is no
        use: `chown -R` on 2026-03-20 reset it for 553 files in 5 chats, and
        any rewrite bumps it.

        Returns {message_id: text} of the human records loaded: the context an
        edit has to touch to matter to the generation that read it.
        '''
        self.chat_history = []
        context = {}
        chat_log_path = self.chat_log_path(chat_id)
        if not os.path.exists(chat_log_path):
            return context

        files = []
        for log_file in os.listdir(chat_log_path):
            # *.json only: a stray file here must not be parsed as a message,
            # nor pruned as an over-quota one.
            if not log_file.endswith('.json'):
                continue
            file_path = os.path.join(chat_log_path, log_file)
            try:
                files.append((file_path, os.stat(file_path).st_mtime_ns))
            except Exception as e:
                self.logger.error(f'Error getting file modification time: {e}')
                continue

        files.sort(key=lambda x: (x[1], os.path.basename(x[0])), reverse=True)

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

            # Deleting is only ever right for a file that is not a history
            # record. Anything else — a failed read, a tokenizer outage — is
            # transient, and this loop walks every file in the chat, so treating
            # it as corruption would wipe the whole conversation in one pass.
            try:
                with open(file_path, 'r') as file:
                    message = json.load(file)
                if not isinstance(message, dict) or 'text' not in message or 'type' not in message:
                    raise ValueError('not a chat history record')
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
                self.logger.error(f'Removing corrupted chat history file {file_path}: {e}')
                try:
                    os.remove(file_path)
                except Exception as remove_error:
                    self.logger.error(f'Error removing corrupted file: {remove_error}')
                continue
            except Exception as e:
                self.logger.error(f'Skipping unreadable chat history file {file_path}: {e}')
                continue

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
                message_id = record_message_id(file_path, message)
                if message_id is not None:
                    context[message_id] = message['text']

            message_count += 1
            token_count += message_tokens

        self.logger.info(f'Loaded {message_count} messages with {token_count} tokens for chat {chat_id}')
        return context

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

    async def _claude_agent_query(self, system_prompt, user_prompt, chat_id=None,
                                  message_id=None, tools_enabled=True, outbox=None):
        """Query Claude using the agent SDK with Perplexity MCP tools.

        Files the bot tools produce are appended to `outbox` and sent with the
        answer.
        """
        self.logger.info(f"Sending query to Claude agent SDK (tools_enabled={tools_enabled})...")

        stderr_lines = []

        def _stderr_callback(line: str) -> None:
            stderr_lines.append(line)
            self.logger.info(f"Claude CLI stderr: {line}")

        perplexity_url = os.environ.get("PERPLEXITY_MCP_URL", "")

        # No built-in tools at all (`tools=[]` -> `--tools ""`): no Bash, no Read,
        # no Write. Everything the model can do is an in-process MCP tool in
        # bot_tools, which keeps the secrets here and sends code to the sandbox
        # container. A non-authorized sender in a granted group gets no tools and
        # no MCP servers either.
        allowed_tools = []
        mcp_servers = {}
        if tools_enabled:
            # Rebuilt per request: chat_id and message_id live in closures, so no
            # tool takes a chat_id and none can be pointed at another chat.
            mcp_servers["bot"] = bot_tools.create_bot_server(chat_id, message_id, outbox)
            allowed_tools.extend(f"mcp__bot__{name}" for name in bot_tools.TOOL_NAMES)
        # Search is the bot's own web_search/deep_research when the bot has a
        # Perplexity key; the Perplexity MCP server is the fallback without one.
        if tools_enabled and perplexity_url and not research.enabled():
            mcp_servers["perplexity"] = {
                "type": "http",
                "url": perplexity_url,
            }
            # The bare server name is a permission rule that allows every tool
            # the server offers. Listing the tools by name left
            # get_research_result asking for permission, which no one can grant
            # from a chat (seen 2026-09-19).
            allowed_tools.append("mcp__perplexity")

        # Vestigial but cheap: tools_cli.py can still be run by an operator, and
        # refuses any chat_id that does not match this turn.
        tool_env = {}
        if chat_id is not None:
            tool_env["PANTHERA_CHAT_ID"] = str(chat_id)

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=self.config['model'],
            max_turns=10,
            tools=[],
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

    def prepare_prompt(self, chat_id, message_id, message_text, tools_enabled=True,
                       from_history=True):
        """(system_prompt, user_prompt, context) for one attempt at an answer.

        Synchronous on purpose: nothing else runs between reading the history
        and handing back its context. The current message is taken from its
        history record, so a regeneration answers the edited text;
        `message_text` is used when there is no record (`response:`, which
        passes from_history=False because its message_id belongs to another
        chat).
        """
        self.logger.info(f'llm_request: {chat_id} (tools_enabled={tools_enabled})')

        # Read chat history
        context = self.read_chat_history(chat_id=chat_id)
        if from_history:
            message_text = context.get(int(message_id), message_text)
        self.logger.info(f'invoking message_text: {message_text}')
        system_prompt = self.get_system_prompt(chat_id, tools_enabled=tools_enabled)

        # Build prompt with chat history context
        history_text = self.format_chat_history()
        user_prompt = f"chat_id: {chat_id}\nmessage_id: {message_id}\n\n"
        if history_text:
            user_prompt += f"Previous conversation:\n{history_text}\n\n"
        user_prompt += f"Current message:\n{message_text}"
        return system_prompt, user_prompt, context

    async def generate(self, system_prompt, user_prompt, chat_id, message_id,
                       tools_enabled=True, outbox=None):
        """The text to send: the model's answer, or an error message.

        Saves nothing: the caller saves the answer it actually sends. Files the
        tools produce go to `outbox`. Cancellation is not an error and passes
        through (only Exception is caught).
        """
        try:
            response = await self._claude_agent_query(
                system_prompt, user_prompt, chat_id=chat_id, message_id=message_id,
                tools_enabled=tools_enabled, outbox=outbox
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
                        '\n\nReminder: a tool only runs when you actually invoke it. '
                        'Writing a tool name like [run_command] as text does nothing. '
                        'Complete the request now by invoking the tool, then reply with text.'
                    )
                else:
                    retry_prompt = user_prompt + (
                        '\n\nReminder: you have no tools in this conversation. '
                        'Answer directly, in plain text.'
                    )
                response = await self._claude_agent_query(
                    system_prompt, retry_prompt, chat_id=chat_id, message_id=message_id,
                    tools_enabled=tools_enabled, outbox=outbox
                )
                response = TOOL_ARTIFACT_RE.sub('', response).strip()
                self.logger.info(f'retry response: {response[:200]}')

            return response

        except Exception as e:
            error_message = f"I encountered an error while processing your request. Please try again later."
            self.logger.error(f"Error in llm_request: {str(e)}", exc_info=True)
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
