"""The head's in-process MCP tools — the bot's "head", holding every secret.

Layer 2 of the hardening plan: the model gets no Bash and no Read. Anything that
needs a secret (Telegram token, Gemini key, Wolfram id, the chat's data) runs
here, inside the FastAPI process, as an MCP tool. Anything that runs code the
model wrote goes to the sandbox container through `run_command`.

The server is rebuilt per request by :func:`create_bot_server`, with chat_id and
message_id captured in closures. The model therefore has no way to name another
chat: there is no chat_id parameter on any tool.
"""
import base64
import mimetypes
import os

import httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

import memory
import tools_cli

SANDBOX_SOCKET = os.environ.get('SANDBOX_SOCKET', '/run/sandbox/exec.sock')
SANDBOX_TOKEN = os.environ.get('SANDBOX_TOKEN', '')
WORK_ROOT = os.environ.get('SANDBOX_WORK_ROOT', '/work')

# Telegram's local file store, mounted read-only. Empty when unset, which just
# means view_image/send_file only accept the chat's scratch directory.
TELEGRAM_FILES_ROOT = (
    f"/{os.environ['BOT_TOKEN_WITHOUT_PREFIX']}"
    if os.environ.get('BOT_TOKEN_WITHOUT_PREFIX') else ''
)

# Max image bytes handed to the model as a base64 block (~5 MB of file).
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_SEND_BYTES = 45 * 1024 * 1024

TOOL_NAMES = [
    'run_command',
    'view_image',
    'send_file',
    'generate_image',
    'wolfram_alpha',
    'render_math',
    'remember',
    'forget',
    'replace_memory',
    'update_system_prompt',
    'reset_system_prompt',
]


def _text(message):
    return {'content': [{'type': 'text', 'text': str(message)}]}


def _error(message):
    return {'content': [{'type': 'text', 'text': str(message)}], 'is_error': True}


def work_dir(chat_id):
    return os.path.join(WORK_ROOT, str(chat_id))


def _resolve(path, chat_id):
    """Resolve a model-supplied path, or raise if it is outside what it may touch.

    Two roots only: Telegram's file store (photos the user sent) and this chat's
    scratch directory in the sandbox volume. Symlinks are resolved first, so a
    link planted from inside the sandbox cannot point out of them.
    """
    roots = [r for r in (TELEGRAM_FILES_ROOT, work_dir(chat_id)) if r]
    real = os.path.realpath(path)
    for root in roots:
        root = os.path.realpath(root)
        if real == root or real.startswith(root + os.sep):
            return real
    raise PermissionError(
        f'path is outside this chat\'s files: {path} (allowed: {", ".join(roots)})'
    )


async def sandbox_exec(chat_id, command, timeout):
    """POST to the sandbox over its Unix socket. It has no network of its own."""
    transport = httpx.AsyncHTTPTransport(uds=SANDBOX_SOCKET)
    async with httpx.AsyncClient(
        transport=transport, base_url='http://sandbox', timeout=timeout + 15
    ) as client:
        resp = await client.post(
            '/exec',
            json={'chat_id': str(chat_id), 'command': command, 'timeout': timeout},
            headers={'X-Sandbox-Token': SANDBOX_TOKEN},
        )
        resp.raise_for_status()
        return resp.json()


def create_bot_server(chat_id, message_id):
    """Build the per-request MCP server. chat_id/message_id live in closures."""
    chat_id = str(chat_id)
    message_id = str(message_id)

    @tool(
        'run_command',
        'Run a shell command (bash) in the sandbox and return its output. Use it '
        'to execute Python or any other code. The sandbox has python3 with '
        'pandas, numpy, matplotlib and pillow, a per-chat working directory, and '
        'read-only access to files the user sent. It has NO network access and '
        'cannot reach the bot\'s data or credentials. Files you write to the '
        'working directory can be delivered with send_file.',
        {
            'type': 'object',
            'properties': {
                'command': {'type': 'string', 'description': 'Shell command to run.'},
                'timeout': {
                    'type': 'integer',
                    'description': 'Seconds before the command is killed (max 120, default 60).',
                },
            },
            'required': ['command'],
        },
    )
    async def run_command(args):
        try:
            result = await sandbox_exec(chat_id, args['command'], int(args.get('timeout') or 60))
        except Exception as e:
            return _error(f'sandbox unavailable: {e}')

        parts = [f"rc={result['rc']}" + (' (timed out)' if result.get('timed_out') else '')]
        if result.get('stdout'):
            parts.append('stdout:\n' + result['stdout'])
        if result.get('stderr'):
            parts.append('stderr:\n' + result['stderr'])
        if not result.get('stdout') and not result.get('stderr'):
            parts.append('(no output)')
        return _text('\n'.join(parts))

    @tool(
        'view_image',
        'Look at an image file — a photo the user sent (paths appear in file_list) '
        'or one produced in the sandbox working directory. Returns the image itself.',
        {'path': str},
    )
    async def view_image(args):
        try:
            path = _resolve(args['path'], chat_id)
        except PermissionError as e:
            return _error(str(e))
        if not os.path.isfile(path):
            return _error(f'no such file: {args["path"]}')
        size = os.path.getsize(path)
        if size > MAX_IMAGE_BYTES:
            return _error(f'image is too large to view ({size} bytes, limit {MAX_IMAGE_BYTES})')

        mime, _ = mimetypes.guess_type(path)
        if not mime or not mime.startswith('image/'):
            mime = 'image/jpeg'
        with open(path, 'rb') as f:
            data = base64.standard_b64encode(f.read()).decode('ascii')
        return {'content': [{'type': 'image', 'data': data, 'mimeType': mime}]}

    @tool(
        'send_file',
        'Send a file from the sandbox working directory to the chat — a plot, a '
        'CSV, a rendered document. Images are sent as photos, everything else as '
        'a document.',
        {
            'type': 'object',
            'properties': {
                'path': {'type': 'string', 'description': 'Path in the working directory.'},
                'caption': {'type': 'string', 'description': 'Optional caption.'},
            },
            'required': ['path'],
        },
    )
    async def send_file(args):
        try:
            path = _resolve(args['path'], chat_id)
        except PermissionError as e:
            return _error(str(e))
        if not os.path.isfile(path):
            return _error(f'no such file: {args["path"]}')
        size = os.path.getsize(path)
        if size > MAX_SEND_BYTES:
            return _error(f'file is too large to send ({size} bytes)')

        caption = (args.get('caption') or '')[:1000] or None
        mime, _ = mimetypes.guess_type(path)
        try:
            with open(path, 'rb') as f:
                if mime and mime.startswith('image/'):
                    tools_cli.bot.send_photo(
                        chat_id=int(chat_id), photo=f,
                        reply_to_message_id=int(message_id), caption=caption,
                    )
                else:
                    tools_cli.bot.send_document(
                        chat_id=int(chat_id), document=f,
                        reply_to_message_id=int(message_id), caption=caption,
                    )
        except Exception as e:
            return _error(f'could not send the file: {e}')
        return _text(f'Sent {os.path.basename(path)} to the chat.')

    @tool(
        'generate_image',
        'Generate an image with Gemini and send it to the chat. Use whenever the '
        'user asks to generate, create, draw or edit an image.',
        {
            'type': 'object',
            'properties': {
                'prompt': {'type': 'string', 'description': 'What to draw.'},
                'file_list': {
                    'type': 'array', 'items': {'type': 'string'},
                    'description': 'Optional input images to edit or compose.',
                },
            },
            'required': ['prompt'],
        },
    )
    async def generate_image(args):
        files = args.get('file_list') or []
        try:
            files = [_resolve(p, chat_id) for p in files]
        except PermissionError as e:
            return _error(str(e))
        return _text(await tools_cli.generate_image(
            prompt=args['prompt'], chat_id=chat_id, message_id=message_id,
            file_list=files or None,
        ))

    @tool(
        'wolfram_alpha',
        'Ask Wolfram|Alpha — math, science, unit conversions, equations, factual lookups.',
        {'query': str},
    )
    async def wolfram_alpha(args):
        return _text(await tools_cli.wolfram_alpha(query=args['query']))

    @tool(
        'render_math',
        'Render a LaTeX formula as a PNG and send it to the chat. Rarely needed — '
        'Telegram renders $...$ and $$...$$ natively in your replies.',
        {'formula': str},
    )
    async def render_math(args):
        return _text(await tools_cli.render_math(
            formula=args['formula'], chat_id=chat_id, message_id=message_id,
        ))

    @tool(
        'remember',
        'Save one short, self-contained note in this chat\'s long-term memory. Use '
        'it when the user asks you to remember, save or keep something in mind. '
        'Notes survive /reset.',
        {'note': str},
    )
    async def remember(args):
        try:
            return _text(memory.remember(chat_id, args['note']))
        except memory.MemoryError_ as e:
            return _error(str(e))

    @tool(
        'forget',
        'Drop notes from this chat\'s memory. Pass the text to match, or "*" to '
        'clear everything.',
        {'pattern': str},
    )
    async def forget(args):
        try:
            return _text(memory.forget(chat_id, args['pattern']))
        except memory.MemoryError_ as e:
            return _error(str(e))

    @tool(
        'replace_memory',
        'Rewrite this chat\'s whole memory, one note per line. Use it to condense '
        'an oversized memory, after the user agrees.',
        {'content': str},
    )
    async def replace_memory(args):
        try:
            return _text(memory.replace_memory(chat_id, args['content']))
        except memory.MemoryError_ as e:
            return _error(str(e))

    @tool(
        'update_system_prompt',
        'Replace this chat\'s system prompt. Only when the user explicitly asks to '
        'change how you behave in this chat.',
        {'new_prompt': str},
    )
    async def update_system_prompt(args):
        return _text(await tools_cli.update_system_prompt(
            chat_id=chat_id, new_prompt=args['new_prompt'],
        ))

    @tool(
        'reset_system_prompt',
        'Restore this chat\'s default system prompt.',
        {'type': 'object', 'properties': {}, 'required': []},
    )
    async def reset_system_prompt(args):
        return _text(await tools_cli.reset_system_prompt(chat_id=chat_id))

    return create_sdk_mcp_server(name='bot', version='1.0.0', tools=[
        run_command, view_image, send_file, generate_image, wolfram_alpha,
        render_math, remember, forget, replace_memory,
        update_system_prompt, reset_system_prompt,
    ])
