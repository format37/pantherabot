"""Per-chat long-term memory.

One file per chat: ``data/users/{chat_id}/CLAUDE.md``, one dated note per line.

It deliberately sits in the chat's *root* folder and not in ``chats/{chat_id}/``:
everything in that folder is treated as a chat-history record by
``Panthera.read_chat_history()`` and wiped by ``Panthera.reset_chat()``. Memory
survives ``/reset`` by design — ``/reset`` clears the conversation, not what the
user asked to keep.

The module is the single implementation of the read/write logic. It is exposed to
the model through ``tools_cli.py`` today and will be wrapped as MCP tools when the
head/hands split lands; nothing here assumes either transport.
"""
import os
import re
import time

# All paths are relative to the process cwd (/server in the container), matching
# panthera.py and tools_cli.py. Overridable for tests.
DATA_DIR = os.environ.get('PANTHERA_DATA_DIR', 'data')

MEMORY_FILENAME = 'CLAUDE.md'

# How much of the file is injected into the system prompt.
MAX_PROMPT_BYTES = 8192
# Refuse to grow the file past this; the model is told to condense instead.
MAX_FILE_BYTES = 65536
MAX_NOTE_CHARS = 2000

_CHAT_ID_RE = re.compile(r'^-?\d+$')

_PROMPT_HEADER = """

## Memory
The following notes were saved earlier at the user's request. They are authoritative over the
conversation history. Use them without being asked; mention that you are using a saved note only
when it matters.
"""

_OVERSIZE_NOTICE = (
    "\n(Only the most recent notes are shown — the memory file is larger than the limit. "
    "Offer to condense it with replace_memory.)\n"
)


class MemoryError_(ValueError):
    """Invalid request from the model (bad chat_id, empty note, file too large)."""


def validate_chat_id(chat_id):
    """Return chat_id as a safe path component, or raise.

    Chat ids are integers (negative for groups). Anything else would let a tool
    argument escape data/users/.
    """
    value = str(chat_id).strip()
    if not _CHAT_ID_RE.match(value):
        raise MemoryError_(f'invalid chat_id: {chat_id!r}')
    return value


def memory_path(chat_id):
    return os.path.join(DATA_DIR, 'users', validate_chat_id(chat_id), MEMORY_FILENAME)


def _header(chat_id):
    return f'# Memory for chat {chat_id}\n\n'


def _read_lines(chat_id):
    """Return (header, note_lines). Missing file -> default header, no notes."""
    path = memory_path(chat_id)
    if not os.path.exists(path):
        return _header(chat_id), []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()
    notes = [line for line in lines if line.strip().startswith('- ')]
    return _header(chat_id), notes


def _write(chat_id, notes):
    path = memory_path(chat_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = _header(chat_id) + ''.join(f'{line}\n' for line in notes)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)
    return len(notes)


def load(chat_id):
    """Raw file content, or '' when nothing is saved.

    A file whose notes have all been forgotten keeps its header; that still
    counts as empty, both for /memory and for the system prompt.
    """
    path = memory_path(chat_id)
    if not os.path.exists(path):
        return ''
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if not any(line.strip().startswith('- ') for line in content.splitlines()):
        return ''
    return content.strip()


def render_for_prompt(chat_id):
    """The '## Memory' block to append to the system prompt, or '' when empty."""
    try:
        content = load(chat_id)
    except MemoryError_:
        return ''
    if not content:
        return ''

    encoded = content.encode('utf-8')
    if len(encoded) <= MAX_PROMPT_BYTES:
        return _PROMPT_HEADER + content + '\n'

    # Keep the tail (newest notes) and cut at a line boundary.
    tail = encoded[-MAX_PROMPT_BYTES:].decode('utf-8', errors='ignore')
    newline = tail.find('\n')
    if newline != -1:
        tail = tail[newline + 1:]
    return _PROMPT_HEADER + tail.strip() + '\n' + _OVERSIZE_NOTICE


def remember(chat_id, note):
    """Append one dated note. Returns a short status line for the model."""
    chat_id = validate_chat_id(chat_id)
    note = ' '.join(str(note).split())
    if not note:
        raise MemoryError_('note is empty')
    if len(note) > MAX_NOTE_CHARS:
        raise MemoryError_(f'note is too long ({len(note)} chars, limit {MAX_NOTE_CHARS})')

    header, notes = _read_lines(chat_id)
    if sum(len(line) + 1 for line in notes) + len(header) + len(note) > MAX_FILE_BYTES:
        raise MemoryError_(
            'memory file is full; condense it with replace_memory or drop notes with forget'
        )

    line = f'- {time.strftime("%Y-%m-%d")}: {note}'
    existing = {n.split(': ', 1)[-1].strip().lower() for n in notes}
    if note.lower() in existing:
        return f'Already saved. Memory has {len(notes)} note(s).'

    notes.append(line)
    count = _write(chat_id, notes)
    return f'Saved. Memory now has {count} note(s).'


def forget(chat_id, pattern):
    """Drop notes containing `pattern` (case-insensitive). '*' clears everything."""
    chat_id = validate_chat_id(chat_id)
    pattern = str(pattern).strip()
    if not pattern:
        raise MemoryError_('pattern is empty')

    _, notes = _read_lines(chat_id)
    if not notes:
        return 'Memory is already empty.'

    if pattern == '*':
        _write(chat_id, [])
        return f'Memory cleared ({len(notes)} note(s) removed).'

    needle = pattern.lower()
    kept = [line for line in notes if needle not in line.lower()]
    removed = len(notes) - len(kept)
    if not removed:
        return f'No note matched {pattern!r}; memory unchanged ({len(notes)} note(s)).'
    _write(chat_id, kept)
    return f'Removed {removed} note(s). Memory now has {len(kept)} note(s).'


def replace_memory(chat_id, content):
    """Rewrite the whole file, for condensing an oversized memory."""
    chat_id = validate_chat_id(chat_id)
    content = str(content).strip()
    if len(content.encode('utf-8')) > MAX_FILE_BYTES:
        raise MemoryError_(f'content is larger than {MAX_FILE_BYTES} bytes')

    notes = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        notes.append(line if line.startswith('- ') else f'- {line}')

    count = _write(chat_id, notes)
    return f'Memory replaced. It now has {count} note(s).'
