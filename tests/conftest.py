"""Test harness: the real FastAPI app with a fake Telegram bot and a fake LLM.

No Docker, no Telegram, no Claude. The server modules open cwd-relative files
(config.json, data/...) when imported and on every request, so each test runs
in its own temp directory seeded with copies of the tracked ones.

Run from the repo root with a venv that has server/requirements.txt and pytest:
    python -m pytest tests -q
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVER = REPO / 'server'

OWNER = 106129214            # the id in the tracked data/users.txt
ALICE = 5001                 # extra authorized users for private chats
BOB = 5002
GUEST = 7001                 # a group member who is not in users.txt
GROUP = -1009001
AUTHORIZED = [OWNER, ALICE, BOB]


def seed_workdir(path):
    """Copy the tracked files the server reads into `path`."""
    path = Path(path)
    (path / 'data' / 'users').mkdir(parents=True, exist_ok=True)
    config = json.loads((REPO / 'config.json').read_text())
    config['TOKEN'] = '123456:TEST-TOKEN'   # telebot insists on the id:secret shape
    (path / 'config.json').write_text(json.dumps(config))
    shutil.copy(REPO / 'data' / 'users' / 'default.json', path / 'data' / 'users' / 'default.json')
    shutil.copy(REPO / 'data' / 'admins.txt', path / 'data' / 'admins.txt')
    (path / 'data' / 'users.txt').write_text('\n'.join(str(u) for u in AUTHORIZED))
    (path / 'data' / 'granted_groups').mkdir()
    (path / 'data' / 'granted_groups' / f'{GROUP}.txt').write_text(str(GROUP))


_boot = tempfile.mkdtemp(prefix='panthera-tests-')
seed_workdir(_boot)
_cwd = os.getcwd()
os.chdir(_boot)
sys.path.insert(0, str(SERVER))
try:
    import server as server_module  # noqa: E402
    import tools_cli  # noqa: E402
    import bot_tools  # noqa: E402
finally:
    os.chdir(_cwd)


class FakeBot:
    """Records every Telegram call in `events`, in order."""

    token = 'TEST-TOKEN'

    def __init__(self, events):
        self.events = events
        self._next_id = 9000
        self._lock = threading.Lock()

    def _sent(self, kind, chat_id, **fields):
        with self._lock:
            self._next_id += 1
            message_id = self._next_id
        self.events.append(dict(kind=kind, chat_id=int(chat_id), **fields))
        photo = [SimpleNamespace(file_id=f'photo-file-{message_id}')]
        return SimpleNamespace(message_id=message_id, photo=photo)

    @staticmethod
    def _reply_to(kwargs):
        params = kwargs.get('reply_parameters')
        if params is not None:
            return params.message_id
        return kwargs.get('reply_to_message_id')

    @staticmethod
    def _read(payload):
        if hasattr(payload, 'read'):
            return payload.read()
        return payload

    def send_message(self, chat_id, text, **kwargs):
        return self._sent('message', chat_id, text=text, reply_to=self._reply_to(kwargs))

    def send_photo(self, chat_id, photo, **kwargs):
        return self._sent('photo', chat_id, data=self._read(photo),
                          caption=kwargs.get('caption'), reply_to=self._reply_to(kwargs))

    def send_document(self, chat_id, document, **kwargs):
        name = kwargs.get('visible_file_name') or getattr(document, 'name', None)
        return self._sent('document', chat_id, data=self._read(document), name=name,
                          caption=kwargs.get('caption'), reply_to=self._reply_to(kwargs))

    def get_file(self, file_id):
        return SimpleNamespace(file_path=f'/12345:TESTTOKEN/photos/{file_id}.jpg')

    def get_chat_member(self, chat_id, user_id):
        raise RuntimeError('not a member')


class FakeLLM:
    """Stands in for Panthera._claude_agent_query.

    `script(call)` decides what an attempt does; by default it answers at once.
    Each call is recorded with its prompt. `started` is set per call index, and
    a script can park on `call.release` (a threading.Event) to simulate a long
    generation — the event loop stays free while it waits.
    """

    def __init__(self):
        self.calls = []
        self.script = None
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)

    async def __call__(self, system_prompt, user_prompt, chat_id=None, message_id=None,
                       tools_enabled=True, **kwargs):
        with self.lock:
            call = SimpleNamespace(
                index=len(self.calls), system_prompt=system_prompt, prompt=user_prompt,
                chat_id=chat_id, message_id=message_id, tools_enabled=tools_enabled,
                kwargs=kwargs, release=threading.Event(), cancelled=False, finished=False,
            )
            self.calls.append(call)
            self.changed.notify_all()
        try:
            if self.script is None:
                return f'answer {call.index + 1}'
            return await self.script(call)
        except asyncio.CancelledError:
            call.cancelled = True
            raise
        finally:
            call.finished = True
            with self.lock:
                self.changed.notify_all()

    def wait_for_calls(self, n, timeout=10):
        with self.lock:
            ok = self.changed.wait_for(lambda: len(self.calls) >= n, timeout)
        assert ok, f'expected {n} LLM calls, saw {len(self.calls)}'
        return self.calls[n - 1]


async def park(call, timeout=30):
    """Block this attempt until the test releases it (or it is cancelled)."""
    deadline = time.monotonic() + timeout
    while not call.release.is_set():
        if time.monotonic() > deadline:
            raise AssertionError(f'LLM call {call.index} was never released')
        await asyncio.sleep(0.01)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fresh work dir, a fake bot and a fake LLM around the real app."""
    seed_workdir(tmp_path)
    monkeypatch.chdir(tmp_path)
    events = []
    bot = FakeBot(events)
    llm = FakeLLM()
    monkeypatch.setattr(server_module, 'bot', bot)
    monkeypatch.setattr(tools_cli, 'bot', bot)
    monkeypatch.setattr(server_module.panthera, '_claude_agent_query', llm)
    monkeypatch.setitem(server_module.panthera.config, 'token_limit', 50000)
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.setattr(bot_tools, 'WORK_ROOT', str(work))

    def rich(chat_id, markdown_text, reply_to=None):
        events.append(dict(kind='rich', chat_id=int(chat_id), text=markdown_text, reply_to=reply_to))
        return True

    monkeypatch.setattr(server_module, 'send_rich_message', rich)
    return SimpleNamespace(
        server=server_module, tools_cli=tools_cli, bot_tools=bot_tools,
        events=events, bot=bot, llm=llm, path=tmp_path, work=work,
    )


@pytest.fixture
def client(env):
    from fastapi.testclient import TestClient
    # One portal, so every request runs on the same event loop, as in uvicorn.
    with TestClient(env.server.app) as c:
        yield c


class Background:
    """POST in a thread, for a request that must stay in flight."""

    def __init__(self, client, path, payload):
        self.response = None
        self.error = None
        self.thread = threading.Thread(target=self._run, args=(client, path, payload), daemon=True)
        self.thread.start()

    def _run(self, client, path, payload):
        try:
            self.response = client.post(path, json=payload)
        except BaseException as e:  # surfaced by join()
            self.error = e

    def join(self, timeout=30):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), 'request did not finish'
        if self.error:
            raise self.error
        assert self.response.status_code == 200, self.response.text
        return self.response


def message(chat_id, message_id, text=None, *, sender=None, chat_type=None, date=1758000000,
            caption=None, photo=None, media_group_id=None, reply_to=None, edit_date=None):
    """A Telegram message dict as the relay forwards it."""
    sender = sender if sender is not None else chat_id
    chat_type = chat_type or ('private' if chat_id > 0 else 'supergroup')
    chat = {'id': chat_id, 'type': chat_type}
    if chat_type == 'private':
        chat['first_name'] = f'user{sender}'
    else:
        chat['title'] = 'Test group'
    msg = {
        'message_id': message_id,
        'from': {'id': sender, 'is_bot': False, 'first_name': f'user{sender}'},
        'chat': chat,
        'date': date,
    }
    if text is not None:
        msg['text'] = text
    if caption is not None:
        msg['caption'] = caption
    if photo is not None:
        msg['photo'] = [{'file_id': f'{photo}-small', 'file_unique_id': f'{photo}-s',
                         'width': 90, 'height': 90},
                        {'file_id': photo, 'file_unique_id': f'u-{photo}',
                         'width': 1280, 'height': 960}]
    if media_group_id is not None:
        msg['media_group_id'] = media_group_id
    if reply_to is not None:
        msg['reply_to_message'] = reply_to
    if edit_date is not None:
        msg['edit_date'] = edit_date
    return msg


def history_dir(env, chat_id):
    return env.path / 'data' / 'users' / str(chat_id) / 'chats' / str(chat_id)


def records(env, chat_id):
    """History records in history order (mtime), with their file names."""
    folder = history_dir(env, chat_id)
    if not folder.exists():
        return []
    files = sorted(folder.glob('*.json'), key=lambda p: (p.stat().st_mtime_ns, p.name))
    return [dict(json.loads(p.read_text()), _file=p.name) for p in files]


def wait_until(predicate, timeout=10, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
