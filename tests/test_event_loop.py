"""P0: generating an image must not freeze the other chats (test 11)."""
import threading
import time
from types import SimpleNamespace

from conftest import ALICE, BOB, Background, message

PNG = b'\x89PNG\r\n\x1a\n fake image'


def fake_gemini(env, monkeypatch, seconds, started):
    """Replace the Gemini client with one whose call blocks its thread."""

    class Models:
        def generate_content(self, model, contents, config):
            started.set()
            time.sleep(seconds)
            part = SimpleNamespace(inline_data=SimpleNamespace(data=PNG), text=None)
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    monkeypatch.setattr(env.tools_cli, 'genai_client',
                        lambda: (SimpleNamespace(models=Models()), 'fake-image-model'))


def test_another_chat_is_served_while_an_image_is_generated(env, client, monkeypatch):
    started = threading.Event()
    fake_gemini(env, monkeypatch, 5, started)

    async def script(call):
        if call.chat_id == ALICE:
            result = await env.tools_cli.generate_image(
                prompt='a cat', chat_id=call.chat_id, message_id=call.message_id)
            return 'here is your cat: ' + result
        return 'pong'

    env.llm.script = script
    drawing = Background(client, '/message', message(ALICE, 10, 'draw a cat'))
    assert started.wait(5), 'the image call never started'

    t0 = time.monotonic()
    response = client.post('/message', json=message(BOB, 20, 'ping'))
    elapsed = time.monotonic() - t0

    assert response.status_code == 200
    assert elapsed < 2, f'the other chat waited {elapsed:.1f}s behind the image'
    assert [e['text'] for e in env.events if e['chat_id'] == BOB] == ['pong']

    drawing.join()
    alice = [e for e in env.events if e['chat_id'] == ALICE]
    assert [e['kind'] for e in alice] == ['photo', 'rich']
    assert alice[0]['data'] == PNG
