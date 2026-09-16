"""P4-P7: edits rewrite the history; Janet checks for them before she sends (tests 1-9)."""
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from conftest import (ALICE, BOB, GROUP, OWNER, Background, message, park, records,
                      wait_until)

EDITED = 1758000100          # 2025-09-16 05:21:40 UTC


def edit(chat_id, message_id, text=None, **kwargs):
    return message(chat_id, message_id, text, edit_date=EDITED, **kwargs)


def sent(env, chat_id):
    return [e for e in env.events if e['chat_id'] == chat_id]


def current_message(call):
    return call.prompt.rsplit('Current message:\n', 1)[1]


def raw_text(call):
    return current_message(call).rsplit('message_text: ', 1)[1]


def test_edit_during_generation_regenerates_from_the_edited_text(env, client):
    """Test 1."""
    async def script(call):
        if call.index == 0:
            await park(call)
        return f'answer to: {raw_text(call)}'

    env.llm.script = script
    asking = Background(client, '/message', message(ALICE, 10, 'what is 2+2?'))
    env.llm.wait_for_calls(1)

    assert client.post('/edited_message', json=edit(ALICE, 10, 'what is 3+3?')).status_code == 200
    assert wait_until(lambda: env.llm.calls[0].cancelled, timeout=5), 'the edit did not cancel the attempt'
    asking.join()

    first, second = env.llm.calls
    assert raw_text(second) == 'what is 3+3?'
    assert 'what is 2+2?' not in second.prompt
    assert [e['text'] for e in sent(env, ALICE)] == ['answer to: what is 3+3?']
    history = records(env, ALICE)
    assert [(r['type'], r.get('raw_text', r['text'])) for r in history] == [
        ('HumanMessage', 'what is 3+3?'), ('AIMessage', 'answer to: what is 3+3?')]


def test_edit_after_the_answer_only_rewrites_the_history(env, client):
    """Test 2."""
    client.post('/message', json=message(ALICE, 10, 'first question'))
    client.post('/message', json=message(ALICE, 11, 'second question'))
    before = records(env, ALICE)
    assert [r['type'] for r in before] == ['HumanMessage', 'AIMessage'] * 2

    assert client.post('/edited_message', json=edit(ALICE, 10, 'first question, fixed')).status_code == 200

    after = records(env, ALICE)
    assert [r['_file'] for r in after] == [r['_file'] for r in before]
    record = after[0]
    assert record['raw_text'] == 'first question, fixed'
    assert record['edit_date'] == '2025-09-16 05:21:40'
    assert record['text'] == (
        'user_name: user5001\nchat_id: 5001\nmessage_id: 10\n'
        'message_date: 2025-09-16 05:20:00\nedit_date: 2025-09-16 05:21:40\n'
        'message_text: first question, fixed')
    assert after[1:] == before[1:]            # the answer it got stays
    assert len(env.llm.calls) == 2 and len(env.events) == 2

    client.post('/message', json=message(ALICE, 12, 'what did I ask first?'))
    assert 'message_text: first question, fixed' in env.llm.calls[2].prompt
    assert 'message_text: first question\n' not in env.llm.calls[2].prompt


def test_edit_outside_the_window_changes_nothing(env, client, monkeypatch):
    """Test 3: the edited message is not in the history the answer was given."""
    client.post('/message', json=message(ALICE, 10, 'an old, long message ' + 'x ' * 300))
    monkeypatch.setitem(env.server.panthera.config, 'token_limit', 60)

    async def script(call):
        if call.index == 1:
            await park(call)
        return 'the new answer'

    env.llm.script = script
    asking = Background(client, '/message', message(ALICE, 11, 'a new question'))
    call = env.llm.wait_for_calls(2)
    assert 'an old, long message' not in call.prompt

    client.post('/edited_message', json=edit(ALICE, 10, 'an old message, edited'))
    time.sleep(0.3)
    call.release.set()
    asking.join()

    assert len(env.llm.calls) == 2 and not call.cancelled
    assert [e['text'] for e in sent(env, ALICE)][-1] == 'the new answer'


def test_edit_of_a_message_newer_than_the_context_changes_nothing(env, client):
    """Test 3, the other way out of the window: a message sent during the generation."""
    async def script(call):
        await park(call)
        return 'answered'

    env.llm.script = script
    asking = Background(client, '/message', message(GROUP, 20, '/* question', sender=ALICE))
    call = env.llm.wait_for_calls(1)
    client.post('/message', json=message(GROUP, 21, 'chatter meanwhile', sender=BOB))
    client.post('/edited_message', json=edit(GROUP, 21, 'chatter, edited', sender=BOB))
    assert records(env, GROUP)[1]['raw_text'] == 'chatter, edited'

    time.sleep(0.3)
    call.release.set()
    asking.join()
    assert len(env.llm.calls) == 1 and not call.cancelled
    assert [e['text'] for e in sent(env, GROUP)] == ['answered']


def test_spurious_edit_changes_nothing(env, client):
    """Test 4: Telegram reports an edit, but the text and the photo are the same."""
    async def script(call):
        if call.index == 0:
            await park(call)
        return 'answered'

    env.llm.script = script
    photo_message = message(ALICE, 10, caption='look', photo='cat')
    asking = Background(client, '/message', photo_message)
    call = env.llm.wait_for_calls(1)
    before = records(env, ALICE)

    spurious = dict(photo_message, edit_date=EDITED)
    spurious['link_preview_options'] = {'is_disabled': True}
    client.post('/edited_message', json=spurious)
    client.post('/edited_message', json=edit(ALICE, 11, 'never saved'))   # not in the history
    time.sleep(0.3)
    call.release.set()
    asking.join()

    assert len(env.llm.calls) == 1 and not call.cancelled
    assert records(env, ALICE)[0] == before[0]


def test_changed_photo_counts_as_an_edit(env, client):
    client.post('/message', json=message(ALICE, 10, caption='look', photo='cat'))
    client.post('/edited_message', json=edit(ALICE, 10, caption='look', photo='dog'))
    record = records(env, ALICE)[0]
    assert record['images'] == ['/TESTTOKEN/photos/dog.jpg']
    assert record['file_unique_ids'] == ['u-dog']
    assert record['edit_date'] == '2025-09-16 05:21:40'


def test_regeneration_is_capped(env, client, caplog):
    """Test 5: an edit during every attempt; after 3 regenerations the 4th is sent."""
    caplog.set_level(logging.INFO)

    async def script(call):
        await park(call)
        return f'answer {call.index + 1} to: {raw_text(call)}'

    env.llm.script = script
    asking = Background(client, '/message', message(ALICE, 10, 'v0'))
    for n in range(1, 5):
        call = env.llm.wait_for_calls(n)
        client.post('/edited_message', json=edit(ALICE, 10, f'v{n}'))
        if n < 4:
            assert wait_until(lambda: call.cancelled, timeout=5), f'attempt {n} was not cancelled'
    time.sleep(0.3)
    assert not call.cancelled
    call.release.set()
    asking.join()

    calls = env.llm.calls
    assert len(calls) == 4
    assert [c.cancelled for c in calls] == [True, True, True, False]
    assert [raw_text(c) for c in calls] == ['v0', 'v1', 'v2', 'v3']
    assert [e['text'] for e in sent(env, ALICE)] == ['answer 4 to: v3']
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('3 regenerations, the cap' in w for w in warnings)
    assert any('sending it although message(s) [10] were edited' in w for w in warnings)
    assert records(env, ALICE)[0]['raw_text'] == 'v4'


def test_edit_between_generation_and_sending_is_caught_by_the_check(env, client, monkeypatch):
    """The pre-send check: an edit that lands after the attempt's last await."""
    edits = env.edits
    original = edits.Generation.run_attempt

    async def run_attempt(self, attempt):
        result = await original(self, attempt)
        if self.attempt == 1:
            # What an edit processed right after the attempt finished looks like.
            await asyncio_to_thread(client.post, '/edited_message', json=edit(ALICE, 10, 'late edit'))
        return result

    monkeypatch.setattr(edits.Generation, 'run_attempt', run_attempt)
    env.llm.script = None
    asking = Background(client, '/message', message(ALICE, 10, 'question'))
    asking.join()

    assert len(env.llm.calls) == 2
    assert not env.llm.calls[0].cancelled
    assert raw_text(env.llm.calls[1]) == 'late edit'
    assert [e['text'] for e in sent(env, ALICE)] == ['answer 2']


async def asyncio_to_thread(fn, *args, **kwargs):
    import asyncio
    return await asyncio.to_thread(fn, *args, **kwargs)


def gemini_drawing(env, monkeypatch, images):
    """A fake Gemini that returns the next of `images` on each call."""
    queue = list(images)

    class Models:
        def generate_content(self, model, contents, config):
            data = queue.pop(0)
            part = SimpleNamespace(inline_data=SimpleNamespace(data=data), text=None)
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    monkeypatch.setattr(env.tools_cli, 'genai_client',
                        lambda: (SimpleNamespace(models=Models()), 'fake-image-model'))


def test_outbox_of_a_cancelled_attempt_is_never_sent(env, client, monkeypatch):
    """Test 6."""
    gemini_drawing(env, monkeypatch, [b'\x89PNG stale', b'\x89PNG fresh'])
    queued = []

    async def script(call):
        tools = env.bot_tools.build_tools(call.chat_id, call.message_id, call.kwargs['outbox'])
        result = await tools['generate_image'].handler({'prompt': raw_text(call)})
        assert result['content'][0]['text'] == 'Image generated. It will be sent to the chat with your reply.'
        queued.append(list(call.kwargs['outbox']))
        if call.index == 0:
            await park(call)
        return 'here it is'

    env.llm.script = script
    asking = Background(client, '/message', message(ALICE, 10, 'draw a cat'))
    env.llm.wait_for_calls(1)
    assert wait_until(lambda: queued)
    assert env.events == []                       # held, not sent
    client.post('/edited_message', json=edit(ALICE, 10, 'draw a dog'))
    assert wait_until(lambda: env.llm.calls[0].cancelled, timeout=5), 'the edit did not cancel the attempt'
    asking.join()

    events = sent(env, ALICE)
    assert [e['kind'] for e in events] == ['photo', 'rich']
    assert events[0]['data'] == b'\x89PNG fresh'
    assert events[0]['caption'] == '||draw a dog||'
    assert events[0]['reply_to'] == 10 and events[1]['reply_to'] == 10
    cached = os.listdir(env.path / 'data' / 'users' / str(ALICE) / 'images')
    assert cached == ['photo-file-9001']


def test_send_file_and_render_math_go_out_before_the_text(env, client):
    (env.work / str(ALICE)).mkdir()
    (env.work / str(ALICE) / 'plot.png').write_bytes(b'\x89PNG plot')
    (env.work / str(ALICE) / 'table.csv').write_bytes(b'a,b\n1,2\n')

    async def script(call):
        tools = env.bot_tools.build_tools(call.chat_id, call.message_id, call.kwargs['outbox'])
        result = await tools['send_file'].handler({'path': 'plot.png', 'caption': 'the plot'})
        assert result['content'][0]['text'] == 'plot.png will be sent to the chat with your reply.'
        # Overwritten after send_file: the snapshot taken at call time is what goes out.
        (env.work / str(ALICE) / 'plot.png').write_bytes(b'\x89PNG overwritten')
        await tools['send_file'].handler({'path': 'table.csv'})
        await tools['render_math'].handler({'formula': r'x^2'})
        return 'done'

    env.llm.script = script
    client.post('/message', json=message(ALICE, 10, 'plot it'))

    events = sent(env, ALICE)
    assert [e['kind'] for e in events] == ['photo', 'document', 'photo', 'rich']
    assert events[0]['data'] == b'\x89PNG plot' and events[0]['caption'] == 'the plot'
    assert events[1]['name'] == 'table.csv' and events[1]['data'] == b'a,b\n1,2\n'
    assert events[2]['data'].startswith(b'\x89PNG')
    assert all(e['reply_to'] == 10 for e in events)


def test_burst_of_edits_restarts_once_with_the_last_text(env, client, monkeypatch):
    """Test 7."""
    monkeypatch.setattr(env.edits, 'DEBOUNCE_SECONDS', 0.6)
    starts = []

    async def script(call):
        starts.append(time.monotonic())
        if call.index == 0:
            await park(call)
        return f'answer to: {raw_text(call)}'

    env.llm.script = script
    asking = Background(client, '/message', message(ALICE, 10, 'first'))
    env.llm.wait_for_calls(1)
    client.post('/edited_message', json=edit(ALICE, 10, 'second'))
    assert wait_until(lambda: env.llm.calls[0].cancelled, timeout=5), 'the edit did not cancel the attempt'
    time.sleep(0.2)
    last_edit = time.monotonic()
    client.post('/edited_message', json=edit(ALICE, 10, 'third'))
    asking.join()

    assert len(env.llm.calls) == 2
    assert raw_text(env.llm.calls[1]) == 'third'
    assert starts[1] - last_edit >= 0.55
    assert [e['text'] for e in sent(env, ALICE)] == ['answer to: third']


def test_edit_never_summons_janet_or_runs_a_command(env, client):
    """Test 8."""
    client.post('/message', json=message(GROUP, 30, 'hello all', sender=ALICE))
    client.post('/edited_message', json=edit(GROUP, 30, '/* hello all', sender=ALICE))
    assert records(env, GROUP)[0]['raw_text'] == '/* hello all'

    client.post('/message', json=message(ALICE, 40, 'hi'))
    client.post('/edited_message', json=edit(ALICE, 40, '/reset'))
    history = records(env, ALICE)
    assert [r['type'] for r in history] == ['HumanMessage', 'AIMessage']
    assert history[0]['raw_text'] == '/reset'

    client.post('/message', json=message(ALICE, 41, '/memory'))      # a command: not saved
    client.post('/edited_message', json=edit(ALICE, 41, 'what do you remember?'))

    assert len(env.llm.calls) == 1
    assert [e['kind'] for e in env.events] == ['rich']
    assert len(records(env, ALICE)) == 2


def test_album_caption_edit_keeps_the_images(env, client, monkeypatch):
    """Test 9."""
    monkeypatch.setattr(env.server, 'MEDIA_GROUP_WAIT_SECONDS', 0.2)
    items = [message(OWNER, 61, caption='three photos', photo='p1', media_group_id='g1'),
             message(OWNER, 62, photo='p2', media_group_id='g1'),
             message(OWNER, 63, photo='p3', media_group_id='g1')]
    for item in items:
        client.post('/message', json=item)
    assert wait_until(lambda: env.events)
    images = ['/TESTTOKEN/photos/p1.jpg', '/TESTTOKEN/photos/p2.jpg', '/TESTTOKEN/photos/p3.jpg']

    # The captioned item is edited.
    client.post('/edited_message', json=dict(items[0], caption='three photos of cats', edit_date=EDITED))
    record = records(env, OWNER)[0]
    assert record['raw_text'] == 'three photos of cats'
    assert record['images'] == images
    assert record['captions'] == {'61': 'three photos of cats', '62': '', '63': ''}
    assert 'edit_date: 2025-09-16 05:21:40' in record['text']
    assert record['text'].endswith('message_text: three photos of cats')

    # Another item: a spurious edit changes nothing, a new caption is added.
    client.post('/edited_message', json=dict(items[1], edit_date=EDITED))
    assert records(env, OWNER)[0] == record
    client.post('/edited_message', json=dict(items[2], caption='the last one sleeps', edit_date=EDITED))
    record = records(env, OWNER)[0]
    assert record['raw_text'] == 'three photos of cats\nthe last one sleeps'
    assert record['images'] == images
    assert record['_file'].endswith('_61.json')

    assert len(env.llm.calls) == 1 and len(env.events) == 1


def test_album_edited_while_being_collected(env, client, monkeypatch):
    monkeypatch.setattr(env.server, 'MEDIA_GROUP_WAIT_SECONDS', 0.5)
    client.post('/message', json=message(OWNER, 71, caption='draft', photo='q1', media_group_id='g2'))
    client.post('/message', json=message(OWNER, 72, photo='q2', media_group_id='g2'))
    client.post('/edited_message', json=edit(OWNER, 71, caption='final', photo='q1', media_group_id='g2'))
    assert wait_until(lambda: env.events)
    assert raw_text(env.llm.calls[0]) == 'final'
    assert records(env, OWNER)[0]['captions'] == {'71': 'final', '72': ''}


def test_old_album_record_keeps_its_caption(env, client):
    """An album saved before captions were kept per item, filed under a captionless item."""
    folder = env.path / 'data' / 'users' / str(OWNER) / 'chats' / str(OWNER)
    folder.mkdir(parents=True)
    (folder / '2026-09-01-10-00-00_81.json').write_text(
        '{"type": "HumanMessage", "text": "user_name: x\\nmessage_id: 81\\n'
        'message_text: old caption", "images": ["/a.jpg", "/b.jpg"]}')
    client.post('/edited_message', json=edit(OWNER, 81, photo='r1', media_group_id='g3'))
    assert 'edit_date' not in records(env, OWNER)[0]
    client.post('/edited_message', json=edit(OWNER, 81, caption='new caption', photo='r1', media_group_id='g3'))
    record = records(env, OWNER)[0]
    assert record['raw_text'] == 'new caption' and record['images'] == ['/a.jpg', '/b.jpg']


def test_legacy_record_is_found_by_its_file_name(env, client):
    folder = env.path / 'data' / 'users' / str(ALICE) / 'chats' / str(ALICE)
    folder.mkdir(parents=True)
    path = folder / '2026-09-01-10-00-00_90.json'
    path.write_text('{"type": "HumanMessage", "text": "user_name: a\\nchat_id: 5001\\nmessage_id: 90\\n'
                    'message_date: x\\nmessage_text: legacy", "images": []}')
    os.utime(path, (1000, 1000))
    client.post('/edited_message', json=edit(ALICE, 90, 'legacy, edited'))
    client.post('/edited_message', json=edit(ALICE, 90, 'legacy, edited'))   # now the same
    record = records(env, ALICE)[0]
    assert record['raw_text'] == 'legacy, edited' and record['message_id'] == 90
    assert path.stat().st_mtime == 1000


def test_group_guest_edit_is_recorded_too(env, client):
    client.post('/message', json=message(GROUP, 50, 'guest says', sender=7001))
    client.post('/edited_message', json=edit(GROUP, 50, 'guest says, edited', sender=7001))
    assert records(env, GROUP)[0]['raw_text'] == 'guest says, edited'


def test_unauthorized_private_edit_is_ignored(env, client):
    client.post('/edited_message', json=edit(424242, 1, 'hello'))
    assert not (env.path / 'data' / 'users' / '424242' / 'chats').exists()


def test_response_answers_without_a_current_record(env, client):
    """`response:` names a group; its message_id is the private chat's."""
    client.post('/message', json=message(GROUP, 99, 'group message 99', sender=ALICE))
    client.post('/message', json=message(OWNER, 99, f'response:{GROUP}'))
    call = env.llm.calls[0]
    assert current_message(call) == ''
    assert 'group message 99' in call.prompt
    assert [(e['chat_id'], e['reply_to']) for e in env.events] == [(GROUP, None)]


def test_answers_run_at_most_two_at_a_time(env, client):
    running = []
    peak = []

    async def script(call):
        running.append(call.index)
        peak.append(len(running))
        await park(call)
        running.remove(call.index)
        return 'ok'

    env.llm.script = script
    asking = [Background(client, '/message', message(user, 10, 'hi')) for user in (ALICE, BOB, OWNER)]
    env.llm.wait_for_calls(2)
    time.sleep(0.3)
    assert len(env.llm.calls) == 2
    env.llm.calls[0].release.set()
    env.llm.wait_for_calls(3)
    for call in env.llm.calls:
        call.release.set()
    for request in asking:
        request.join()
    assert max(peak) == 2
    assert len(env.events) == 3


FAKE_CLI = Path(__file__).with_name('fake_claude.py')


def test_cancelling_an_attempt_ends_its_cli(env, client, monkeypatch, tmp_path):
    """P7, through the real SDK: the cancelled attempt's CLI process is gone."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    state = tmp_path / 'fake-cli'
    state.mkdir()
    wrapper = tmp_path / 'claude'
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE_CLI} "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv('FAKE_CLAUDE_DIR', str(state))
    monkeypatch.setenv('CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK', '1')
    monkeypatch.setattr(subprocess_cli.SubprocessCLITransport, '_find_cli', lambda self: str(wrapper))
    monkeypatch.delattr(env.server.panthera, '_claude_agent_query')   # the real one

    asking = Background(client, '/message', message(ALICE, 10, 'slow question'))
    assert wait_until(lambda: (state / '00.pid').exists(), timeout=15)
    pid = int((state / '00.pid').read_text())
    time.sleep(0.5)
    t0 = time.monotonic()
    client.post('/edited_message', json=edit(ALICE, 10, 'slow question, edited'))
    asking.join(timeout=60)

    assert not Path(f'/proc/{pid}').exists() or Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0] == 'Z'
    assert (state / 'answered').exists()
    assert [e['text'] for e in sent(env, ALICE)] == ['hello from the fake CLI']
    assert time.monotonic() - t0 < 30
