"""P1-P3: history order, record fields, and the shared human-record builder."""
import json
import os
import time

from conftest import ALICE, GROUP, OWNER, message, records, wait_until


def write_record(env, chat_id, name, record, mtime):
    folder = env.path / 'data' / 'users' / str(chat_id) / 'chats' / str(chat_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(record))
    os.utime(path, (mtime, mtime))
    return path


def test_order_survives_chmod_and_rewrite(env):
    """Test 10: history order is mtime, which chmod and rewrite_record leave alone."""
    panthera = env.server.panthera
    base = time.time() - 3600
    paths = []
    for i, text in enumerate(['first', 'second', 'third']):
        paths.append(write_record(
            env, ALICE, f'2026-01-0{i + 1}-00-00-00_{10 + i}.json',
            {'type': 'HumanMessage' if i != 1 else 'AIMessage', 'text': text, 'images': []},
            base + i))

    # What `chown -R` did on 2026-03-20: a fresh ctime, newest for the oldest file.
    for path in reversed(paths):
        os.chmod(path, 0o644)
        time.sleep(0.02)
    assert paths[0].stat().st_ctime_ns > paths[2].stat().st_ctime_ns

    context = panthera.read_chat_history(ALICE)
    assert [m['content'] for m in panthera.chat_history] == ['first', 'second', 'third']
    assert context == {10: 'first', 12: 'third'}

    mtime = paths[0].stat().st_mtime_ns
    panthera.rewrite_record(str(paths[0]), {'type': 'HumanMessage', 'text': 'first, edited', 'images': []})
    assert paths[0].stat().st_mtime_ns == mtime
    assert not [p for p in paths[0].parent.iterdir() if p.suffix != '.json']

    panthera.read_chat_history(ALICE)
    assert [m['content'] for m in panthera.chat_history] == ['first, edited', 'second', 'third']


def test_a_chat_without_history_gets_an_empty_one(env):
    panthera = env.server.panthera
    write_record(env, ALICE, '2026-01-01-00-00-00_1.json',
                 {'type': 'HumanMessage', 'text': 'private', 'images': []}, time.time())
    panthera.read_chat_history(ALICE)
    assert panthera.chat_history

    # Before: the previous chat's history stayed loaded (the `response:` path).
    assert panthera.read_chat_history(GROUP) == {}
    assert panthera.chat_history == []


def test_legacy_records_still_yield_id_and_text():
    import panthera as panthera_module
    legacy = {'type': 'HumanMessage',
              'text': 'user_name: A\nchat_id: 1\nmessage_id: 77\nmessage_date: x\nmessage_text: hi\nthere',
              'images': []}
    assert panthera_module.record_message_id('/d/2026-01-01-00-00-00_77.json', legacy) == 77
    assert panthera_module.record_raw_text(legacy) == 'hi\nthere'
    photo_only = {'type': 'HumanMessage', 'text': 'user_name: A\nfile_list: [1]', 'images': []}
    assert panthera_module.record_raw_text(photo_only) == ''


def test_message_record_carries_id_raw_text_and_files(env, client):
    replied = message(ALICE, 40, photo='replied-photo')
    msg = message(ALICE, 41, caption='what is this?\nsecond line', photo='own-photo', reply_to=replied)
    assert client.post('/message', json=msg).status_code == 200

    # The answer came within the same second, and still both records are there.
    both = records(env, ALICE)
    assert [(r['type'], r['_file'][-8:]) for r in both] == [('HumanMessage', '_41.json'), ('AIMessage', '_41.json')]
    record = both[0]
    assert record['message_id'] == 41
    assert record['raw_text'] == 'what is this?\nsecond line'
    assert record['images'] == ['/TESTTOKEN/photos/own-photo.jpg', '/TESTTOKEN/photos/replied-photo.jpg']
    assert record['file_unique_ids'] == ['u-own-photo', 'u-replied-photo']
    assert 'edit_date' not in record
    assert record['text'] == (
        'user_name: user5001\nchat_id: 5001\nmessage_id: 41\nreply_to_message: 40\n'
        'message_date: 2025-09-16 05:20:00\n'
        "file_list: ['/TESTTOKEN/photos/own-photo.jpg', '/TESTTOKEN/photos/replied-photo.jpg']\n"
        'message_text: what is this?\nsecond line')
    # The prompt's current message is the record's text.
    assert env.llm.calls[0].prompt.endswith('Current message:\n' + record['text'])


def test_album_is_filed_under_its_first_item(env, client, monkeypatch):
    monkeypatch.setattr(env.server, 'MEDIA_GROUP_WAIT_SECONDS', 0.2)
    # The relay may deliver the items in any order.
    for msg in (message(OWNER, 62, photo='p2', media_group_id='album1'),
                message(OWNER, 61, photo='p1', caption='/* compare these', media_group_id='album1'),
                message(OWNER, 63, photo='p3', media_group_id='album1')):
        assert client.post('/message', json=msg).status_code == 200

    assert wait_until(lambda: any(e['kind'] == 'rich' for e in env.events))
    human = [r for r in records(env, OWNER) if r['type'] == 'HumanMessage']
    assert len(human) == 1
    record = human[0]
    assert record['_file'].endswith('_61.json')
    assert record['message_id'] == 61
    assert record['media_group_id'] == 'album1'
    assert record['captions'] == {'61': '/* compare these', '62': '', '63': ''}
    assert record['raw_text'] == '/* compare these'
    assert record['images'] == ['/TESTTOKEN/photos/p1.jpg', '/TESTTOKEN/photos/p2.jpg',
                                '/TESTTOKEN/photos/p3.jpg']
    assert 'message_id: 61\n' in record['text']
    assert [e['reply_to'] for e in env.events] == [61]
    assert env.server.media_group_buffers == {}


def test_group_message_is_saved_but_not_answered(env, client):
    msg = message(GROUP, 5, 'just chatting', sender=ALICE)
    assert client.post('/message', json=msg).status_code == 200
    assert [r['raw_text'] for r in records(env, GROUP)] == ['just chatting']
    assert env.llm.calls == [] and env.events == []
