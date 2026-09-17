"""Bot-to-bot loop guard: another bot summons Janet only by a reply, a few times in a row."""
import pytest

from conftest import GROUP, GUEST, OWNER, message, records

OTHER_BOT = 8001
JANET = {'message_id': 1, 'from': {'id': 1, 'is_bot': True, 'username': 'your_bot_name'}}


@pytest.fixture(autouse=True)
def fresh_guard(env, monkeypatch):
    monkeypatch.setattr(env.server, 'bot_reply_chain', {})
    monkeypatch.setattr(env.server, 'bot_reply_times', {})


def from_bot(message_id, text, reply=True):
    msg = message(GROUP, message_id, text, sender=OTHER_BOT, reply_to=JANET if reply else None)
    msg['from']['is_bot'] = True
    return msg


def test_a_bot_summons_only_by_reply(env, client):
    client.post('/message', json=from_bot(10, '/* do something', reply=False))
    client.post('/message', json=from_bot(11, '/help', reply=False))
    assert env.llm.calls == [] and env.events == []
    # Both are history all the same.
    assert [r['message_id'] for r in records(env, GROUP)] == [10, 11]

    client.post('/message', json=from_bot(12, 'and what about this?'))
    assert len(env.llm.calls) == 1


def test_the_chain_is_limited_and_a_human_restarts_it(env, client):
    limit = env.server.BOT_REPLY_CHAIN_LIMIT
    for i in range(limit + 2):
        client.post('/message', json=from_bot(20 + i, f'reply {i}'))
    assert len(env.llm.calls) == limit
    sent = len(env.events)

    # Any human message restarts the count, summoning or not.
    client.post('/message', json=message(GROUP, 40, 'just talking', sender=GUEST))
    assert len(env.llm.calls) == limit and len(env.events) == sent
    client.post('/message', json=from_bot(41, 'again'))
    assert len(env.llm.calls) == limit + 1


def test_the_window_holds_when_the_chain_keeps_restarting(env, client, monkeypatch):
    monkeypatch.setattr(env.server, 'BOT_REPLY_WINDOW_LIMIT', 2)
    for i in range(4):
        client.post('/message', json=message(GROUP, 50 + 2 * i, 'hm', sender=GUEST))
        client.post('/message', json=from_bot(51 + 2 * i, f'reply {i}'))
    assert len(env.llm.calls) == 2


def test_humans_are_not_limited(env, client):
    for i in range(env.server.BOT_REPLY_CHAIN_LIMIT + 2):
        client.post('/message', json=message(GROUP, 60 + i, f'/* question {i}', sender=OWNER))
    assert len(env.llm.calls) == env.server.BOT_REPLY_CHAIN_LIMIT + 2
