"""Deep research: started by a tool, delivered by the bot itself when it is done."""
import asyncio
import time

import pytest

from conftest import ALICE, message, records, wait_until

import research

RESPONSE = {
    'choices': [{'message': {'content': '<think>hidden</think>Report body [1][2].'}}],
    'citations': ['https://a.example/1', 'https://b.example/2'],
    'search_results': [{'title': 'A', 'url': 'https://a.example/1'},
                       {'title': 'C', 'url': 'https://c.example/3'}],
}


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv('PERPLEXITY_API_KEY', 'test-key')
    research._jobs.clear()


def test_parse_strips_thinking_and_merges_sources():
    text, sources = research.parse(RESPONSE)
    assert text == 'Report body [1][2].'
    assert sources == [('A', 'https://a.example/1'), ('', 'https://b.example/2'),
                       ('C', 'https://c.example/3')]
    assert research.with_sources(text, sources).endswith('[3] C — https://c.example/3')


def start_research(env, monkeypatch, post):
    """The LLM's first call runs the deep_research tool, as the model would."""
    monkeypatch.setattr(research, '_post', post)
    results = []

    async def script(call):
        if call.index == 0:
            tools = env.bot_tools.build_tools(call.chat_id, call.message_id, [])
            results.append(await tools['deep_research'].handler({'request': 'the market'}))
            results.append(await tools['deep_research'].handler({'request': 'the market'}))
            return 'started'
        return f'presenting: {call.prompt[-60:]}'

    env.llm.script = script
    return results


def test_the_report_comes_back_on_its_own(env, client, monkeypatch):
    def post(model, request, reasoning_effort=None, timeout=None):
        assert model == research.RESEARCH_MODEL and reasoning_effort == 'medium'
        time.sleep(0.2)
        return RESPONSE

    results = start_research(env, monkeypatch, post)
    client.post('/message', json=message(ALICE, 10, 'research the market'))
    assert 'started' in results[0]['content'][0]['text']
    assert 'already running' in results[1]['content'][0]['text']
    assert [e['kind'] for e in env.events] == ['rich']

    assert wait_until(lambda: len(env.events) == 3, timeout=10), env.events
    document, answer = env.events[1:]
    assert document['kind'] == 'document' and document['reply_to'] == 10
    assert document['name'] == 'research-10.md'
    report = document['data'].decode()
    assert 'Report body' in report and '3. C — https://c.example/3' in report

    call = env.llm.wait_for_calls(2)
    assert 'deep_research (tool)' in call.prompt and 'Report body [1][2].' in call.prompt
    assert '[2] https://b.example/2' in call.prompt
    assert answer['kind'] == 'rich' and answer['reply_to'] == 10
    assert answer['text'].startswith('presenting:')
    assert not research.running()

    # The report is history, under a suffix no message has: an edit of the
    # message that asked for it leaves the report alone.
    kinds = [(r['type'], r['_file'].rsplit('_', 1)[1]) for r in records(env, ALICE)]
    assert kinds == [('HumanMessage', '10.json'), ('AIMessage', '10.json'),
                     ('HumanMessage', 'research-10.json'), ('AIMessage', '10.json')]
    client.raw.post('/edited_message', json=message(ALICE, 10, 'research the market, edited',
                                                    edit_date=1758000100))
    report_record = [r for r in records(env, ALICE) if r['_file'].endswith('research-10.json')][0]
    assert 'Report body' in report_record['text'] and 'edit_date' not in report_record


def test_a_failure_is_reported(env, client, monkeypatch):
    def post(*args, **kwargs):
        raise RuntimeError('401 Unauthorized')

    start_research(env, monkeypatch, post)
    client.post('/message', json=message(ALICE, 20, 'research the market'))
    assert wait_until(lambda: len(env.events) == 2, timeout=10), env.events
    call = env.llm.wait_for_calls(2)
    assert 'failed: RuntimeError: 401 Unauthorized' in call.prompt
    assert env.events[1]['kind'] == 'rich' and env.events[1]['reply_to'] == 20


def test_limits():
    with pytest.raises(ValueError, match='reasoning_effort'):
        research.start(1, 1, 'x', 'max')
    with pytest.raises(ValueError, match='empty'):
        research.start(1, 1, '  ')

    async def run():
        for i in range(research.MAX_RESEARCH_JOBS):
            research._jobs[('1', i)] = research.Job(1, i, 'x', 'low')
        with pytest.raises(ValueError, match='already running'):
            research.start(1, 99, 'x')
        research._jobs.clear()

    asyncio.run(run())


def test_without_a_key(env, monkeypatch):
    monkeypatch.delenv('PERPLEXITY_API_KEY')
    tools = env.bot_tools.build_tools(1, 1, [])

    async def run():
        assert (await tools['web_search'].handler({'query': 'x'}))['is_error']
        assert (await tools['deep_research'].handler({'request': 'x'}))['is_error']

    asyncio.run(run())
