"""Model and effort per chat: set by tools, read by every later answer and by /reset."""
import asyncio
import sys
from pathlib import Path

from conftest import ALICE, OWNER, message


def set_effort(env, chat_id, effort):
    tools = env.bot_tools.build_tools(chat_id, 1, [])
    return asyncio.run(tools['set_effort'].handler({'effort': effort}))


def set_model(env, chat_id, model):
    tools = env.bot_tools.build_tools(chat_id, 1, [])
    return asyncio.run(tools['set_model'].handler({'model': model}))


def test_the_tool_sets_this_chat_only(env):
    assert env.tools_cli.get_effort(ALICE) == 'high'
    result = set_effort(env, ALICE, 'max')
    text = result['content'][0]['text']
    assert 'Effort in this chat: max (was high)' in text
    assert f'Model: {env.tools_cli.primary_model()}.' in text
    assert env.tools_cli.get_effort(ALICE) == 'max'
    assert env.tools_cli.get_effort(OWNER) == 'high'
    prompt = env.server.panthera.get_system_prompt(str(ALICE))
    assert 'at effort max.' in prompt


def test_an_unknown_level_is_refused(env):
    assert set_effort(env, ALICE, 'ultra')['is_error']
    assert env.tools_cli.get_effort(ALICE) == 'high'


def test_the_model_is_switched_in_this_chat_only(env):
    default = env.tools_cli.primary_model()
    assert env.tools_cli.get_model(ALICE) == default
    set_effort(env, ALICE, 'low')
    text = set_model(env, ALICE, 'claude-haiku-5-5')['content'][0]['text']
    assert text == (f'Model in this chat: claude-haiku-5-5 (was {default}), from the next '
                    f'message on; this reply still runs on {default}. Effort: low.')
    assert env.tools_cli.get_model(ALICE) == 'claude-haiku-5-5'
    assert env.tools_cli.get_model(OWNER) == default
    prompt = env.server.panthera.get_system_prompt(str(ALICE))
    assert 'you run on claude-haiku-5-5 at effort low.' in prompt


def test_an_unknown_model_is_refused(env):
    assert set_model(env, ALICE, 'claude-sonnet-4')['is_error']
    assert env.tools_cli.get_model(ALICE) == env.tools_cli.primary_model()


def test_reset_names_the_bot_model_and_effort(env, client, monkeypatch):
    monkeypatch.setenv('BOT_USERNAME', 'janet_bot')
    set_effort(env, ALICE, 'low')
    set_model(env, ALICE, 'claude-opus-5-5')
    body = client.raw.post('/message', json=message(ALICE, 5, '/reset')).json()['body']
    assert body == ('Chat messages memory has been cleaned.\n'
                    'Bot: Janet (@janet_bot)\n'
                    'Model: claude-opus-5-5\n'
                    'Effort: low')
    # Both outlive /reset, as the memory and the custom prompt do.
    assert env.tools_cli.get_effort(ALICE) == 'low'
    assert env.tools_cli.get_model(ALICE) == 'claude-opus-5-5'


def test_the_cli_gets_the_chats_model_and_effort(env, client, monkeypatch, tmp_path):
    """Through the real SDK: the next answer runs with this chat's --model and --effort."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    state = tmp_path / 'fake-cli'
    state.mkdir()
    argv = tmp_path / 'argv'
    wrapper = tmp_path / 'claude'
    wrapper.write_text(f'#!/bin/sh\necho "$@" >> {argv}\n'
                       f'exec {sys.executable} {Path(__file__).with_name("fake_claude.py")} "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv('FAKE_CLAUDE_DIR', str(state))
    monkeypatch.setenv('FAKE_CLAUDE_IMAGE_BASE64', '4')     # answer at once, no hanging run
    monkeypatch.setenv('CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK', '1')
    monkeypatch.setattr(subprocess_cli.SubprocessCLITransport, '_find_cli', lambda self: str(wrapper))
    monkeypatch.delattr(env.server.panthera, '_claude_agent_query')   # the real one

    set_effort(env, ALICE, 'xhigh')
    set_model(env, ALICE, 'claude-sonnet-5-5')
    client.post('/message', json=message(ALICE, 10, 'think hard'))

    runs = [line for line in argv.read_text().splitlines() if '--effort' in line]
    assert runs and '--effort xhigh' in runs[-1] and '--model claude-sonnet-5-5' in runs[-1]


def test_help_lists_the_models_and_efforts(env, client):
    set_model(env, ALICE, 'claude-opus-5-5')
    client.raw.post('/message', json=message(ALICE, 6, '/help'))
    text = [e['text'] for e in env.events if e['kind'] == 'message'][-1]
    # MarkdownV2: every - . ( ) outside markup is escaped.
    assert '*Model & Effort*\n• This chat: claude\\-opus\\-5\\-5, effort high\n' in text
    for model in env.tools_cli.MODELS:
        assert model.replace('-', '\\-') in text
    assert 'Haiku 5\\.5: fastest for quick answers \\(claude\\-haiku\\-5\\-5\\)' in text
    assert 'low, medium, high, xhigh, max \\(default high\\)' in text
