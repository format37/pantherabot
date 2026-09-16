"""The sandbox's exec service, over a real Unix socket as in production."""
import asyncio
import threading
import time

import pytest
import uvicorn

import bot_tools
import exec_service
from conftest import wait_until


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    socket_path = str(tmp_path / 'exec.sock')
    monkeypatch.setattr(exec_service, 'WORK_ROOT', str(work))
    monkeypatch.setattr(exec_service, 'TOKEN', '')
    monkeypatch.setattr(bot_tools, 'SANDBOX_SOCKET', socket_path)
    monkeypatch.setattr(bot_tools, 'SANDBOX_TOKEN', '')
    server = uvicorn.Server(uvicorn.Config(exec_service.app, uds=socket_path,
                                           log_level='warning', lifespan='off'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert wait_until(lambda: server.started, timeout=10)
    yield work
    server.should_exit = True
    thread.join(10)


def test_command_output_and_timeout(sandbox):
    result = asyncio.run(bot_tools.sandbox_exec('7', 'pwd; echo out; echo err >&2', 10))
    assert result['rc'] == 0 and not result['timed_out']
    assert result['stdout'] == f'{sandbox / "7"}\nout\n' and result['stderr'] == 'err\n'

    t0 = time.monotonic()
    result = asyncio.run(bot_tools.sandbox_exec('7', 'sleep 30', 1))
    assert result['timed_out'] and result['rc'] == -9
    assert time.monotonic() - t0 < 5


def test_command_of_a_cancelled_answer_is_killed(sandbox):
    """An edit cancels the attempt; its command must not go on writing to the work dir."""
    async def cancelled():
        call = asyncio.ensure_future(
            bot_tools.sandbox_exec('7', 'sleep 1.5; touch late-marker', 30))
        await asyncio.sleep(0.5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(cancelled())
    time.sleep(2.5)
    assert not (sandbox / '7' / 'late-marker').exists()

    # A command that is left alone still finishes.
    result = asyncio.run(bot_tools.sandbox_exec('7', 'sleep 1.5; touch marker', 30))
    assert result['rc'] == 0 and (sandbox / '7' / 'marker').exists()
