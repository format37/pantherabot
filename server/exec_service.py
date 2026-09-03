"""Command execution service — the bot's "hands".

Runs in its own container (`panthera_sandbox`) with `network_mode: none`, no
secrets, no `data/`, no Claude config: nothing worth stealing and nowhere to
send it. The head talks to it over a Unix socket on a shared bind mount, which
is why the container needs no network stack at all.

Everything the model asks to run lands here, in `/work/{chat_id}`.
"""
import asyncio
import os
import signal
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

WORK_ROOT = os.environ.get('SANDBOX_WORK_ROOT', '/work')
TOKEN = os.environ.get('SANDBOX_TOKEN', '')
MAX_TIMEOUT = 120
DEFAULT_TIMEOUT = 60
# Roughly 20 KB back to the model; anything longer is the model's problem to
# narrow down, not something to blow the context window on.
MAX_OUTPUT = 20000


class ExecRequest(BaseModel):
    chat_id: str
    command: str
    timeout: int = DEFAULT_TIMEOUT


def _work_dir(chat_id: str) -> str:
    """Per-chat scratch directory, created on demand.

    chat_id is validated by the head before it gets here; keep it to one path
    segment anyway so a bad caller cannot escape WORK_ROOT.
    """
    safe = os.path.basename(str(chat_id).strip()) or 'default'
    path = os.path.join(WORK_ROOT, safe)
    os.makedirs(path, exist_ok=True)
    return path


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT:
        return text, False
    half = MAX_OUTPUT // 2
    return f'{text[:half]}\n...[{len(text) - MAX_OUTPUT} characters cut]...\n{text[-half:]}', True


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/exec')
async def execute(req: ExecRequest, x_sandbox_token: str = Header(default='')):
    if TOKEN and x_sandbox_token != TOKEN:
        raise HTTPException(status_code=403, detail='bad sandbox token')

    timeout = max(1, min(int(req.timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    cwd = _work_dir(req.chat_id)
    started = time.monotonic()

    proc = await asyncio.create_subprocess_exec(
        'bash', '-lc', req.command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Own process group, so a timeout takes the whole tree down and not
        # just the shell that spawned it.
        start_new_session=True,
    )

    timed_out = False
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        stdout, stderr = await proc.communicate()

    out, out_cut = _truncate((stdout or b'').decode('utf-8', 'replace'))
    err, err_cut = _truncate((stderr or b'').decode('utf-8', 'replace'))

    return JSONResponse(content={
        'rc': proc.returncode,
        'stdout': out,
        'stderr': err,
        'truncated': out_cut or err_cut,
        'timed_out': timed_out,
        'duration_s': round(time.monotonic() - started, 2),
        'cwd': cwd,
    })
