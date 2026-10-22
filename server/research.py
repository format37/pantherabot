"""Web search and deep research through the Perplexity Agent API, without an MCP server.

Perplexity ended Sonar chat completions on 2026-09-27; both tools now call
the Agent API (`/v1/agent`) with a preset. `search()` is Pro Search (preset
`low`, formerly `pro-search`: a few search steps, seconds), one synchronous
call. `start()` runs a deep research job (preset `medium` or `high`,
minutes) as a background run of the API, polled by a task of the head's event
loop, independent of the answer that started it: the answer says "started"
and ends, and when the job is done the task calls `on_done` (set by
server.py), which posts the report to the chat and has Janet present it.
Nobody has to ask.

Both return the model's text and the sources as a compact text, never the raw
API response: the raw JSON of a deep research report (search result
snippets) was larger than what the CLI hands the model as a tool result.
"""
import asyncio
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)

API_URL = 'https://api.perplexity.ai/v1/agent'
SEARCH_PRESET = 'low'            # Pro Search
SEARCH_TIMEOUT = 120
RESEARCH_TIMEOUT = 1800          # a background run; high can take ~20 min
POLL_SECONDS = 10
MAX_RESEARCH_JOBS = 2            # per process; a job costs real money
# The deep_research tool's effort -> (preset, reasoning effort override).
EFFORTS = {
    'low': ('medium', None),
    'medium': ('high', None),
    'high': ('high', 'high'),
}
TERMINAL = ('completed', 'failed', 'cancelled', 'incomplete')

# Presets cite as [web:3] (`fast` as [3]), the 3 being a search result's id.
_CITE_RE = re.compile(r'\[(web:)?(\d+)\]')


def api_key():
    return os.environ.get('PERPLEXITY_API_KEY', '')


def enabled():
    """True when the bot can reach Perplexity (PERPLEXITY_API_KEY is set)."""
    return bool(api_key())


def _headers():
    return {'Authorization': f'Bearer {api_key()}', 'Content-Type': 'application/json'}


def _create(payload, timeout=SEARCH_TIMEOUT):
    """POST one run; the API's response object."""
    response = requests.post(API_URL, json=payload, headers=_headers(), timeout=timeout)
    response.raise_for_status()
    return response.json()


def _retrieve(response_id):
    """GET a (background) run by id."""
    response = requests.get(f'{API_URL}/{response_id}', headers=_headers(), timeout=60)
    response.raise_for_status()
    return response.json()


def parse(response):
    """(text, sources) of an API response.

    `sources` is a list of (title, url); the text cites them as [1], [2], ...
    in that order. The API's own markers ([web:7]) point at search result ids
    across the whole run; they are renumbered to the sources the text cites,
    in the order it first cites them. A text that cites nothing gets every
    result, so the links are not lost.
    """
    output = response.get('output') or []
    text = ''.join(
        part.get('text') or ''
        for item in output if item.get('type') == 'message'
        for part in item.get('content') or [] if part.get('type') == 'output_text'
    )

    results = {}
    for item in output:
        if item.get('type') != 'search_results':
            continue
        for result in item.get('results') or []:
            if isinstance(result, dict) and result.get('url') and result.get('id') is not None:
                results.setdefault(int(result['id']), (result.get('title') or '', result['url']))

    order = {}

    def renumber(match):
        prefixed, number = match.group(1), int(match.group(2))
        if number not in results:
            return '' if prefixed else match.group(0)
        return f'[{order.setdefault(number, len(order) + 1)}]'

    text = _CITE_RE.sub(renumber, text).strip()
    if not order:
        order = {number: i for i, number in enumerate(sorted(results), 1)}
    sources = [results[number] for number in sorted(order, key=order.get)]
    return text, sources


def error_of(response):
    """Why a run that is not `completed` has no usable answer."""
    status = response.get('status')
    detail = response.get('error') or response.get('incomplete_details') or ''
    if isinstance(detail, dict):
        detail = detail.get('message') or detail.get('reason') or detail
    return f'the run ended as {status}' + (f': {detail}' if detail else '')


def format_sources(sources):
    if not sources:
        return ''
    lines = []
    for i, (title, url) in enumerate(sources, 1):
        lines.append(f'[{i}] {title} — {url}' if title else f'[{i}] {url}')
    return 'Sources:\n' + '\n'.join(lines)


def with_sources(text, sources):
    block = format_sources(sources)
    return f'{text}\n\n{block}' if block else text


def search(query):
    """A grounded answer with its sources, as text. Synchronous: run it in a thread."""
    response = _create({'preset': SEARCH_PRESET, 'input': query})
    text, sources = parse(response)
    if response.get('status') not in (None, 'completed') and not text:
        raise RuntimeError(error_of(response))
    return with_sources(text, sources)


class Job:
    """One deep research job: what was asked, by which message, and its result."""

    def __init__(self, chat_id, message_id, request, reasoning_effort):
        self.chat_id = str(chat_id)
        self.message_id = int(message_id)
        self.request = request
        self.reasoning_effort = reasoning_effort
        self.started = time.monotonic()
        self.text = ''
        self.sources = []
        self.error = None
        self.response_id = None
        self.task = None

    def __repr__(self):
        return f'deep research for message {self.message_id} in chat {self.chat_id}'

    @property
    def elapsed(self):
        return int(time.monotonic() - self.started)


# (chat_id, message_id) -> Job, while it runs. The same message never starts
# two jobs: an answer regenerated after an edit calls the tool again.
_jobs = {}

# async (job) -> None: posts the result to the chat. Set by server.py.
on_done = None


def running():
    return list(_jobs.values())


def start(chat_id, message_id, request, reasoning_effort='medium'):
    """Start a job on the running event loop; the text the tool returns.

    Raises ValueError when it cannot start (limits, arguments, no key).
    """
    if not enabled():
        raise ValueError('deep research is not configured (no PERPLEXITY_API_KEY)')
    if reasoning_effort not in EFFORTS:
        raise ValueError(f'reasoning_effort must be one of {", ".join(EFFORTS)}')
    request = (request or '').strip()
    if not request:
        raise ValueError('the request is empty')

    key = (str(chat_id), int(message_id))
    job = _jobs.get(key)
    if job is not None:
        return (f'Deep research for this message is already running ({job.elapsed}s so far). '
                f'Its report will be posted in this chat when it is ready; do not start it again.')
    if len(_jobs) >= MAX_RESEARCH_JOBS:
        raise ValueError(f'{MAX_RESEARCH_JOBS} deep research jobs are already running; '
                         f'try again when one has finished')

    job = Job(chat_id, message_id, request, reasoning_effort)
    _jobs[key] = job
    job.task = asyncio.get_running_loop().create_task(_run(job, key))
    logger.info(f'{job}: started (effort {reasoning_effort}, {len(request)} chars)')
    return ('Deep research started. It takes several minutes (up to 20 at high effort) '
            'and runs on its own: the full report will be posted in this chat as a file, '
            'and you will be asked to present it, when it is ready. Do not poll for it; '
            'tell the user it has started and finish your reply.')


async def _run(job, key):
    preset, effort = EFFORTS[job.reasoning_effort]
    payload = {'preset': preset, 'input': job.request, 'background': True}
    if effort:
        payload['reasoning'] = {'effort': effort}
    try:
        try:
            response = await asyncio.to_thread(_create, payload)
            job.response_id = response.get('id')
            logger.info(f'{job}: run {job.response_id} ({response.get("preset")}) '
                        f'{response.get("status")}')
            while response.get('status') not in TERMINAL:
                if job.elapsed > RESEARCH_TIMEOUT:
                    raise TimeoutError(f'no result after {job.elapsed}s')
                await asyncio.sleep(POLL_SECONDS)
                response = await asyncio.to_thread(_retrieve, job.response_id)
            job.text, job.sources = parse(response)
            if response.get('status') != 'completed' and not job.text:
                job.error = error_of(response)
            elif not job.text:
                job.error = 'the API returned an empty report'
            cost = ((response.get('usage') or {}).get('cost') or {}).get('total_cost')
            logger.info(f'{job}: {response.get("status")} in {job.elapsed}s, '
                        f'{len(job.text)} chars, {len(job.sources)} sources, cost {cost}')
        except Exception as e:
            job.error = f'{type(e).__name__}: {e}'
            logger.error(f'{job}: failed after {job.elapsed}s: {job.error}')
    finally:
        _jobs.pop(key, None)
    if on_done is None:
        logger.error(f'{job}: no on_done handler, the report is lost')
        return
    try:
        await on_done(job)
    except Exception:
        logger.exception(f'{job}: delivering the result failed')
