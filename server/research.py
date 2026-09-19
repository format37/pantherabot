"""Web search and deep research through the Perplexity API, without an MCP server.

`search()` is one synchronous call (sonar-pro, seconds). `start()` runs a deep
research job (sonar-deep-research, minutes) as a task of the head's event loop,
independent of the answer that started it: the answer says "started" and ends,
and when the job is done the task calls `on_done` (set by server.py), which
posts the report to the chat and has Janet present it. Nobody has to ask.

Both return the model's text and the sources as a compact text, never the raw
API response: the raw JSON of a deep research report (reasoning, search result
snippets) was larger than what the CLI hands the model as a tool result.
"""
import asyncio
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)

API_URL = 'https://api.perplexity.ai/chat/completions'
SEARCH_MODEL = 'sonar-pro'
RESEARCH_MODEL = 'sonar-deep-research'
SEARCH_TIMEOUT = 120
RESEARCH_TIMEOUT = 1500          # the API itself takes up to ~20 min at high effort
MAX_RESEARCH_JOBS = 2            # per process; a job costs real money
EFFORTS = ('low', 'medium', 'high')

_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def api_key():
    return os.environ.get('PERPLEXITY_API_KEY', '')


def enabled():
    """True when the bot can reach Perplexity itself (PERPLEXITY_API_KEY is set)."""
    return bool(api_key())


def _post(model, request, reasoning_effort=None, timeout=SEARCH_TIMEOUT):
    """One chat completion; the API's JSON."""
    payload = {'model': model, 'messages': [{'role': 'user', 'content': request}]}
    if reasoning_effort is not None:
        payload['reasoning_effort'] = reasoning_effort
    headers = {'Authorization': f'Bearer {api_key()}', 'Content-Type': 'application/json'}
    response = requests.post(API_URL, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse(response):
    """(text, sources) of an API response.

    `sources` is a list of (title, url); the text cites them as [1], [2], ...
    in that order.
    """
    choices = response.get('choices') or []
    content = (choices[0].get('message') or {}).get('content', '') if choices else ''
    text = _THINK_RE.sub('', content or '').strip()

    titles = {}
    for item in response.get('search_results') or []:
        if isinstance(item, dict) and item.get('url'):
            titles.setdefault(item['url'], item.get('title') or '')
    urls = [u for u in (response.get('citations') or []) if isinstance(u, str)]
    for url in titles:
        if url not in urls:
            urls.append(url)
    sources = [(titles.get(url, ''), url) for url in urls]
    return text, sources


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
    text, sources = parse(_post(SEARCH_MODEL, query))
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
    return ('Deep research started. It takes about 3-10 minutes (up to 20 at high effort) '
            'and runs on its own: the full report will be posted in this chat as a file, '
            'and you will be asked to present it, when it is ready. Do not poll for it; '
            'tell the user it has started and finish your reply.')


async def _run(job, key):
    try:
        try:
            response = await asyncio.to_thread(
                _post, RESEARCH_MODEL, job.request, job.reasoning_effort, RESEARCH_TIMEOUT)
            job.text, job.sources = parse(response)
            if not job.text:
                job.error = 'the API returned an empty report'
            logger.info(f'{job}: done in {job.elapsed}s, {len(job.text)} chars, '
                        f'{len(job.sources)} sources')
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
