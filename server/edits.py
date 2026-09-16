"""Edits that land while Janet is answering.

The rule (Alex, 2026-09-16):

1. An edit of any message updates the history in place. It never triggers an
   answer by itself.
2. A sent answer is final: it is never edited, and a later edit never produces
   a new one.
3. Before sending, Janet checks whether a message in the history she was given
   was edited while she was generating. If so she generates again from the
   current history, and checks again. Edits outside that window don't matter.

`answer()` runs one answer as a series of attempts. An edit that touches the
running attempt's context (the human records it loaded) cancels it at once;
the answer is regenerated once the cancelled attempt has shut down and the
edits have stopped for DEBOUNCE_SECONDS (or MAX_SETTLE_SECONDS have passed).
An attempt that finishes is checked once more before anything is sent, so an
edit that slipped in after the last cancel point sends it back as well. After
MAX_REGENERATIONS the next attempt is sent whatever happens.

Cancelling an attempt ends its Claude CLI: the SDK closes the CLI's stdin,
gives it 5 s, then SIGTERM (measured 2026-09-16 with CLI 2.1.259, which does not
exit on EOF mid-turn), so a restart comes about 5 s after the edit. The next
attempt waits for that rather than run a second 230 MB CLI next to it. A Gemini
call already in a worker thread cannot be stopped; its result is dropped with
the attempt's outbox.

At most MAX_CONCURRENT_ANSWERS answers are generated at once. The relay's two
worker threads used to be the only limit; it now runs more of them so that
edits get through while answers run, and the limit lives here instead.

The registry lives in memory: a restart loses the running generations too.
"""
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 1.5
# The wait for edits to stop ends after this long even if they don't: a message
# edited over and over must not hold an answer (and its slot) forever.
MAX_SETTLE_SECONDS = 10
MAX_REGENERATIONS = 3
# Each answer runs a Claude CLI of about 230 MB, and the production host has no
# swap. Two is what the relay allowed before it got more threads.
MAX_CONCURRENT_ANSWERS = 2

# chat_id (str) -> the generations running in that chat
_running = {}

_CANCELLED = object()

# (event loop, semaphore): a semaphore belongs to one loop, and tests run several.
_slots = None


def _answer_slots():
    global _slots
    loop = asyncio.get_running_loop()
    if _slots is None or _slots[0] is not loop:
        _slots = (loop, asyncio.Semaphore(MAX_CONCURRENT_ANSWERS))
    return _slots[1]


class Generation:
    """One answer in one chat, across all of its attempts."""

    def __init__(self, chat_id, message_id):
        self.chat_id = str(chat_id)
        self.message_id = message_id
        self.attempt = 0
        self.regenerations = 0
        # Message ids of the human records the current attempt loaded. The
        # attempt sets them before its first await.
        self.context = set()
        # Message ids edited since the current attempt started.
        self.edited = set()
        # Monotonic time of the last edit that touched the context.
        self.last_edit = 0.0
        # Past the cap: this attempt is sent even if its context is edited.
        self.final = False
        # Past the pre-send check: from now on an edit changes history only.
        self.committed = False
        self.task = None
        self.cancel_requested = False
        self.cancelled_at = 0.0

    def __repr__(self):
        return (f'the answer to message {self.message_id} in chat {self.chat_id} '
                f'(attempt {self.attempt})')

    def on_edit(self, message_id):
        if self.committed:
            return
        self.edited.add(message_id)
        if message_id not in self.context:
            return
        self.last_edit = time.monotonic()
        if self.final or self.cancel_requested or self.task is None or self.task.done():
            return
        logger.info(f'edit of message {message_id}: cancelling {self}')
        self.cancel_requested = True
        self.cancelled_at = time.monotonic()
        self.task.cancel()

    async def run_attempt(self, attempt):
        """The attempt's result, or _CANCELLED if an edit cancelled it."""
        self.attempt += 1
        self.final = self.regenerations >= MAX_REGENERATIONS
        if self.final:
            logger.warning(f'{self}: {self.regenerations} regenerations, the cap; '
                           f'this attempt is sent even if the history changes again')
        self.context = set()
        self.edited = set()
        self.cancel_requested = False
        self.task = asyncio.get_running_loop().create_task(attempt(self))
        try:
            # Awaiting the task itself also waits for a cancelled attempt to
            # shut its CLI down.
            return await self.task
        except asyncio.CancelledError:
            if (self.cancel_requested and self.task.cancelled()
                    and asyncio.current_task().cancelling() == 0):
                took = time.monotonic() - self.cancelled_at
                logger.info(f'{self}: cancelled; it shut down in {took:.1f}s')
                return _CANCELLED
            raise
        finally:
            if not self.task.done():
                # We are being cancelled ourselves (shutdown): take the attempt along.
                self.task.cancel()

    async def settle(self):
        """Wait until no edit has touched the context for DEBOUNCE_SECONDS.

        Never longer than MAX_SETTLE_SECONDS; the regeneration cap then bounds
        the whole answer.
        """
        deadline = time.monotonic() + MAX_SETTLE_SECONDS
        while True:
            remaining = min(self.last_edit + DEBOUNCE_SECONDS, deadline) - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)


async def _current_result(generation, attempt):
    """Attempt until an attempt passes the pre-send check (or the cap is reached)."""
    while True:
        result = await generation.run_attempt(attempt)
        if result is not _CANCELLED:
            changed = generation.edited & generation.context
            if not changed:
                return result
            if generation.final:
                logger.warning(f'{generation}: sending it although message(s) '
                               f'{sorted(changed)} were edited meanwhile')
                return result
            logger.info(f'{generation}: message(s) {sorted(changed)} were edited '
                        f'during generation; discarding it')
        generation.regenerations += 1
        await generation.settle()
        logger.info(f'{generation}: regenerating '
                    f'({generation.regenerations} of {MAX_REGENERATIONS})')


async def answer(chat_id, message_id, attempt, commit):
    """Produce one answer that survives the pre-send check, then commit it.

    `attempt(generation)` reads the history, sets `generation.context` before
    its first await, and returns the attempt's result. `commit(result)` saves
    and sends it; once it starts, edits only change the history.
    """
    generation = Generation(chat_id, message_id)
    slots = _answer_slots()
    if slots.locked():
        logger.info(f'{generation}: waiting, {MAX_CONCURRENT_ANSWERS} answers are running')
    try:
        async with slots:
            _running.setdefault(generation.chat_id, set()).add(generation)
            result = await _current_result(generation, attempt)
            # No await between the pre-send check and this point.
            generation.committed = True
        return await commit(result)
    finally:
        chats = _running.get(generation.chat_id)
        if chats is not None:
            chats.discard(generation)
            if not chats:
                del _running[generation.chat_id]


def record_edit(chat_id, message_id):
    """Tell the generations running in a chat that a history record changed."""
    for generation in list(_running.get(str(chat_id), ())):
        generation.on_edit(message_id)


def running(chat_id):
    """The generations running in a chat (for logs and tests)."""
    return set(_running.get(str(chat_id), ()))
