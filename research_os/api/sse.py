"""Server-Sent Events for a run: replays from a sequence number, then follows live until the run is done."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from research_os.api.service import RunRecord


async def event_stream(record: RunRecord, from_seq: int = 0, poll_s: float = 0.05) -> AsyncIterator[dict]:
    """Yield sse-starlette events (id = seq). Ends after the run finished and every event was sent."""
    seq = from_seq
    while True:
        batch = record.ctx.events.since(seq)
        for ev in batch:
            yield {"id": str(ev.seq), "event": "run_event", "data": json.dumps(ev.model_dump(mode="json"))}
            seq = ev.seq + 1
        if record.done and not record.ctx.events.since(seq):
            return
        await asyncio.sleep(poll_s)
