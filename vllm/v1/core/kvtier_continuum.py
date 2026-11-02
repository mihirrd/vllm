# SPDX-License-Identifier: Apache-2.0
"""kvtier baseline: Continuum (arXiv 2511.02230), ported from the authors' vLLM 0.10.2 release.

Enabled with KVTIER_SCHED=continuum. Source: github.com/Hanchenli/vllm-continuum (commit 316a587),
vllm/v1/core/sched/scheduler.py, request_queue.py and estimate_with_func.py. Their release ships a
fixed pin rule ("without the estimation in the paper"), which this port follows:

- Pin: when a request that is not a session's last finishes, its blocks stay held (not freed, so
  nothing can evict them) for 2 s if the tool it calls has averaged at most 2 s, else not at all.
  A tool not yet seen counts as 0 s and is pinned. The tool's time is measured from the request's
  finish to the session's next arrival.
- Unpin: when the pin expires and the session has nothing waiting, when the session's next request
  finishes, or to break a deadlock (nothing running, nothing fits): then the pin that ends last goes.
- Waiting order: sessions holding a pin first, then by the session's first arrival (job-level FCFS).
- Preemption: the running request whose session arrived last, preferring ones that are not a
  session's last step; with at most one running, a pin is released instead.

Their parser reads the tool name from the model's reply and the client flags the last step; the
replay harness sends both in vllm_xargs (program; pause_s is absent on a session's last turn).
Unpinned blocks are evicted by vLLM's LRU, or by KVTIER_POLICY if that is set too.
"""

import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.request import Request

logger = init_logger(__name__)

SCHED = os.environ.get("KVTIER_SCHED", "")
PIN_S = 2.0  # FIXED_THRESHOLD_CONTINUUM in their release
_LOG_EVERY_S = 30.0


def make_continuum() -> "Continuum | None":
    if SCHED == "":
        return None
    if SCHED != "continuum":
        raise ValueError(f"KVTIER_SCHED={SCHED!r}: expected continuum")
    logger.info("kvtier: Continuum scheduling, %.1f s pins", PIN_S)
    return Continuum()


def _xargs(request: Request) -> dict:
    params = request.sampling_params
    return (params.extra_args if params else None) or {}


def session_of(request: Request) -> str:
    return _xargs(request).get("session_id") or request.request_id


def is_last_step(request: Request) -> bool:
    return _xargs(request).get("pause_s") is None


class Continuum:
    def __init__(self):
        self.first_arrival: dict[str, float] = {}  # session -> first request's arrival
        self.pinned: dict[str, tuple[Request, float]] = {}  # session -> (finished request, pin end)
        self._departed: dict[str, tuple[str, float]] = {}  # session -> (tool, finish time)
        self._tool_times: dict[str, list[float]] = {}
        self.pins = self.expired = self.released_early = self.deadlock_unpins = 0
        self._log_t = time.monotonic()

    def on_arrival(self, request: Request, now: float) -> None:
        sid = session_of(request)
        self.first_arrival.setdefault(sid, now)
        if (left := self._departed.pop(sid, None)) is not None:
            tool, t = left
            self._tool_times.setdefault(tool, []).append(now - t)

    def pin_length(self, request: Request) -> float:
        tool = _xargs(request).get("program")
        if not tool:
            return 0.0
        times = self._tool_times.get(tool)
        avg = sum(times) / len(times) if times else 0.0
        return 0.0 if avg > PIN_S else PIN_S

    def on_finish(self, request: Request, now: float) -> tuple[list[Request], bool]:
        """Pins to release now (the session's earlier pin), and whether to pin this request."""
        sid = session_of(request)
        release = [self.pinned.pop(sid)[0]] if sid in self.pinned else []
        if is_last_step(request):
            self.first_arrival.pop(sid, None)
            return release, False
        self._departed[sid] = (_xargs(request).get("program") or "", now)
        length = self.pin_length(request)
        if length <= 0.01:
            return release, False
        self.pinned[sid] = (request, now + length)
        self.pins += 1
        return release, True

    def expired_pins(self, waiting_sessions: set[str], now: float) -> list[Request]:
        out = []
        for sid, (req, end) in list(self.pinned.items()):
            if now >= end and sid not in waiting_sessions:
                del self.pinned[sid]
                out.append(req)
        self.expired += len(out)
        self._maybe_log(now)
        return out

    def release_latest(self) -> Request | None:
        """The pin that ends last, released to make room (their deadlock path)."""
        if not self.pinned:
            return None
        sid = max(self.pinned, key=lambda s: self.pinned[s][1])
        self.released_early += 1
        return self.pinned.pop(sid)[0]

    def waiting_key(self, request: Request) -> tuple[bool, float]:
        sid = session_of(request)
        return sid not in self.pinned, self.first_arrival.get(sid, float("inf"))

    def preempt_key(self, request: Request) -> tuple[bool, float]:
        """Largest goes first: not a last step, then the session that arrived last."""
        sid = session_of(request)
        return not is_last_step(request), self.first_arrival.get(sid, float("inf"))

    def _maybe_log(self, now: float) -> None:
        if now - self._log_t >= _LOG_EVERY_S:
            self._log_t = now
            logger.info(
                "kvtier: continuum %d pins, %d expired, %d released early (%d deadlock); %d held now",
                self.pins, self.expired, self.released_early, self.deadlock_unpins, len(self.pinned),
            )


class ContinuumRequestQueue(FCFSRequestQueue):
    """Waiting queue in Continuum's order: pinned sessions first, then job-level FCFS."""

    def __init__(self, continuum: Continuum) -> None:
        super().__init__()
        self.continuum = continuum

    def peek_request(self) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        return min(self, key=self.continuum.waiting_key)

    def pop_request(self) -> Request:
        request = self.peek_request()
        self.remove(request)
        return request
