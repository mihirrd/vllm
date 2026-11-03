# SPDX-License-Identifier: Apache-2.0
"""brb (C1): session table and policy-ordered eviction of paused sessions' cached blocks.

Enabled with BRB_POLICY=remaining (our remaining-time eviction) or BRB_POLICY=oracle
(Belady: the true pause, sent by the replay harness). Unset, vLLM's LRU order is untouched.

The scheduler reports each request's arrival and finish; the session ID, the program of the tool
that follows, and (oracle only) that tool's true duration come in `vllm_xargs`. When the block pool
must evict a cached block, it asks `take` instead of popping the LRU end of its free queue. Blocks
no session's latest request holds go first; then victims come from paused sessions in the policy's order, tail blocks first like vLLM, so a partly evicted
session keeps its prefix. With no paused session left to evict, the pool falls back to LRU.
"""

import os
import time

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

logger = init_logger(__name__)

POLICY = os.environ.get("BRB_POLICY", "")
# Where the tool pause history is kept across restarts (remaining only). Unset: every server starts
# with no history, as in the experiments.
PAUSE_DB = os.environ.get("BRB_PAUSE_DB", "")
_REORDER_EVERY_S = 0.25  # time left changes slowly; re-rank at most this often unless sessions change
_LOG_EVERY_S = 30.0
_SAVE_EVERY_S = 60.0  # a crash loses at most this much pause history


def make_session_eviction() -> "SessionEviction | None":
    if POLICY in ("", "lru"):
        return None
    if POLICY not in ("remaining", "oracle"):
        raise ValueError(f"BRB_POLICY={POLICY!r}: expected remaining, oracle or lru")
    logger.info("brb: session eviction policy %s", POLICY)
    return SessionEviction(oracle=POLICY == "oracle", pause_db=PAUSE_DB or None)


class SessionEviction:
    def __init__(self, oracle: bool, pause_db: str | None = None):
        # The policy is shared with the simulator (brb package, installed in the same env).
        from brb.policy import Session, State
        from brb.sim.policies import RemainingTimeEviction

        self._Session, self._State = Session, State
        self.policy = RemainingTimeEviction(oracle=oracle)
        self.sessions: dict[str, Session] = {}
        # Block chain of each paused session's last finished request, prefix first; dropped once
        # nothing in it is left to evict.
        self.chains: dict[str, list[KVCacheBlock]] = {}
        self.owner: dict[int, str] = {}  # block_id -> session whose chain holds it
        # Cached blocks that left a session's chain without being reused, oldest first: mostly the
        # KV of a reply the next prompt does not contain (the replay sends the recorded reply, a
        # real agent may edit its history). Nothing will hit them, so they go before any session.
        self.orphans: dict[int, KVCacheBlock] = {}
        self._order: list[Session] = []
        self._order_t = -1e9
        self._dirty = True
        self._victims: set[str] = set()  # paused sessions that lost blocks to the policy (for logging)
        self.evicted_orphans = 0
        self.evicted_by_policy = 0
        self.evicted_by_lru = 0
        self._log_t = time.monotonic()
        # The oracle reads true pauses and learns nothing, so it has no history to keep.
        self.pause_db = None if oracle else pause_db
        self._unsaved = False
        self._saved_t = -1e9
        if self.pause_db:
            n = self.policy.predictor.load(self.pause_db)
            logger.info("brb: loaded pause history for %d programs from %s", n, self.pause_db)

    def on_arrival(self, sid: str, now: float) -> None:
        s = self.sessions.get(sid)
        if s is None:
            self.sessions[sid] = self._Session(sid, program_arrival=now)
            return
        if s.state is self._State.PAUSED:
            pause = now - s.pause_start
            if s.session_id in self._victims:
                chain = self.chains.get(sid, ())
                logger.info("brb: return sid=%s program=%s pause=%.2f evicted, %d blocks left",
                            sid, s.program, pause, sum(self.owner.get(b.block_id) == sid for b in chain))
            self._victims.discard(sid)
            self.policy.on_return(s, pause)
            self._dirty = self._unsaved = True
            self._maybe_save(now)
        s.state = self._State.ACTIVE

    def _maybe_save(self, now: float) -> None:
        if not (self.pause_db and self._unsaved) or now - self._saved_t < _SAVE_EVERY_S:
            return
        try:
            self.policy.predictor.save(self.pause_db)
            self._unsaved, self._saved_t = False, now
        except OSError as e:  # a full or read-only disk must not take the server down
            logger.warning("brb: could not save pause history to %s: %s", self.pause_db, e)
            self._saved_t = now

    def on_finish(self, sid: str, program: str, true_pause_s: float | None,
                  blocks: list[KVCacheBlock], now: float) -> None:
        s = self.sessions.setdefault(sid, self._Session(sid, program_arrival=now))
        s.state = self._State.PAUSED
        s.pause_start = s.last_used = now
        s.program = program
        s.true_pause_s = true_pause_s
        old = [b for b in self.chains.pop(sid, ()) if self.owner.get(b.block_id) == sid]
        for b in old:
            del self.owner[b.block_id]
        chain = [b for b in blocks if not b.is_null]
        self.chains[sid] = chain
        for b in chain:
            self.owner[b.block_id] = sid
            self.orphans.pop(b.block_id, None)
        for b in old:
            if b.block_id not in self.owner:
                self.orphans[b.block_id] = b
        self._dirty = True

    def on_allocated(self, blocks: list[KVCacheBlock]) -> None:
        for b in blocks:
            self.owner.pop(b.block_id, None)
            self.orphans.pop(b.block_id, None)

    def take(self, queue: FreeKVCacheBlockQueue, n: int, now: float) -> list[KVCacheBlock]:
        """Remove up to n cached free blocks from `queue`: orphans first, then paused sessions' blocks
        in the policy's order."""
        out: list[KVCacheBlock] = []
        while self.orphans and len(out) < n:
            b = self.orphans.pop(next(iter(self.orphans)))
            # An orphan in use again (hit by another request) is skipped; its next finish re-owns it.
            if b.ref_cnt == 0 and b.block_hash is not None and b.block_id not in self.owner:
                queue.remove(b)
                out.append(b)
        self.evicted_orphans += len(out)
        if len(out) == n:
            return out
        if self._dirty or now - self._order_t > _REORDER_EVERY_S:
            paused = [self.sessions[sid] for sid in self.chains
                      if self.sessions[sid].state is self._State.PAUSED]
            self._order = self.policy.eviction_order(paused, now)
            self._order_t, self._dirty = now, False
        n_orphans = len(out)
        for s in self._order:
            sid = s.session_id
            chain = self.chains.get(sid)
            if chain is None or s.state is not self._State.PAUSED:
                continue
            if sid not in self._victims:
                self._victims.add(sid)
                logger.info("brb: victim sid=%s program=%s elapsed=%.2f time_left=%.2f paused=%d",
                            sid, s.program, now - s.pause_start, self.policy.time_left(s, now), len(self._order))
            while chain and len(out) < n:
                b = chain.pop()
                if self.owner.get(b.block_id) != sid:
                    continue  # reused by another request since
                del self.owner[b.block_id]
                # Blocks in use elsewhere (a shared prefix) are not free; they leave the chain.
                if b.ref_cnt == 0 and b.block_hash is not None:
                    queue.remove(b)
                    out.append(b)
            if not chain:
                del self.chains[sid]
            if len(out) == n:
                break
        self.evicted_by_policy += len(out) - n_orphans
        self.evicted_by_lru += n - len(out)
        if now - self._log_t > _LOG_EVERY_S:
            self._log_t = now
            logger.info("brb: evicted %d orphan blocks, %d by policy, %d by LRU fallback; %d paused "
                        "sessions hold blocks", self.evicted_orphans, self.evicted_by_policy,
                        self.evicted_by_lru, len(self.chains))
        return out
