"""
Server-side usage limits keyed by the authenticated username.

Because the state lives in the server process (not in the WebSocket connection or the browser),
reconnecting, opening a second tab or refreshing the page does not reset anything.
Note: state is in memory, so it resets when the server restarts and is not shared between
processes. That matches the single-worker deployment that local-mode Qdrant already requires;
use Redis if you ever scale out.
"""
import threading
import time
from collections import defaultdict, deque
from typing import Callable


class RateLimitStore:
    """Sliding window: at most `max_requests` per `window_seconds` for each key."""

    def __init__(self, max_requests: int, window_seconds: int, clock: Callable[[], float] = time.time):
        self.max_requests, self.window, self.clock = max_requests, window_seconds, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._calls = 0

    def check(self, key: str) -> tuple[bool, int]:
        """Records the request if allowed. Returns (allowed, retry_after_seconds)."""
        with self._lock:
            now = self.clock()
            self._calls += 1
            if self._calls % 200 == 0:
                self._purge(now)
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) < self.max_requests:
                q.append(now)
                return True, 0
            return False, int(self.window - (now - q[0])) + 1

    def _purge(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window]:
            del self._hits[k]


class TokenBudget:
    """Rolling token allowance per key (default: 10 000 tokens per 24 h)."""

    def __init__(self, limit: int, window_seconds: int, clock: Callable[[], float] = time.time):
        self.limit, self.window, self.clock = limit, window_seconds, clock
        self._events: dict[str, deque] = defaultdict(deque)   # (timestamp, tokens)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque:
        q = self._events[key]
        while q and now - q[0][0] >= self.window:
            q.popleft()
        return q

    def used(self, key: str) -> int:
        with self._lock:
            return sum(t for _, t in self._prune(key, self.clock()))

    def remaining(self, key: str) -> int:
        return max(0, self.limit - self.used(key))

    def add(self, key: str, tokens: int) -> None:
        if tokens > 0:
            with self._lock:
                self._events[key].append((self.clock(), int(tokens)))
