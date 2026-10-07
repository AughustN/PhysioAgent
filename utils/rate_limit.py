from __future__ import annotations

import random
import threading
import time
from typing import Optional

RATE_LIMIT_PER_MIN = 5
RATE_LIMIT_WINDOW_S = 60.0
RATE_LIMIT_COOLDOWN_S = 65.0


class RateLimiter:
    def __init__(self, per_min: Optional[int] = None, window: Optional[float] = None,
                 cooldown: Optional[float] = None) -> None:
        self.per_min = RATE_LIMIT_PER_MIN if per_min is None else per_min
        self.window = RATE_LIMIT_WINDOW_S if window is None else window
        self.cooldown = RATE_LIMIT_COOLDOWN_S if cooldown is None else cooldown
        self._hits: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                self._hits = [t for t in self._hits if now - t < self.window]
                if len(self._hits) < self.per_min:
                    self._hits.append(now)
                    return
                sleep_for = self.window - (now - self._hits[0]) + 0.05
            time.sleep(max(sleep_for, 0.05) + random.uniform(0.0, 2.0))

    def penalise(self, seconds: Optional[float] = None) -> None:
        seconds = self.cooldown if seconds is None else seconds
        with self._lock:
            now = time.time()
            self._hits = [now + seconds - self.window] * self.per_min
