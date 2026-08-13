"""Peak-RSS tracking and enforcement.

MEMORY SAFETY IS A TOP PRIORITY (proposal.md): every stage runs under a hard
budget (default < 4 GB peak RSS). Two enforcement layers live here:

- :class:`PeakTracker` — in-process context manager wrapped around a stage body;
- :func:`watch_child` — the umbrella orchestrator's watchdog that kills a
  runaway child process at ``budget * kill_factor``.
"""

from __future__ import annotations

import os
import resource
import signal
import sys
import threading
import time
from types import TracebackType
from typing import Literal

import psutil
from loguru import logger

_GB = 1024**3


class MemoryBudgetError(RuntimeError):
    """A stage exceeded its peak-RSS budget."""


def current_rss_bytes(pid: int | None = None) -> int:
    proc = psutil.Process(pid) if pid is not None else psutil.Process()
    return int(proc.memory_info().rss)


def _ru_maxrss_bytes() -> int:
    """Lifetime peak RSS of this process. macOS reports bytes, Linux kilobytes."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def assert_rss_under(budget_gb: float, context: str) -> None:
    """Point-in-time check; raises MemoryBudgetError if current RSS exceeds budget."""
    rss = current_rss_bytes()
    if rss > budget_gb * _GB:
        raise MemoryBudgetError(
            f"{context}: RSS {rss / _GB:.2f} GB exceeds budget {budget_gb:.2f} GB"
        )


class PeakTracker:
    """Sample RSS on a background thread; report (and optionally enforce) the peak.

    Peak = max(sampled peak, ru_maxrss) — ru_maxrss catches spikes shorter than
    the sampling interval, but is process-lifetime, so the sampled peak is the
    per-stage number when multiple stages run in one process (the umbrella
    avoids that by running one stage per subprocess).
    """

    def __init__(
        self,
        stage: str,
        budget_gb: float = 4.0,
        interval_s: float = 0.2,
        hard: bool = True,
    ) -> None:
        self.stage = stage
        self.budget_gb = budget_gb
        self.interval_s = interval_s
        self.hard = hard
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = int(proc.memory_info().rss)
            except psutil.Error:  # pragma: no cover - process teardown race
                break
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PeakTracker:
        self.peak_rss_bytes = current_rss_bytes()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, name=f"peak-rss-{self.stage}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_rss_bytes = max(self.peak_rss_bytes, _ru_maxrss_bytes())
        over = self.peak_rss_bytes > self.budget_gb * _GB
        logger.log(
            "WARNING" if over else "INFO",
            "{} peak_rss={:.2f} GB (budget {:.1f} GB){}",
            self.stage,
            self.peak_rss_bytes / _GB,
            self.budget_gb,
            " — OVER BUDGET" if over else "",
        )
        if over and self.hard and exc_type is None:
            raise MemoryBudgetError(
                f"{self.stage}: peak RSS {self.peak_rss_bytes / _GB:.2f} GB "
                f"exceeds budget {self.budget_gb:.2f} GB"
            )
        return False  # never swallow exceptions


class ChildWatch:
    """Handle for a watched child process (see :func:`watch_child`)."""

    def __init__(self, pid: int, budget_gb: float, kill_factor: float, interval_s: float) -> None:
        self.pid = pid
        self.budget_gb = budget_gb
        self.kill_factor = kill_factor
        self.interval_s = interval_s
        self.peak_rss_bytes = 0
        self.killed_for_memory = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"watch-{pid}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except psutil.Error:
            return
        limit = self.budget_gb * self.kill_factor * _GB
        while not self._stop.is_set():
            try:
                rss = int(proc.memory_info().rss)
            except psutil.Error:
                return  # child exited
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            if rss > limit:
                self.killed_for_memory = True
                logger.error(
                    "child pid={} RSS {:.2f} GB > {:.2f} GB limit — SIGTERM",
                    self.pid,
                    rss / _GB,
                    limit / _GB,
                )
                try:
                    os.kill(self.pid, signal.SIGTERM)
                    time.sleep(5.0)
                    if proc.is_running():
                        os.kill(self.pid, signal.SIGKILL)
                except (ProcessLookupError, psutil.Error):
                    pass
                return
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def watch_child(
    pid: int, budget_gb: float, kill_factor: float = 1.25, interval_s: float = 0.5
) -> ChildWatch:
    """Watch a child process; SIGTERM (then SIGKILL) it past ``budget * kill_factor``."""
    return ChildWatch(pid, budget_gb, kill_factor, interval_s)
