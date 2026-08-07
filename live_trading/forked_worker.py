"""Linux fork workers that share imported runtime pages copy-on-write."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Callable, TextIO


def worker_lifecycle_line(
    action: str, *, strategy: str, pid: int, launcher: str, game_pk: int,
    home_ticker: str, away_ticker: str, status: int | None = None,
) -> str:
    fields = [
        f"WORKER {action.upper()}", f"strategy={strategy}", f"pid={pid}",
        f"launcher={launcher}",
    ]
    if status is not None:
        fields.append(f"status={status}")
    fields.extend((
        f"game_pk={game_pk}", f"home_ticker={home_ticker}",
        f"away_ticker={away_ticker}",
    ))
    return " ".join(fields)


@dataclass
class ForkedWorker:
    pid: int
    stdout: TextIO
    returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        found, status = os.waitpid(self.pid, os.WNOHANG)
        if found:
            self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.05)
        return int(self.returncode)

    def terminate(self) -> None:
        if self.poll() is None:
            os.kill(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        if self.poll() is None:
            os.kill(self.pid, signal.SIGKILL)


def spawn_forked_worker(
    target: Callable[[], None], env: dict[str, str],
) -> ForkedWorker:
    """Fork an imported coordinator and capture the child's combined output."""
    if not hasattr(os, "fork"):
        raise RuntimeError("Shared-import workers require Linux fork support")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            os.dup2(write_fd, 1)
            os.dup2(write_fd, 2)
            os.close(write_fd)
            os.environ.clear()
            os.environ.update(env)
            target()
        except BaseException as error:
            print(f"Forked worker failed: {error!r}", flush=True)
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    os.close(write_fd)
    return ForkedWorker(
        pid, os.fdopen(read_fd, "r", buffering=1, errors="replace")
    )
