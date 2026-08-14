#!/usr/bin/env python3
"""Supervise the FiqhMind backend processes inside one container.

Runs the web API, a single Celery worker (all 5 queues) and Celery Beat as
children of this process so all three share the same persistent disk.

Behavior:
- SIGTERM/SIGINT are forwarded to every child so uvicorn and Celery can
  perform their graceful (warm) shutdown.
- If any child exits, the remaining children are stopped and the container
  exits with that child's exit code so Render restarts on failure.
- No secrets or production URLs are hardcoded; everything comes from the
  environment (Render injects PORT automatically for web services).
"""

import os
import signal
import subprocess
import sys
import time

CELERY_APP = "app.worker.celery_app:celery_app"
WORKER_QUEUES = "ingest,extract,embed,index,maintenance"

PROCESSES = {
    "web": [
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        os.environ.get("PORT", "8000"),
    ],
    "worker": [
        "celery",
        "-A",
        CELERY_APP,
        "worker",
        "-Q",
        WORKER_QUEUES,
        "--concurrency",
        os.environ.get("CELERY_WORKER_CONCURRENCY", "1"),
        "--loglevel",
        "info",
    ],
    "beat": ["celery", "-A", CELERY_APP, "beat", "--loglevel", "info"],
}


def main() -> int:
    children: dict[str, subprocess.Popen] = {}
    for name, argv in PROCESSES.items():
        print(f"[entrypoint] starting {name}: {' '.join(argv)}", flush=True)
        children[name] = subprocess.Popen(argv)

    def forward(signum: int, frame: object) -> None:
        print(f"[entrypoint] received signal {signum}, forwarding to children", flush=True)
        for name, proc in children.items():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    status = 0
    failed_name: str | None = None
    while children:
        time.sleep(0.5)
        for name, proc in list(children.items()):
            code = proc.poll()
            if code is not None:
                print(f"[entrypoint] {name} exited with code {code}", flush=True)
                status = code
                failed_name = name
                del children[name]
                break
        else:
            continue
        break

    # One child finished: stop the survivors so the whole container exits and
    # Render restarts every process together.
    for name, proc in list(children.items()):
        if proc.poll() is None:
            print(f"[entrypoint] stopping {name}", flush=True)
            proc.terminate()

    for name, proc in list(children.items()):
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"[entrypoint] {name} did not stop in time, killing", flush=True)
            proc.kill()
            proc.wait()

    print(f"[entrypoint] {failed_name or 'shutdown'} complete, exiting {status}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
