"""Fail fast unless the AppWorld server reports bookmarks correctly.

Starts the same AppWorld server the training job will use, then checks one read-only and one
state-changing call through execute_with_bookmark. Exits non-zero if the route is missing
(HTTP 404, which training would silently treat as "no bookmark") or if either call is
classified wrongly. Run it from the experiment folder with APPWORLD_ROOT set, before training.

Usage:
    APPWORLD_ROOT=... poetry run python scripts/preflight_bookmarks.py
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path().resolve()))

from phi_agents.appworld.interface import AppWorldInterface  # noqa: E402

TASK_ID = "82e2fac_1"  # any AppWorld train task works; this one is in the train split


def main() -> int:
    if not os.environ.get("APPWORLD_ROOT"):
        print("PREFLIGHT FAILED: APPWORLD_ROOT is not set.")
        return 1

    world = AppWorldInterface(stdout_to_devnull=True)
    try:
        world.initialize(TASK_ID, "preflight_bookmarks", raise_on_unsafe_syntax=False)
        checks = [
            ("print('hello')", False),
            ("apis.supervisor.complete_task()", True),
        ]
        for code, expected in checks:
            output, is_bookmark = world.execute_with_bookmark(code)
            print(f"{code!r}: is_bookmark={is_bookmark} (expected {expected})")
            if is_bookmark is not expected:
                print(
                    "PREFLIGHT FAILED: bookmark detection is not working. If the route is "
                    "missing, run `python scripts/patch_appworld_server.py` and resubmit."
                )
                return 1
    finally:
        world.close_server()

    print("PREFLIGHT OK: execute_with_bookmark is live and classifies calls correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
