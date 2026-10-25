"""Add the /execute_with_bookmark route that VCBM needs to the AppWorld environment server.

Patches the `appworld` installation the training job will actually launch, resolved the same
way as phi_agents/appworld/server.py: `appworld-env/bin/appworld` relative to the current
directory if it exists, otherwise `appworld` on PATH. Pass a path to environment.py, or set
APPWORLD_ENV_SERVER_PATH, to patch a specific file instead.

Exits non-zero when it cannot patch, so job scripts can stop instead of training without
bookmarks. (An earlier version hardcoded a python3.12 path and exited 0 on failure; both
published AppWorld runs trained with an unpatched server as a result.)

Usage:
    python scripts/patch_appworld_server.py [path/to/appworld/serve/environment.py]
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_APPWORLD_BIN = "appworld-env/bin/appworld"

NEW_ENDPOINT = """

@app.post("/execute_with_bookmark")
async def execute_with_bookmark(task_id: str = Body(...), code: str = Body(...)) -> dict[str, Any]:
    maybe_raise_exception(task_id)
    n_requests_before = len(world.requester.request_tracker.requests)
    output = world.execute(code)
    new_requests = world.requester.request_tracker.requests[n_requests_before:]
    is_bookmark = False
    if "Execution failed." not in output:
        is_bookmark = any(
            req.get("method", "").upper() in ("POST", "PUT", "DELETE", "PATCH")
            for req in new_requests
        )
    return {"output": {"output": output, "is_bookmark": is_bookmark}}"""


def find_cli_python(appworld_bin: Path) -> str:
    """Return the Python interpreter that runs the appworld CLI.

    pip writes `#!/path/to/python` for normal paths, but when the venv path contains spaces
    it writes a `#!/bin/sh` wrapper that re-execs Python, so the shebang alone is not enough.
    """
    sibling = appworld_bin.parent / "python"
    if sibling.exists():
        return str(sibling)
    lines = appworld_bin.read_text(errors="replace").splitlines()
    if lines and lines[0].startswith("#!") and "python" in lines[0]:
        return lines[0][2:].strip()
    for line in lines[1:3]:  # the /bin/sh wrapper: '''exec' "/path/python" "$0" "$@"
        if "exec" in line and '"' in line:
            return line.split('"')[1]
    return sys.executable


def find_environment_py() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if os.environ.get("APPWORLD_ENV_SERVER_PATH"):
        return Path(os.environ["APPWORLD_ENV_SERVER_PATH"])

    appworld_bin = DEFAULT_APPWORLD_BIN if Path(DEFAULT_APPWORLD_BIN).exists() else shutil.which("appworld")
    if not appworld_bin:
        sys.exit("Error: no appworld-env/bin/appworld here and no `appworld` on PATH.")

    python = find_cli_python(Path(appworld_bin))
    result = subprocess.run(
        [python, "-c", "import appworld, os; print(os.path.dirname(appworld.__file__))"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"Error: could not import appworld with {python}:\n{result.stderr}")
    print(f"appworld CLI: {appworld_bin}\nappworld package: {result.stdout.strip()}")
    return Path(result.stdout.strip()) / "serve" / "environment.py"


def patch_server(filepath: Path) -> None:
    if not filepath.exists():
        sys.exit(f"Error: {filepath} not found. Is appworld installed?")

    content = filepath.read_text()
    if "execute_with_bookmark" in content:
        print(f"Already patched: {filepath}")
        return

    idx = content.find('@app.post("/execute")')
    if idx == -1:
        sys.exit(f"Error: no @app.post('/execute') route in {filepath}")
    end_marker = 'return {"output": output}'
    end_idx = content.find(end_marker, idx)
    if end_idx == -1:
        sys.exit(f"Error: could not find the end of the execute route in {filepath}")

    end_of_func = end_idx + len(end_marker)
    filepath.write_text(content[:end_of_func] + NEW_ENDPOINT + content[end_of_func:])
    print(f"Patched: {filepath}")


if __name__ == "__main__":
    patch_server(find_environment_py())
