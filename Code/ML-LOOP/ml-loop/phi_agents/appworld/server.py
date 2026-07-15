#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import signal
import subprocess
from pathlib import Path

from phi_agents.utils.logger import get_phi_logger

DEFAULT_APPWORLD_BIN = "appworld-env/bin/appworld"


logger = get_phi_logger()


def launch_appworld_environment_server(
    port: int | None, docker: bool, appworld_root: str | None, stdout_to_devnull: bool
) -> subprocess.Popen[bytes]:
    """Launch an AppWorld server.

    Run `appworld serve --help` for more details.

    Args:
        port: .
        docker: Whether to run it in a docker container.
        appworld_root: .
        stdout_to_devnull: Whether to redirect stdout to devnull.
    """
    appworld_cmd = DEFAULT_APPWORLD_BIN if Path(DEFAULT_APPWORLD_BIN).exists() else "appworld"
    args = [appworld_cmd, "serve", "environment", "--no-show-usage"]
    if port:
        args.extend(["--port", str(port)])
    if docker:
        args.append("--docker")
    if appworld_root:
        args.extend(["--root", appworld_root])
    
    # Always log the output to a file for debugging, unless stdout_to_devnull is specifically set
    # and we don't care about debugging. But wait, during launch failures, we always want logs.
    # So we write to logs/appworld/server_{port}.log.
    log_dir = Path("logs/appworld")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"server_{port or 'default'}.log"
    
    f = open(log_file_path, "w", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
    f.close()
    return proc


def stop_appworld_environment_server(popen: subprocess.Popen[bytes]) -> None:
    logger.info("sending SIGINT command to the appworld subprocess...")
    popen.send_signal(signal.SIGINT)
    popen.wait(timeout=20)
    logger.info("appworld server terminated")


def force_stop_appworld_environment_server(popen: subprocess.Popen[bytes]) -> None:
    logger.info("sending SIGKILL command to the appworld subprocess...")
    popen.send_signal(signal.SIGKILL)
    popen.wait(timeout=10)
    logger.info("appworld server forcefully terminated")
