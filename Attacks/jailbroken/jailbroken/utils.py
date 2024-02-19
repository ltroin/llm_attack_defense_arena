import json
import logging
import pathlib
import signal
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

EXPERIMENT_DIR: Optional[pathlib.Path] = None


def _run_subprocess(command: List[str]):
    """Execute command in shell and return output"""
    return subprocess.check_output(command).decode("utf-8").strip()


def get_hostname():
    """Get hostname"""
    return _run_subprocess(["hostname"])


def get_git_revision_hash():
    """Get current git revision hash"""
    return _run_subprocess(["git", "rev-parse", "HEAD"])


def get_git_diff():
    """Get git diff of all files except notebooks and shell scripts"""
    return _run_subprocess(["git", "diff", "HEAD", ":!*.ipynb", ":!*.sh", ":!*.json"])


def get_git_root():
    """Get git root directory"""
    return _run_subprocess(["git", "rev-parse", "--show-toplevel"])


def log_uncaught_errors():
    """Set up logging for uncaught errors"""

    def excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = excepthook


def log_replication_info():
    """Log command and git info"""
    replication_info = {
        "hostname": get_hostname(),
        "pwd": str(pathlib.Path.cwd()),
        "Command": f"python3 {' '.join(sys.argv)}",
        "Git hash": get_git_revision_hash(),
        "Git diff": get_git_diff(),
    }
    for key, value in replication_info.items():
        separator = "\n" if "\n" in value else " "
        logging.info(f"{key}:{separator}{value}")
    save_experiment_json("replication_info", replication_info)


def log_to_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} {levelname} [{filename:s}:{lineno:d}] {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        style="{",
    )
    log_uncaught_errors()


def init_experiment(
    handle: str,
    experiments: Optional[pathlib.Path] = None,
    log_to_file=True,
    log_to_stdout=True,
):
    """Initialize experiment"""
    if experiments is None:
        experiments = pathlib.Path(get_git_root()) / "experiments"

    global EXPERIMENT_DIR
    assert EXPERIMENT_DIR is None
    datetime_now = datetime.now()
    date = datetime_now.strftime("%Y%m%d")
    timestamp = int(datetime_now.timestamp())
    EXPERIMENT_DIR = experiments / handle / f"{date}-{timestamp}"
    EXPERIMENT_DIR.mkdir(parents=True)

    logfile = EXPERIMENT_DIR / "default.log"
    handlers = []
    if log_to_file:
        handlers.append(logging.FileHandler(filename=logfile))
    if log_to_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} {levelname} [{filename:s}:{lineno:d}] {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        style="{",
    )
    if log_to_file:
        logging.info(f"Logging to {logfile}")

    log_replication_info()
    log_uncaught_errors()

    # Symlink to latest experiment
    latest = experiments / "_latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(EXPERIMENT_DIR.relative_to(experiments))


def save_experiment_json(handle: str, data, partial: bool = False):
    assert EXPERIMENT_DIR is not None

    if partial:
        # Log partial results as "{handle}.partial.json"
        handle = f"{handle}.partial"
    else:
        # Remove partial file if it exists
        (EXPERIMENT_DIR / f"{handle}.partial.json").unlink(missing_ok=True)

    filename = EXPERIMENT_DIR / f"{handle}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved experiment {handle} to {filename}")


def save_datapoint_jsonl(handle: str, datapoint):
    assert EXPERIMENT_DIR is not None

    filename = EXPERIMENT_DIR / f"{handle}.jsonl"
    with open(filename, "a") as f:
        f.write(f"{json.dumps(datapoint)}\n")
    logging.info(f"Added datapoint to experiment {handle} at {filename}")


class Timeout:
    """Timeout class using SIGALRM"""

    class Timeout(Exception):
        pass

    def __init__(self, seconds=1, error_message="Call timed out!"):
        self._seconds = seconds
        self._error_message = error_message

    def __enter__(self):
        def signal_handler(signum, frame):
            raise Timeout.Timeout(self._error_message)

        # Set the signal handler and alarm
        self._old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self._seconds)

    def __exit__(self, type, value, traceback):
        # Reset the alarm
        signal.alarm(0)

        # Reset the signal handler to its previous value
        signal.signal(signal.SIGALRM, self._old_handler)
