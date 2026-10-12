"""Run pytest on a grading directory in an isolated environment.

Two backends:

* ``DockerSandbox`` (primary): a container from ``python:3.12-slim`` plus
  pytest, with no network, memory/CPU/PID limits, a read-only root
  filesystem, all capabilities dropped, an unprivileged user, and a wall-clock
  timeout. The grading directory is the only writable bind mount.
* ``SubprocessSandbox`` (fallback): a plain subprocess in the grading directory
  with a timeout. This is WEAKER isolation: the code runs as your user, with
  your network and filesystem. Use it only for trusted code or quick local
  iteration.

The sandbox only ever sees the grading directory. The agent workspace is never
mounted; the grader copies the solution out of it first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

RUNNER_IMAGE = "rlsandbox-runner:py312"
DOCKERFILE_DIR = Path(__file__).parent / "docker"

PYTEST_ARGS = ["-q", "-p", "no:cacheprovider", "--junitxml", "report.xml", "--rootdir", "."]


@dataclass
class SandboxResult:
    backend: str
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stdout"] = d["stdout"][-2000:]
        d["stderr"] = d["stderr"][-2000:]
        return d


class Sandbox:
    name = "base"
    isolation = "none"

    def run_pytest(self, workdir: Path, timeout_s: float) -> SandboxResult:
        raise NotImplementedError


class DockerSandbox(Sandbox):
    name = "docker"
    isolation = "container: --network none, read-only root, memory/cpu/pids limits, cap-drop ALL, uid 65534"

    def __init__(self, memory: str = "256m", cpus: str = "1", pids: int = 64, image: str = RUNNER_IMAGE):
        self.memory, self.cpus, self.pids, self.image = memory, cpus, pids, image
        self._image_ready = False

    @staticmethod
    def available() -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                               capture_output=True, text=True, timeout=20)
            return r.returncode == 0 and r.stdout.strip() != ""
        except (OSError, subprocess.TimeoutExpired):
            return False

    def ensure_image(self) -> None:
        if self._image_ready:
            return
        r = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True, text=True)
        if r.returncode != 0:
            # One-time build. This step needs network access to pull the base image and pytest.
            b = subprocess.run(["docker", "build", "-t", self.image, str(DOCKERFILE_DIR)],
                               capture_output=True, text=True, timeout=900)
            if b.returncode != 0:
                raise RuntimeError(f"docker build failed:\n{b.stderr[-3000:]}")
        self._image_ready = True

    def run_pytest(self, workdir: Path, timeout_s: float) -> SandboxResult:
        self.ensure_image()
        name = f"rlsandbox-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--memory", self.memory, "--memory-swap", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids),
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "-e", "HOME=/tmp",
            "-v", f"{Path(workdir).resolve()}:/work:rw",
            "-w", "/work",
            self.image,
            "python", "-m", "pytest", *PYTEST_ARGS,
        ]
        start = time.monotonic()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            return SandboxResult(self.name, r.returncode, r.stdout, r.stderr, False, time.monotonic() - start)
        except subprocess.TimeoutExpired as e:
            subprocess.run(["docker", "kill", name], capture_output=True, text=True)
            return SandboxResult(self.name, None, _text(e.stdout), _text(e.stderr), True, time.monotonic() - start)


class SubprocessSandbox(Sandbox):
    name = "subprocess"
    isolation = "WEAK: local subprocess with timeout only; same user, network and filesystem as the host"

    def run_pytest(self, workdir: Path, timeout_s: float) -> SandboxResult:
        env = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE"}}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        start = time.monotonic()
        proc = subprocess.Popen([sys.executable, "-m", "pytest", *PYTEST_ARGS], cwd=str(workdir), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=timeout_s)
            return SandboxResult(self.name, proc.returncode, out, err, False, time.monotonic() - start)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            out, err = proc.communicate()
            return SandboxResult(self.name, None, out or "", err or "", True, time.monotonic() - start)


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def _text(x) -> str:
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else x


_CACHE: dict[str, Sandbox] = {}


def get_sandbox(kind: str | None = None) -> Sandbox:
    """Return a sandbox. ``kind`` is 'docker', 'subprocess' or 'auto' (default).

    The RLSANDBOX_SANDBOX environment variable overrides the default. 'auto'
    uses Docker when the daemon answers and falls back to a subprocess otherwise.
    """
    kind = (kind or os.environ.get("RLSANDBOX_SANDBOX") or "auto").lower()
    if kind in _CACHE:
        return _CACHE[kind]
    if kind == "docker":
        if not DockerSandbox.available():
            raise RuntimeError("Docker requested but the daemon is not reachable")
        sb: Sandbox = DockerSandbox()
    elif kind == "subprocess":
        sb = SubprocessSandbox()
    elif kind == "auto":
        sb = DockerSandbox() if DockerSandbox.available() else SubprocessSandbox()
    else:
        raise ValueError(f"unknown sandbox kind {kind!r}")
    _CACHE[kind] = sb
    return sb
