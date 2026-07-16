"""
Local process launcher for UI-started clients and job workers.

Enabled when ENABLE_LOCAL_LAUNCHER=true (default). Disable in multi-tenant
production. Only spawns known scripts under the repo client/ tree.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    # coordinator/src/core/local_launcher.py → repo root
    return Path(__file__).resolve().parents[3]


def _client_root() -> Path:
    return _repo_root() / "client"


def _client_python() -> Path:
    venv = _client_root() / "venv" / "bin" / "python"
    if venv.exists():
        return venv
    return Path(os.environ.get("CLIENT_PYTHON", "python3"))


def _coordinator_url() -> str:
    return os.getenv(
        "PUBLIC_COORDINATOR_URL",
        os.getenv("COORDINATOR_URL", "http://127.0.0.1:8000"),
    )


@dataclass
class ManagedProcess:
    id: str
    kind: str  # train | worker
    name: str
    pid: int
    started_at: float
    cmd: List[str]
    env_summary: Dict[str, str] = field(default_factory=dict)
    log_path: Optional[str] = None
    running: bool = True
    exit_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["uptime_seconds"] = round(time.time() - self.started_at, 1) if self.running else None
        return d


class LocalLauncher:
    """Spawn and track local train clients / job workers."""

    ALLOWED_KINDS = {"train", "worker"}
    DATASET_PRESETS = {
        "none": "",
        "sample_private": "examples/sample_private.csv",
        "sample_tabular": "examples/sample_tabular.csv",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: Dict[str, subprocess.Popen] = {}
        self._meta: Dict[str, ManagedProcess] = {}
        self._log_dir = _repo_root() / "coordinator" / "logs" / "launch"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return os.getenv("ENABLE_LOCAL_LAUNCHER", "false").lower() in ("1", "true", "yes")

    def list(self) -> List[Dict[str, Any]]:
        self._refresh()
        with self._lock:
            return [m.to_dict() for m in sorted(self._meta.values(), key=lambda m: m.started_at, reverse=True)]

    def status(self) -> Dict[str, Any]:
        items = self.list()
        running = [p for p in items if p.get("running")]
        return {
            "enabled": self.enabled,
            "running": len(running),
            "total": len(items),
            "by_kind": {
                "train": sum(1 for p in running if p["kind"] == "train"),
                "worker": sum(1 for p in running if p["kind"] == "worker"),
            },
            "processes": items,
            "dataset_presets": list(self.DATASET_PRESETS.keys()),
        }

    def start(
        self,
        kind: str,
        *,
        count: int = 1,
        model_id: Optional[str] = None,
        model_module: Optional[str] = None,
        dataset_preset: str = "none",
        dataset_path: Optional[str] = None,
        job_types: str = "inference,label,compute",
        client_name_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("Local launcher disabled (set ENABLE_LOCAL_LAUNCHER=true)")
        if kind not in self.ALLOWED_KINDS:
            raise ValueError(f"kind must be one of {sorted(self.ALLOWED_KINDS)}")
        count = max(1, min(int(count), 8))

        started: List[Dict[str, Any]] = []
        for i in range(count):
            name = f"{client_name_prefix or kind}-{uuid.uuid4().hex[:6]}"
            if count > 1:
                name = f"{client_name_prefix or kind}-{i + 1}-{uuid.uuid4().hex[:4]}"
            meta = self._spawn_one(
                kind=kind,
                name=name,
                model_id=model_id,
                model_module=model_module,
                dataset_preset=dataset_preset,
                dataset_path=dataset_path,
                job_types=job_types,
            )
            started.append(meta.to_dict())
        return started

    def stop(self, process_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(process_id)
            meta = self._meta.get(process_id)
            if not proc or not meta:
                return False
            self._terminate(proc)
            meta.running = False
            meta.exit_code = proc.poll()
            del self._procs[process_id]
            return True

    def stop_all(self, kind: Optional[str] = None) -> int:
        self._refresh()
        stopped = 0
        with self._lock:
            ids = list(self._procs.keys())
        for pid in ids:
            meta = self._meta.get(pid)
            if meta and kind and meta.kind != kind:
                continue
            if self.stop(pid):
                stopped += 1
        return stopped

    def _resolve_dataset(
        self,
        dataset_preset: str,
        dataset_path: Optional[str],
    ) -> Optional[str]:
        if dataset_path and dataset_path.strip():
            p = Path(dataset_path.strip()).expanduser()
            if not p.is_absolute():
                p = (_client_root() / p).resolve()
            else:
                p = p.resolve()
            # Confine to client tree
            try:
                p.relative_to(_client_root().resolve())
            except ValueError as e:
                raise ValueError("dataset_path must be under client/") from e
            if not p.exists():
                raise FileNotFoundError(f"Dataset not found: {p}")
            return str(p)

        preset = self.DATASET_PRESETS.get(dataset_preset or "none", "")
        if not preset:
            return None
        p = (_client_root() / preset).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Preset dataset missing: {p}")
        return str(p)

    def _spawn_one(
        self,
        *,
        kind: str,
        name: str,
        model_id: Optional[str],
        model_module: Optional[str],
        dataset_preset: str,
        dataset_path: Optional[str],
        job_types: str,
    ) -> ManagedProcess:
        client_root = _client_root()
        src = client_root / "src"
        py = str(_client_python())
        script = "client.py" if kind == "train" else "worker.py"
        script_path = src / script
        if not script_path.exists():
            raise FileNotFoundError(f"Missing {script_path}")

        env = os.environ.copy()
        env["COORDINATOR_URL"] = _coordinator_url()
        env["CLIENT_NAME"] = name
        env["PYTHONPATH"] = f"{src}:{client_root}:{env.get('PYTHONPATH', '')}"
        env["SLEEP_BETWEEN_ROUNDS"] = env.get("SLEEP_BETWEEN_ROUNDS", "5")
        env["JOB_POLL_SECONDS"] = env.get("JOB_POLL_SECONDS", "3")

        summary: Dict[str, str] = {
            "COORDINATOR_URL": env["COORDINATOR_URL"],
            "CLIENT_NAME": name,
        }

        if kind == "train":
            mid = (model_id or "simple_mlp").strip()
            env["MODEL_ID"] = mid
            summary["MODEL_ID"] = mid
            if model_module:
                env["MODEL_MODULE"] = model_module.strip()
                summary["MODEL_MODULE"] = env["MODEL_MODULE"]
            elif mid == "custom":
                env["MODEL_MODULE"] = "examples.custom_linear:CustomLinearTrainer"
                env["MODEL_ID"] = "custom"
                summary["MODEL_MODULE"] = env["MODEL_MODULE"]
            ds = self._resolve_dataset(dataset_preset, dataset_path)
            if ds:
                env["DATASET_PATH"] = ds
                env["DATASET_FORMAT"] = "auto"
                summary["DATASET_PATH"] = ds
        else:
            types = job_types.strip() or "inference,label,compute"
            env["JOB_TYPES"] = types
            env["WORK_MODES"] = types
            env["COMPUTE_PLUGIN_ALLOWLIST"] = env.get(
                "COMPUTE_PLUGIN_ALLOWLIST",
                "examples.science_plugin",
            )
            summary["JOB_TYPES"] = types
            summary["COMPUTE_PLUGIN_ALLOWLIST"] = env["COMPUTE_PLUGIN_ALLOWLIST"]
            ds = self._resolve_dataset(dataset_preset, dataset_path)
            if ds:
                env["DATASET_PATH"] = ds
                env["DATASET_FORMAT"] = "auto"
                summary["DATASET_PATH"] = ds

        proc_id = uuid.uuid4().hex[:10]
        log_path = self._log_dir / f"{kind}_{proc_id}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        cmd = [py, str(script_path)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(src),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        meta = ManagedProcess(
            id=proc_id,
            kind=kind,
            name=name,
            pid=proc.pid,
            started_at=time.time(),
            cmd=cmd,
            env_summary=summary,
            log_path=str(log_path),
            running=True,
        )
        with self._lock:
            self._procs[proc_id] = proc
            self._meta[proc_id] = meta
        return meta

    def _refresh(self) -> None:
        with self._lock:
            for pid, proc in list(self._procs.items()):
                code = proc.poll()
                meta = self._meta.get(pid)
                if code is not None and meta:
                    meta.running = False
                    meta.exit_code = code
                    del self._procs[pid]

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()


_launcher: Optional[LocalLauncher] = None


def get_local_launcher() -> LocalLauncher:
    global _launcher
    if _launcher is None:
        _launcher = LocalLauncher()
    return _launcher
