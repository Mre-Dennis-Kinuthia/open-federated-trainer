"""Compute runtime abstraction (Milestone 6).

Default: allowlisted local importlib. Container runtime is a stub until wired.
"""

from __future__ import annotations

import importlib
import os
import re
import time
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

# Dotted module + function identifiers only — no paths, URLs, or relative imports.
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_FUNC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RuntimeError_(Exception):
    """Raised when a compute runtime cannot execute a plugin safely."""


class EntrypointRejected(RuntimeError_):
    """Network/path-supplied or otherwise unsafe entrypoint."""


def parse_entrypoint(entrypoint: str) -> tuple[str, str]:
    """Parse and sanitize ``module.path:function``; reject path/URL forms."""
    text = (entrypoint or "").strip()
    if not text:
        raise EntrypointRejected("empty entrypoint")
    if any(bad in text for bad in ("..", "/", "\\", "\x00", "://", " ")):
        raise EntrypointRejected(
            "entrypoint must be a dotted module:function (no paths or URLs)"
        )
    if text.count(":") != 1:
        raise EntrypointRejected("entrypoint must be exactly module.path:function")
    module_name, function_name = text.split(":", 1)
    if not _MODULE_RE.match(module_name):
        raise EntrypointRejected(f"invalid module identifier: {module_name!r}")
    if not _FUNC_RE.match(function_name):
        raise EntrypointRejected(f"invalid function identifier: {function_name!r}")
    return module_name, function_name


def allowlist_modules() -> list[str]:
    return [
        item.strip()
        for item in os.getenv("COMPUTE_PLUGIN_ALLOWLIST", "").split(",")
        if item.strip()
    ]


def assert_allowlisted(module_name: str, allowed: Optional[list[str]] = None) -> None:
    prefixes = allowed if allowed is not None else allowlist_modules()
    if not prefixes:
        raise EntrypointRejected(
            "COMPUTE_PLUGIN_ALLOWLIST is empty; refusing all compute plugins"
        )
    if not any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in prefixes
    ):
        raise EntrypointRejected(
            f"Compute module {module_name!r} is not allowlisted"
        )


@runtime_checkable
class ComputeRuntime(Protocol):
    name: str

    def execute(self, entrypoint: str, work_unit: Dict[str, Any]) -> Dict[str, Any]:
        """Run allowlisted compute; return JSON-serializable result dict."""


class LocalImportRuntime:
    """In-process allowlisted importlib runtime (demo / default)."""

    name = "local_import"

    def execute(self, entrypoint: str, work_unit: Dict[str, Any]) -> Dict[str, Any]:
        module_name, function_name = parse_entrypoint(entrypoint)
        assert_allowlisted(module_name)
        module = importlib.import_module(module_name)
        function = getattr(module, function_name, None)
        if not callable(function):
            raise EntrypointRejected(f"Compute entrypoint is not callable: {entrypoint}")
        started = time.monotonic()
        result = function(work_unit)
        return {
            "result": result,
            "runtime": self.name,
            "entrypoint": entrypoint,
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }


class ContainerRuntime:
    """Stub for future OCI/container isolation (digest-pinned images)."""

    name = "container"

    def execute(self, entrypoint: str, work_unit: Dict[str, Any]) -> Dict[str, Any]:
        parse_entrypoint(entrypoint)  # still sanitize
        raise RuntimeError_(
            "COMPUTE_RUNTIME=container is not wired yet; use local_import "
            "(allowlisted in-process plugins) or set COMPUTE_RUNTIME=local_import"
        )


def get_compute_runtime(name: Optional[str] = None) -> ComputeRuntime:
    key = (name or os.getenv("COMPUTE_RUNTIME", "local_import") or "local_import").strip().lower()
    if key in {"local", "local_import", "importlib"}:
        return LocalImportRuntime()
    if key in {"container", "docker", "oci"}:
        return ContainerRuntime()
    raise RuntimeError_(f"Unknown COMPUTE_RUNTIME={key!r}")
