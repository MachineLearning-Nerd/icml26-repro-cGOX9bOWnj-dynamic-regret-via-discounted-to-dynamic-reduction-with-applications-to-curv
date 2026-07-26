"""Shared scaffolding: artifact paths, environment stamping, verdict types.

Every claim verifier in this repository follows the same contract:

    run(ctx) -> ClaimResult

and writes its raw, machine-readable evidence under
``.openresearch/artifacts/<claim_id>/``. Nothing is reported as a result unless
it was written to disk first, so the independent checker (``repro/checker.py``)
can re-derive every headline number from the raw files alone.

Verdicts are exactly one of VERIFIED / FALSIFIED / BLOCKED. There is no PASS.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT_ROOT = os.path.join(REPO_ROOT, ".openresearch", "artifacts")

VERIFIED = "VERIFIED"
FALSIFIED = "FALSIFIED"
BLOCKED = "BLOCKED"
VALID_VERDICTS = (VERIFIED, FALSIFIED, BLOCKED)


@dataclasses.dataclass
class ClaimResult:
    """Outcome of one claim verifier.

    ``ok`` is the gate the run command exits on. It is True only when the
    verifier reached a defensible terminal verdict *and* every negative control
    behaved as designed. A BLOCKED verdict is an honest outcome and does not by
    itself fail the run; a negative control that failed to fail always does,
    because it means the test had no power.
    """

    claim_id: str
    title: str
    verdict: str
    ok: bool
    headline: dict[str, Any]
    controls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)
    runtime_s: float = 0.0

    def __post_init__(self) -> None:
        if self.verdict not in VALID_VERDICTS:
            raise ValueError(
                f"{self.claim_id}: verdict {self.verdict!r} is not one of {VALID_VERDICTS}. "
                "Toy/skipped/inconclusive evidence must never be relabelled as a pass."
            )


def artifact_dir(claim_id: str) -> str:
    d = os.path.join(ARTIFACT_ROOT, claim_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(claim_id: str, name: str, payload: Any) -> str:
    path = os.path.join(artifact_dir(claim_id), name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")
    return path


def write_csv(claim_id: str, name: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> str:
    path = os.path.join(artifact_dir(claim_id), name)
    if not rows:
        # An empty sweep is a bug, not an empty file: make it loud rather than
        # writing a zero-row CSV that later reads as "nothing violated".
        raise ValueError(f"{claim_id}/{name}: refusing to write an empty results table")
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_text(claim_id: str, name: str, text: str) -> str:
    path = os.path.join(artifact_dir(claim_id), name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        raise TypeError(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(o)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT_SHA", "unknown")


def cpu_info() -> dict[str, Any]:
    """Record the actual CPU allocation, not the requested one.

    ``os.cpu_count()`` reports the host; ``os.sched_getaffinity`` (Linux) reports
    what this process may actually use, which is what a container-scheduled HF
    job really got. Both are recorded so the evidence states the true allocation.
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "os_cpu_count": os.cpu_count(),
        "blas_threads_env": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        info["sched_affinity_cpus"] = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        info["sched_affinity_cpus"] = None
    try:
        with open("/proc/cpuinfo") as fh:
            models = [l.split(":", 1)[1].strip() for l in fh if l.startswith("model name")]
        info["cpu_model"] = models[0] if models else None
        info["proc_cpuinfo_entries"] = len(models)
    except OSError:
        info["cpu_model"] = None
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except ImportError:
        pass
    return info


def environment_stamp() -> dict[str, Any]:
    stamp = {
        "git_sha": git_sha(),
        "run_command": "bash scripts/run.sh",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": cpu_info(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    try:
        stamp["frozen_packages"] = sorted(
            subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
            ).split()
        )
    except Exception:
        stamp["frozen_packages"] = None
    return stamp


class Timer:
    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a: Any) -> None:
        self.elapsed = time.perf_counter() - self.t0


def banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)
