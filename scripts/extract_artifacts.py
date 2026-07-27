#!/usr/bin/env python3
"""Recover a run's raw artifacts from its log, verifying every hash.

Local-mode projects have no artifact upload channel, so ``repro/run_all.py``
inlines every CSV/JSON it produced into the run log between delimiters:

    <<<ARTIFACT BEGIN <relpath> sha256=<hex> bytes=<n>>>>
    ...file content...
    <<<ARTIFACT END <relpath>>>>

This reverses that. The sha256 recorded at write time is recomputed over the
recovered bytes, so a truncated or interleaved log is a hard error rather than
a silently short file -- which matters, because these recovered files are the
inputs to the figures and the published report.

Usage:
    orx logs <runId> --head --bytes 2000000 > run.log
    python scripts/extract_artifacts.py run.log <outdir>
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

BEGIN = re.compile(r"^<<<ARTIFACT BEGIN (?P<path>\S+) sha256=(?P<sha>[0-9a-f]{64}) bytes=(?P<n>\d+)>>>$")
END = re.compile(r"^<<<ARTIFACT END (?P<path>\S+)>>>$")


def extract(log_path: str, outdir: str) -> int:
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")

    written = failed = 0
    i = 0
    while i < len(lines):
        m = BEGIN.match(lines[i].strip())
        if not m:
            i += 1
            continue
        rel, want_sha, want_n = m["path"], m["sha"], int(m["n"])
        body: list[str] = []
        i += 1
        while i < len(lines) and not END.match(lines[i].strip()):
            body.append(lines[i])
            i += 1
        if i >= len(lines):
            print(f"  TRUNCATED (no END marker): {rel}", file=sys.stderr)
            failed += 1
            break
        i += 1

        data = ("\n".join(body) + "\n").encode()
        got_sha = hashlib.sha256(data).hexdigest()
        if got_sha != want_sha or len(data) != want_n:
            # Do not write a file that does not match what the run recorded.
            # A near-miss here almost always means the log was fetched with a
            # byte budget that clipped the middle of the artifact.
            print(f"  HASH MISMATCH {rel}: bytes {len(data)} vs {want_n}, "
                  f"sha {got_sha[:12]} vs {want_sha[:12]}", file=sys.stderr)
            failed += 1
            continue

        dest = os.path.join(outdir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as out:
            out.write(data)
        print(f"  ok {rel}  ({want_n} bytes, sha256 {want_sha[:12]}...)")
        written += 1

    print(f"\nrecovered {written} artifacts, {failed} failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(extract(sys.argv[1], sys.argv[2]))
