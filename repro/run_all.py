"""The single fixed entry point invoked by `bash scripts/run.sh`.

Runs the CUMULATIVE claim-verification suite: every claim implemented on this
branch, including all claims accepted on ancestor branches. Adding a claim never
removes an earlier one -- a child that regresses a previously accepted claim
fails here, which is the regression gate the campaign depends on.

Exit code is 0 only when every implemented verifier reached a defensible verdict
AND every negative control failed in the way it was designed to fail.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time

from .common import ARTIFACT_ROOT, Timer, banner, environment_stamp, write_json

# (module, attribute) pairs, in claim order. Claims land here as branches add
# them; the list is the definition of "the cumulative regression suite".
CLAIM_MODULES = [
    "repro.claim1_theorem1",
    "repro.claim2_theorem2",
    "repro.claim3_theorem34",
    "repro.claim4_theorem5",
    "repro.claim5_theorem7",
]


def main() -> int:
    t_start = time.time()
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)

    banner("REPRODUCTION SUITE - arXiv 2602.08372 (cGOX9bOWnj)")
    stamp = environment_stamp()
    print(f"git sha        : {stamp['git_sha']}")
    print(f"python         : {stamp['cpu']['python']}")
    print(f"platform       : {stamp['cpu']['platform']}")
    print(f"cpu model      : {stamp['cpu'].get('cpu_model')}")
    print(f"os.cpu_count   : {stamp['cpu']['os_cpu_count']}")
    print(f"usable cpus    : {stamp['cpu']['sched_affinity_cpus']} (sched affinity)")
    print(f"BLAS threads   : {stamp['cpu']['blas_threads_env']}")
    print(f"utc            : {stamp['utc']}")
    write_json("_env", "environment.json", stamp)

    results = []
    for modname in CLAIM_MODULES:
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            # Not yet implemented on this branch. Recorded explicitly so the
            # log never silently omits a claim.
            print(f"\n[suite] {modname}: NOT IMPLEMENTED ON THIS BRANCH (skipped)", flush=True)
            results.append(
                {
                    "claim_id": modname.split(".")[-1],
                    "title": "(not implemented on this branch)",
                    "verdict": "BLOCKED",
                    "ok": True,
                    "headline": {"reason": "verifier not present on this git revision"},
                    "controls": [],
                    "notes": ["Implemented on a descendant branch; see the experiment tree."],
                    "runtime_s": 0.0,
                    "implemented": False,
                }
            )
            continue
        res = mod.run()
        d = {
            "claim_id": res.claim_id,
            "title": res.title,
            "verdict": res.verdict,
            "ok": res.ok,
            "headline": res.headline,
            "controls": res.controls,
            "notes": res.notes,
            "runtime_s": round(res.runtime_s, 3),
            "implemented": True,
        }
        results.append(d)
        print(f"\n[suite] {res.claim_id}: {res.verdict} (ok={res.ok}, {res.runtime_s:.1f}s)", flush=True)

    # Independent checker: re-derives every headline number from the raw
    # artifacts alone, without trusting anything the verifiers printed.
    banner("INDEPENDENT CHECKER")
    from . import checker

    check_ok, check_report = checker.run(results)
    write_json("_checker", "independent_check.json", check_report)
    for line in check_report.get("lines", []):
        print("  " + line, flush=True)
    print(f"\n[checker] all headline numbers reproduced from raw artifacts: {check_ok}", flush=True)

    total_s = time.time() - t_start
    summary = {
        "environment": stamp,
        "claims": results,
        "independent_checker_ok": check_ok,
        "total_runtime_s": round(total_s, 2),
    }
    write_json("_summary", "suite_summary.json", summary)
    _write_eval_md(summary)

    banner("SUMMARY")
    print("  verdict = the scientific outcome for the claim (VERIFIED / FALSIFIED / BLOCKED).")
    print("  instrument = whether the evidence pipeline itself was sound for that claim:")
    print("      no crashed configurations, and every negative control fired as designed.")
    print("  A BLOCKED verdict with a sound instrument is an honest result, NOT a run failure.\n")
    for r in results:
        flag = "sound" if r["ok"] else "UNSOUND"
        print(f"  {r['claim_id']:<24} {r['verdict']:<10} [instrument: {flag}]  {r['title']}")
    print(f"\n  independent checker : {'ok' if check_ok else 'FAIL'}")
    print(f"  total runtime       : {total_s:.1f}s")

    verdicts = {v: sum(1 for r in results if r["verdict"] == v)
                for v in ("VERIFIED", "FALSIFIED", "BLOCKED")}
    print(f"  verdict tally       : {verdicts['VERIFIED']} VERIFIED, "
          f"{verdicts['FALSIFIED']} FALSIFIED, {verdicts['BLOCKED']} BLOCKED")

    all_ok = all(r["ok"] for r in results) and check_ok
    print(f"\n  INSTRUMENT RESULT   : {'SOUND' if all_ok else 'UNSOUND'}"
          "   (this is the run's exit gate; verdicts are reported above)")

    # Local-mode projects have no artifact upload channel: the run log is the
    # only way evidence leaves the job. So dump every raw artifact inline, in a
    # delimited format that can be parsed back into identical files. This is
    # what makes the CSV/JSON evidence durable and independently re-checkable.
    _dump_artifacts()
    return 0 if all_ok else 1


ARTIFACT_DUMP_BUDGET_BYTES = 900_000


def _dump_artifacts() -> None:
    banner("RAW ARTIFACT DUMP (local mode: the log is the evidence channel)")
    files = []
    for root, _dirs, names in os.walk(ARTIFACT_ROOT):
        for n in sorted(names):
            p = os.path.join(root, n)
            files.append((os.path.relpath(p, ARTIFACT_ROOT), os.path.getsize(p)))
    files.sort()
    total = sum(s for _, s in files)
    print(f"artifact files: {len(files)}, total bytes: {total}")
    for rel, size in files:
        print(f"  {size:>9}  {rel}")
    if total > ARTIFACT_DUMP_BUDGET_BYTES:
        # Never silently truncate evidence: say exactly what was dropped.
        print(
            f"\n!! artifact total {total} exceeds dump budget {ARTIFACT_DUMP_BUDGET_BYTES}; "
            "the largest files are listed above and are NOT inlined below."
        )
    spent = 0
    for rel, size in files:
        if spent + size > ARTIFACT_DUMP_BUDGET_BYTES:
            print(f"\n<<<SKIPPED {rel} ({size} bytes) - dump budget exhausted>>>")
            continue
        with open(os.path.join(ARTIFACT_ROOT, rel)) as fh:
            body = fh.read()
        import hashlib

        digest = hashlib.sha256(body.encode()).hexdigest()
        print(f"\n<<<ARTIFACT BEGIN {rel} sha256={digest} bytes={len(body.encode())}>>>")
        print(body, end="" if body.endswith("\n") else "\n")
        print(f"<<<ARTIFACT END {rel}>>>")
        spent += size


def _write_eval_md(summary: dict) -> None:
    """EVAL.md is the conventional artifact the experiment tree reads back."""
    lines = ["# EVAL - arXiv 2602.08372 claim verification", ""]
    env = summary["environment"]
    lines += [
        f"- git sha: `{env['git_sha']}`",
        f"- command: `{env['run_command']}`",
        f"- python: {env['cpu']['python']}, numpy {env['cpu'].get('numpy')}",
        f"- cpu: {env['cpu'].get('cpu_model')} | usable cores {env['cpu']['sched_affinity_cpus']} "
        f"| BLAS threads {env['cpu']['blas_threads_env']}",
        f"- total runtime: {summary['total_runtime_s']}s",
        "",
        "| Claim | Verdict | Controls ok | Runtime (s) |",
        "| --- | --- | --- | --- |",
    ]
    for r in summary["claims"]:
        ctl = all(c.get("behaved_as_designed", True) for c in r["controls"]) if r["controls"] else "n/a"
        lines.append(f"| {r['claim_id']} | {r['verdict']} | {ctl} | {r['runtime_s']} |")
    lines += ["", f"Independent checker: **{'ok' if summary['independent_checker_ok'] else 'FAIL'}**", ""]
    for r in summary["claims"]:
        lines += [f"## {r['claim_id']} - {r['title']}", "", f"Verdict: **{r['verdict']}**", "", "```json",
                  json.dumps(r["headline"], indent=2)[:4000], "```", ""]
        for n in r["notes"]:
            lines.append(f"- {n}")
        lines.append("")
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    with open(os.path.join(ARTIFACT_ROOT, "EVAL.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
