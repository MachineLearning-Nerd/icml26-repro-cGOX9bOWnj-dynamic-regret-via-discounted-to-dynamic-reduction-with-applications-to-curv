"""Independent checker.

Re-derives every headline number from the RAW ARTIFACT FILES alone. It does not
import the verifiers or reuse their helper functions; where it needs to redo a
computation it reimplements it from the paper's formulas by a different route.
Its job is to catch a verifier that reports a number its own raw data does not
support -- transcription slips, stale results, silently-empty sweeps.

A checker failure fails the whole run.
"""

from __future__ import annotations

import csv
import json
import os
import random
from fractions import Fraction
from typing import Any, Callable

from .common import ARTIFACT_ROOT


def _read_csv(claim_id: str, name: str) -> list[dict[str, str]]:
    path = os.path.join(ARTIFACT_ROOT, claim_id, name)
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _read_json(claim_id: str, name: str) -> Any:
    path = os.path.join(ARTIFACT_ROOT, claim_id, name)
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# claim 1
# --------------------------------------------------------------------------
def check_claim1(lines: list[str]) -> bool:
    ok = True
    claim = "claim1_theorem1"

    rows_a = _read_csv(claim, "route_a_symbolic_identity.csv")
    ts = sorted(int(r["T"]) for r in rows_a)
    a_all = all(r["residual_is_identically_zero"] == "True" for r in rows_a)
    contiguous = ts == list(range(1, max(ts) + 1)) and max(ts) >= 20
    lines.append(f"claim1 route A : {len(rows_a)} rows, T={min(ts)}..{max(ts)}, all-zero={a_all}, contiguous={contiguous}")
    ok &= a_all and contiguous

    rows_b = _read_csv(claim, "route_b_coefficient_proof.csv")
    b_all = all(r["difference_simplifies_to_zero"] == "True" for r in rows_b)
    classes = {r["symbol_class"] for r in rows_b}
    # All five symbol classes must be covered, or the general-T proof has a hole.
    need = {"a_s = f_s(x_s)", "Lambda_s", "phi_t(u_{t+1})", "phi_t(u_t) interior", "f_T(u_T)"}
    covered = need.issubset(classes)
    lines.append(f"claim1 route B : {len(rows_b)} coefficient cases, all-match={b_all}, classes-covered={covered}")
    ok &= b_all and covered

    # Independent re-derivation of the certificate identity: substitute random
    # exact rationals into BOTH sides, implemented here from the paper formulas
    # directly rather than via the verifier's sympy expressions.
    rng = random.Random(31337)
    indep_ok = True
    worst = None
    for _ in range(300):
        T = rng.randint(1, 8)
        b = Fraction(rng.randint(1, 30), 30)
        a = {s: Fraction(rng.randint(-30, 30), 7) for s in range(1, T + 1)}
        c = {(s, k): Fraction(rng.randint(-30, 30), 7) for s in range(1, T + 1) for k in range(1, T + 1)}
        p = {(t, k): Fraction(rng.randint(-30, 30), 7) for t in range(1, T + 1) for k in range(1, T + 1)}
        L = {s: Fraction(rng.randint(-30, 30), 7) for s in range(1, T + 1)}

        def reg(t: int, k: int) -> Fraction:
            return sum(b ** (t - s) * (a[s] - c[(s, k)]) for s in range(1, t + 1))

        def F(t: int, k: int) -> Fraction:
            return b**t * p[(t, k)] + sum(b ** (t - s) * c[(s, k)] for s in range(1, t + 1))

        def S(t: int, k: int) -> Fraction:
            return b**t * p[(t, k)] + b**t * sum(L[s] for s in range(1, t + 1)) - reg(t, k)

        lhs = sum(a[t] - c[(t, t)] for t in range(1, T + 1))
        rhs = (
            b * p[(1, 1)]
            + sum(b**t * L[t] for t in range(1, T + 1))
            + b * sum(F(t, t + 1) - F(t, t) for t in range(1, T))
            + b * sum(b**t * (p[(t + 1, t + 1)] - p[(t, t + 1)]) for t in range(1, T))
        )
        cert = (1 - b) * sum(S(t, t) for t in range(1, T + 1)) + b * S(T, T)
        resid = rhs - lhs - cert
        if resid != 0:
            indep_ok = False
            worst = (T, str(resid))
            break
    lines.append(
        f"claim1 indep   : certificate identity re-derived independently on 300 random exact-rational "
        f"instances (free symbols, no sign assumptions): holds={indep_ok}"
        + (f" FIRST FAILURE T={worst[0]} residual={worst[1]}" if worst else "")
    )
    ok &= indep_ok

    # Recompute route-C margins from the stored exact lhs/rhs strings.
    rows_c = _read_csv(claim, "route_c_rational_instances.csv")
    recomputed_ok = True
    n_neg = 0
    for r in rows_c:
        lhs = Fraction(r["lhs_dynamic_regret"])
        rhs = Fraction(r["rhs_theorem1_bound"])
        stated = Fraction(r["margin_rhs_minus_lhs"])
        if rhs - lhs != stated:
            recomputed_ok = False
        if rhs - lhs < 0:
            n_neg += 1
    summary = _read_json(claim, "summary.json")
    lines.append(
        f"claim1 route C : {len(rows_c)} stored rows re-checked, margins consistent={recomputed_ok}, "
        f"negative margins={n_neg}; verifier reported {summary['route_c']['violations']} violations "
        f"over {summary['route_c']['n_instances']} instances"
    )
    ok &= recomputed_ok and n_neg == 0 and summary["route_c"]["violations"] == 0

    ctl = _read_json(claim, "negative_controls.json")
    ctl_ok = all(c["behaved_as_designed"] for c in ctl)
    lines.append(f"claim1 controls: {len(ctl)} controls, all broke as designed={ctl_ok}")
    ok &= ctl_ok
    return ok


CHECKS: dict[str, Callable[[list[str]], bool]] = {
    "claim1_theorem1": check_claim1,
}


def run(results: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    lines: list[str] = []
    all_ok = True
    checked = []
    for r in results:
        if not r.get("implemented"):
            continue
        fn = CHECKS.get(r["claim_id"])
        if fn is None:
            lines.append(f"{r['claim_id']}: NO INDEPENDENT CHECK REGISTERED -- treated as failure")
            all_ok = False
            continue
        try:
            ok = fn(lines)
        except Exception as exc:  # a checker that crashes is a failed check
            lines.append(f"{r['claim_id']}: checker raised {type(exc).__name__}: {exc}")
            ok = False
        checked.append({"claim_id": r["claim_id"], "ok": ok})
        all_ok &= ok
    return all_ok, {"ok": all_ok, "lines": lines, "checked": checked}
