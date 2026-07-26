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


def _check_adam_claim(lines: list[str], claim: str, label: str) -> bool:
    """Shared independent check for Claims 4 and 5 (Theorems 5 and 7)."""
    import numpy as np

    ok = True
    rows = _read_csv(claim, "settings_results.csv")
    summary = _read_json(claim, "summary.json")

    # 1. recompute the acceptance test from the stored raw per-run values
    per_run = _read_json(claim, "per_run_values.json")
    mismatches = 0
    for r, pr in zip(rows, per_run):
        means = np.array(pr["per_run_means"])
        est = means.mean()
        se = means.std(ddof=1) / np.sqrt(len(means))
        if not np.isclose(est, float(r["E_grad_c_upper_estimate"]), rtol=1e-9):
            mismatches += 1
        elif not np.isclose(est + 2 * se, float(r["E_grad_c_upper_2se"]), rtol=1e-9):
            mismatches += 1
    lines.append(f"{label} stats  : {len(rows)} settings, mean+2SE recomputed from raw per-run values, "
                 f"mismatches={mismatches}")
    ok &= mismatches == 0

    # 2. NON-VACUITY: every setting must start strictly outside the target set,
    #    or "the conclusion held" says nothing about the algorithm
    vac = [r for r in rows if r["non_vacuous"] != "True"]
    ratios = [float(r["initial_grad_c_upper"]) / float(r["eps"]) for r in rows]
    lines.append(f"{label} vacuity: {len(vac)} vacuous settings (start already inside the target); "
                 f"min ||grad F(x0)||_c / eps = {min(ratios):.3f} (must exceed 1)")
    ok &= len(vac) == 0 and min(ratios) > 1.0

    # 3. the conclusion, recomputed rather than read off
    fails = sum(1 for r in rows if float(r["E_grad_c_upper_2se"]) > float(r["eps"]))
    lines.append(f"{label} concl  : recomputed failures={fails}, verifier reported {summary['n_failures']}")
    ok &= fails == summary["n_failures"]

    # 4. assumptions genuinely audited
    aud = _read_json(claim, "assumption_audit.json")
    aud_ok = all(a["A1_holds"] and a["A4_variance_holds"] and a["A4_bounded_holds"]
                 and a["A4_unbiased_within_mc_error"] for a in aud)
    lines.append(f"{label} assump : {len(aud)} objective/dimension pairs audited, all satisfy A1 and A4={aud_ok}")
    ok &= aud_ok

    # 5. controls
    ctl = _read_json(claim, "negative_controls.json")
    ctl_ok = all(c["behaved_as_designed"] for c in ctl)
    lines.append(f"{label} ctrls  : {len(ctl)} controls "
                 f"({', '.join(c['control'] for c in ctl)}), all failed as intended={ctl_ok}")
    ok &= ctl_ok

    comp = _read_json(claim, "complexity_symbolic.json")
    comp_ok = comp["branch1_is_constant_multiple"] and comp["branch2_is_constant_multiple"]
    lines.append(f"{label} complx : T formula is a constant multiple of the claimed order={comp_ok} "
                 f"(branch1 ratio {comp['branch1_over_target']})")
    ok &= comp_ok
    return ok


def check_claim4(lines: list[str]) -> bool:
    ok = _check_adam_claim(lines, "claim4_theorem5", "claim4")
    rows = _read_csv("claim4_theorem5", "settings_results.csv")
    # the claim's novelty: settings must actually sit in the relaxed-only band
    relaxed = [r for r in rows if r["beta2_band"] == "relaxed_only"]
    fails = sum(1 for r in relaxed if float(r["E_grad_c_upper_2se"]) > float(r["eps"]))
    # and that band must be genuinely below beta_1^2
    ok_band = all(float(r["beta2"]) < float(r["beta1"]) ** 2 for r in relaxed)
    lines.append(
        f"claim4 band   : {len(relaxed)}/{len(rows)} settings inside [beta_1^4, beta_1^2) -- the band prior "
        f"beta_2 >= beta_1^2 conditions forbid -- with {fails} failures; all strictly below beta_1^2={ok_band}"
    )
    return ok and len(relaxed) > 0 and fails == 0 and ok_band


def check_claim5(lines: list[str]) -> bool:
    ok = _check_adam_claim(lines, "claim5_theorem7", "claim5")
    rows = _read_csv("claim5_theorem7", "settings_results.csv")
    # Theorem 7 keeps the standard condition: every setting must satisfy it
    ok_cond = all(float(r["beta2"]) >= float(r["beta2_lower_bound"]) - 1e-12 for r in rows)
    mus = {float(r["mu"]) for r in rows}
    lines.append(f"claim5 cond   : all {len(rows)} settings satisfy beta_2 >= max(1-nu/(G+sigma), beta_1^2)"
                 f"={ok_cond}; damping mu values used: {len(mus)} distinct, all > 0="
                 f"{all(m > 0 for m in mus)}")
    return ok and ok_cond and all(m > 0 for m in mus)


CHECKS: dict[str, Callable[[list[str]], bool]] = {
    "claim1_theorem1": check_claim1,
    "claim4_theorem5": check_claim4,
    "claim5_theorem7": check_claim5,
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
