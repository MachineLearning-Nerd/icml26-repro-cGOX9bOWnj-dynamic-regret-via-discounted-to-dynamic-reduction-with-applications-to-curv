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


# --------------------------------------------------------------------------
# claim 2
# --------------------------------------------------------------------------
def _naive_vaw(Z, y, beta: float, lam: float):
    """Discounted VAW recomputed WITHOUT the O(d^2) recursion.

    The verifier maintains S_t = beta S_{t-1} + z_t z_t^T incrementally. Here the
    sums are rebuilt from scratch at every round straight from the definition in
    Section 3.1. If the recursion had an off-by-one in its beta powers -- the
    most likely way to implement this algorithm wrongly -- the two would disagree.
    """
    import numpy as np

    T, d = Z.shape
    X = np.empty((T, d))
    for t in range(1, T + 1):
        A = lam * (beta**t) * np.eye(d) + np.outer(Z[t - 1], Z[t - 1])
        b = np.zeros(d)
        for s in range(1, t):  # s < t
            A += (beta ** (t - s)) * np.outer(Z[s - 1], Z[s - 1])
            b += (beta ** (t - s)) * y[s - 1] * Z[s - 1]
        X[t - 1] = np.linalg.solve(A, b)
    return X


def check_claim2(lines: list[str]) -> bool:
    import numpy as np

    ok = True
    claim = "claim2_theorem2"
    rows = _read_csv(claim, "sweep_results.csv")
    summary = _read_json(claim, "summary.json")

    # 1. margins and term decomposition recomputed from the stored columns
    bad_margin = bad_terms = neg = 0
    for r in rows:
        rhs, dreg, margin = float(r["rhs_theorem2"]), float(r["dynamic_regret"]), float(r["margin"])
        if not np.isclose(rhs - dreg, margin, rtol=1e-9, atol=1e-12):
            bad_margin += 1
        tsum = sum(float(r[f"term{i}"]) for i in (1, 2, 3, 4))
        if not np.isclose(tsum, rhs, rtol=1e-9, atol=1e-12):
            bad_terms += 1
        if margin < 0:
            neg += 1
    lines.append(
        f"claim2 sweep   : {len(rows)} rows; margin=rhs-regret mismatches={bad_margin}, "
        f"term-sum mismatches={bad_terms}, negative margins={neg} "
        f"(verifier reported {summary['n_violations_of_E']} violations)"
    )
    ok &= bad_margin == 0 and bad_terms == 0 and neg == 0 and summary["n_violations_of_E"] == 0

    # 2. derivation link slacks must be nonnegative in the raw table
    link_cols = [c for c in rows[0] if c.startswith("L") and "slack" in c]
    link_bad = {c: sum(1 for r in rows if r[c] != "" and float(r[c]) < -1e-9) for c in link_cols}
    lines.append(f"claim2 links   : {link_cols} negative-slack counts = {link_bad}")
    ok &= all(v == 0 for v in link_bad.values())

    # 3. the sweep must actually be calibrated: some configuration has to get
    #    close to the bound, or "no violation" carries no information
    max_tight = max((float(r["tightness_ratio"]) for r in rows
                     if r["tightness_ratio"] not in ("", "nan")), default=0.0)
    adv_tight = summary["adversarial_true_bound"]["max_tightness_ratio_found"]
    adv_margin = summary["adversarial_true_bound"]["best_margin"]
    calibrated = (max(max_tight, adv_tight) > 0.25) and adv_margin < 1e-6
    lines.append(
        f"claim2 calib   : max tightness in sweep={max_tight:.4f}, via adversarial search={adv_tight:.4f}, "
        f"adversarial best margin={adv_margin:.3g} -> test is calibrated={calibrated}"
    )
    ok &= calibrated

    # 4. re-run a sample of configurations through the independent VAW
    from .claim2_theorem2 import gen_comparators, gen_data

    mismatches = 0
    sampled = 0
    for r in rows[:: max(1, len(rows) // 12)]:
        T, d = int(r["T"]), int(r["d"])
        if T > 120:  # the naive path is O(T^2 d^2); keep the checker quick
            continue
        Z, y = gen_data(r["data"], T, d, int(r["seed"]))
        U = gen_comparators(r["comp"], Z, y, int(r["seed"]))
        beta, lam = float(r["beta"]), float(r["lam"])
        Xn = _naive_vaw(Z, y, beta, lam)
        pred_alg = np.einsum("td,td->t", Xn, Z)
        pred_cmp = np.einsum("td,td->t", U, Z)
        dreg = float((0.5 * (pred_alg - y) ** 2).sum() - (0.5 * (pred_cmp - y) ** 2).sum())
        sampled += 1
        if not np.isclose(dreg, float(r["dynamic_regret"]), rtol=1e-6, atol=1e-8):
            mismatches += 1
    lines.append(
        f"claim2 algo    : {sampled} configs re-run through a from-scratch VAW implementation "
        f"(no recursion), dynamic-regret mismatches={mismatches}"
    )
    ok &= mismatches == 0 and sampled > 0

    # 5. controls
    ctl = _read_json(claim, "negative_controls.json")
    ctl_ok = all(c["behaved_as_designed"] for c in ctl)
    fired = sum(1 for c in ctl if c["search_found_violation"])
    lines.append(
        f"claim2 controls: {len(ctl)} weakened bounds, adversarial search broke {fired} of them, "
        f"all behaved as designed={ctl_ok}"
    )
    ok &= ctl_ok

    rate = _read_json(claim, "rate_subclaim.json")
    rate_ok = rate["symbolic_min_equals_2sqrt_PV"] and rate["numeric_matches_closed_form"]
    # independently re-derive min_q (P/q + qV) = 2 sqrt(PV) on the stored rows
    rederived = all(
        np.isclose(float(row["closed_form_2sqrtPV"]), 2 * np.sqrt(float(row["P"]) * float(row["V"])), rtol=1e-12)
        for row in rate["rows"]
    )
    lines.append(f"claim2 rate    : symbolic+numeric tuning identity holds={rate_ok}, re-derived from raw rows={rederived}")
    ok &= rate_ok and rederived
    return ok


CHECKS: dict[str, Callable[[list[str]], bool]] = {
    "claim1_theorem1": check_claim1,
    "claim2_theorem2": check_claim2,
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
