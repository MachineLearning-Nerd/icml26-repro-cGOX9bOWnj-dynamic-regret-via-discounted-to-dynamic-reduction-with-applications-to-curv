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
# claim 3
# --------------------------------------------------------------------------
def check_claim3(lines: list[str]) -> bool:
    """Re-derive Theorem 3's bound from the paper formula, not from repro.aioli.

    Terms 2, 3 and 4 of (E3) are functions of quantities the sweep already
    stored per row (T, d, beta, B, R, lambda, P_T^beta), so they can be
    recomputed here from the printed statement of the theorem without touching
    the verifier's code. Term 1 needs ||u_1||, which is not a stored column, so
    it is recovered as the residual and only sanity-checked for sign and scale.
    """
    import math

    ok = True
    claim = "claim3_theorem34"
    rows = _read_csv(claim, "sweep_results.csv")
    summary = _read_json(claim, "summary.json")

    if not rows:
        lines.append("claim3 sweep   : EMPTY sweep_results.csv")
        return False

    # --- 1. margin, holds and the violation count, recomputed from raw columns
    n_viol = 0
    n_margin_mismatch = 0
    for r in rows:
        dreg, rhs = float(r["dynamic_regret"]), float(r["rhs_theorem3"])
        margin = rhs - dreg
        if abs(margin - float(r["margin"])) > 1e-6 * max(1.0, abs(margin)):
            n_margin_mismatch += 1
        if (margin >= 0) != (r["holds"] == "True"):
            n_margin_mismatch += 1
        if margin < 0:
            n_viol += 1
    lines.append(f"claim3 margins : {len(rows)} rows recomputed from raw columns, "
                 f"mismatches={n_margin_mismatch}, violations={n_viol} "
                 f"(summary says {summary['n_violations_of_E3']})")
    ok &= n_margin_mismatch == 0 and n_viol == int(summary["n_violations_of_E3"])

    # --- 2. terms 2-4 of (E3) re-derived from the theorem statement itself
    worst_rel = 0.0
    checked = 0
    for r in rows:
        T, d = int(r["T"]), int(r["d"])
        beta, B, R, lam = float(r["beta"]), float(r["B"]), float(r["R"]), float(r["lam"])
        P = float(r["P_T_beta"])
        # sum_{t=1..T} beta^{T-t} = (1 - beta^T) / (1 - beta)
        geo = (1.0 - beta**T) / (1.0 - beta)
        t2 = d * (1 + B * R) * math.log(1 + R * R * geo / (d * lam * (1 + B * R)))
        t3 = (beta / (1 - beta)) * P
        t4 = ((1 - beta) / beta) * d * (1 + B * R) * T
        for name, mine, theirs in (("term2", t2, float(r["term2"])),
                                   ("term3", t3, float(r["term3"])),
                                   ("term4", t4, float(r["term4"]))):
            rel = abs(mine - theirs) / max(1.0, abs(theirs))
            worst_rel = max(worst_rel, rel)
        checked += 1
    lines.append(f"claim3 formula : terms 2-4 of (E3) re-derived from the paper statement on "
                 f"{checked} rows, worst relative deviation {worst_rel:.3e}")
    ok &= worst_rel < 1e-9

    # --- 3. term1 must be beta*lambda*||u_1||^2 >= 0 and the four terms must sum to the RHS
    n_sum_bad = sum(
        1 for r in rows
        if abs(sum(float(r[k]) for k in ("term1", "term2", "term3", "term4")) - float(r["rhs_theorem3"]))
        > 1e-6 * max(1.0, abs(float(r["rhs_theorem3"])))
    )
    n_t1_bad = sum(1 for r in rows if float(r["term1"]) < -1e-12)
    lines.append(f"claim3 terms   : rows where term1+..+term4 != rhs: {n_sum_bad}; "
                 f"rows with negative term1: {n_t1_bad}")
    ok &= n_sum_bad == 0 and n_t1_bad == 0

    # --- 4. assumptions of Theorems 3/4 actually enforced in the data
    bad_u = sum(1 for r in rows if float(r["max_comparator_norm"]) > float(r["B"]) * (1 + 1e-9))
    bad_z = sum(1 for r in rows if float(r["max_feature_norm"]) > float(r["R"]) * (1 + 1e-9))
    lines.append(f"claim3 assumpt : rows violating ||u_t||<=B: {bad_u}; rows violating ||z_t||<=R: {bad_z}")
    ok &= bad_u == 0 and bad_z == 0

    # --- 5. negative controls must each have fired
    controls = _read_json(claim, "negative_controls.json")
    dead = [c["control"] for c in controls if not c["behaved_as_designed"]]
    lines.append(f"claim3 controls: {len(controls)} weakened bounds, none-fired={dead or 'none'}")
    ok &= not dead and len(controls) >= 5

    # --- 6. B-dependence gates recomputed from the raw per-B table
    bd = _read_csv(claim, "b_dependence.csv")
    bsum = _read_json(claim, "b_dependence_summary.json")
    a_max = max(float(r["regret_aioli_mean"]) for r in bd)
    o_max = max(float(r["regret_ons_mean"]) for r in bd)
    within = all(float(r["regret_aioli_mean"]) <= float(r["bound_theorem3_mean"]) for r in bd)
    n_seeds_bad = sum(1 for r in bd if int(r["n_seeds"]) == 0)
    lines.append(f"claim3 B-dep   : {len(bd)} B values, no empty strata={n_seeds_bad == 0}, "
                 f"AIOLI max {a_max:.3f} vs ONS max {o_max:.3f}, inside (E3) at every B={within}")
    ok &= (n_seeds_bad == 0
           and within == bool(bsum["aioli_regret_within_theorem3_bound_at_every_B"])
           and abs(a_max - float(bsum["levels"]["aioli_max_regret_over_B_range"])) < 1e-6
           and abs(o_max - float(bsum["levels"]["ons_max_regret_over_B_range"])) < 1e-6)

    # A probe whose positive control does not move cannot detect B-dependence
    # at all, so the separation is only meaningful if ONS actually grew.
    ons_grew = o_max > 10.0 * max(a_max, 1.0)
    lines.append(f"claim3 B-ctrl  : ONS positive control grew to >10x AIOLI={ons_grew} "
                 f"(if False the probe has no power and no B conclusion is supportable)")
    ok &= ons_grew == bool(bsum["ons_regret_far_exceeds_aioli"])

    # --- 7. Theorem 4 ensemble ratios recomputed from the raw table
    ens = _read_csv(claim, "ensemble_theorem4.csv")
    esum = _read_json(claim, "ensemble_summary.json")
    ratios = [float(r["regret_ensemble"]) / float(r["theorem4_scale_dBlogBT_plus_sqrt_dBTP"])
              for r in ens if float(r["theorem4_scale_dBlogBT_plus_sqrt_dBTP"]) > 0]
    max_ratio = max(ratios) if ratios else float("nan")
    lines.append(f"claim3 thm4    : {len(ens)} ensemble configs, max regret/scale recomputed "
                 f"{max_ratio:.4f} (summary says {float(esum['max_ratio_regret_over_theorem4_scale']):.4f})")
    ok &= len(ens) > 0 and abs(max_ratio - float(esum["max_ratio_regret_over_theorem4_scale"])) < 1e-6

    # --- 8. the implicit update must have been solved, not approximated
    unconv = sum(1 for r in rows if r.get("solver_converged") != "True")
    lines.append(f"claim3 solver  : rows with an uncertified implicit solve: {unconv}; "
                 f"worst relative Newton gradient norm {float(summary['worst_relative_newton_gradnorm']):.3e}")
    ok &= unconv == 0 and float(summary["worst_relative_newton_gradnorm"]) < 1e-6

    return ok


CHECKS: dict[str, Callable[[list[str]], bool]] = {
    "claim1_theorem1": check_claim1,
    "claim3_theorem34": check_claim3,
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
