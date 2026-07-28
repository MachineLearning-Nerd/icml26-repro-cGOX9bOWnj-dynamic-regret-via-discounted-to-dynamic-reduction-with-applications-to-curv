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

    # --- 5. negative controls: each must either have fired, or carry a
    #        QUANTIFIED power deficit. An un-fired control with no number
    #        attached is an untested claim dressed as a passing one.
    controls = _read_json(claim, "negative_controls.json")
    fired = [c["control"] for c in controls if c["behaved_as_designed"]]
    dead = [c for c in controls if not c["behaved_as_designed"]]
    undocumented = [c["control"] for c in dead
                    if not (0.0 <= float(c.get("power_ratio_removed_over_slack", -1)) < 1.0)
                    or "NO POWER" not in str(c.get("power_note", ""))]
    lines.append(f"claim3 controls: {len(controls)} weakened bounds, fired={fired or 'none'}, "
                 f"no-power={[c['control'] for c in dead] or 'none'}, "
                 f"undocumented-no-power={undocumented or 'none'}")
    # A control that did not fire must report a power ratio strictly below 1:
    # a ratio >= 1 would mean it COULD have fired and the verifier missed it.
    ok &= not undocumented and len(fired) > 0 and len(controls) >= 5

    # --- 5b. calibration: the search must have had power, and must not have
    #         quietly used a smaller budget against the true bound than against
    #         the weakened ones (which would manufacture a "survived" result)
    adv = _read_json(claim, "adversarial_search.json")
    tgts = adv["per_target"]
    broke = [p for p in tgts if p != "true" and tgts[p]["violated"]]
    survived = [p for p in tgts if p != "true" and not tgts[p]["violated"]]
    budgets = {p: tgts[p]["n_local_moves"] for p in tgts}
    equal_budget = max(budgets.values()) - min(budgets.values()) <= 0.25 * max(budgets.values())
    lines.append(f"claim3 calib   : {adv['n_evaluations']} evaluations; weakened bounds broken "
                 f"{len(broke)}/{len(tgts) - 1} (survived: {survived or 'none'}); "
                 f"true bound best margin {tgts['true']['best_margin']:.6g}; "
                 f"equal search budget across targets={equal_budget}")
    ok &= equal_budget
    ok &= (len(broke) == len(tgts) - 1) == bool(
        adv["n_weakened_bounds_broken"] == adv["n_weakened_bounds"])
    ok &= (tgts["true"]["best_margin"] >= 0) == bool(adv["true_bound_survived"])

    # A "survived" result is only meaningful if the search actually approached
    # the bound. A max tightness ratio near zero would mean it never got close.
    lines.append(f"claim3 tight   : max tightness ratio found by search "
                 f"{float(adv['max_tightness_ratio_found']):.4f} "
                 f"(near 0 would mean the search never approached the bound)")

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
    # L3 is Lemma 25, which the campaign proved false by certified counterexample,
    # so negative L3 slack is an expected documented finding. Every other link
    # must hold on every configuration.
    non_l3_bad = {k: v for k, v in link_bad.items() if not k.startswith("L3")}
    lines.append(f"claim2 links   : negative-slack counts = {link_bad}; "
                 f"non-L3 links all hold = {all(v == 0 for v in non_l3_bad.values())} "
                 f"(L3 = Lemma 25, independently certified false)")
    ok &= all(v == 0 for v in non_l3_bad.values())

    # Independently re-certify the Lemma 25 counterexample from the stored raw
    # instance, recomputing both sides here rather than trusting the verifier.
    l25 = _read_json(claim, "lemma25_counterexample.json")
    ce = l25["certified_counterexample"]
    from mpmath import log as mlog
    from mpmath import mp, mpf

    mp.dps = 80
    zz, cc, bb, ll = ce["z"], ce["c"], mpf(ce["beta"]), mpf(ce["lambda"])
    A, lhs = ll, mpf(0)
    for t in range(ce["T"]):
        A = mpf(zz[t]) ** 2 + bb * A
        lhs += mpf(cc[t]) ** 2 * mpf(zz[t]) ** 2 / A
    S = sum(bb ** (ce["T"] - 1 - t) * mpf(zz[t]) ** 2 for t in range(ce["T"]))
    rhs = mlog(1 / bb) * sum(mpf(x) ** 2 for x in cc) + max(mpf(x) ** 2 for x in cc) * mlog(1 + S / ll)
    recert = bool(rhs - lhs < 0)
    agrees = np.isclose(float(lhs), ce["lhs_float"], rtol=1e-9) and np.isclose(float(rhs), ce["rhs_float"], rtol=1e-9)
    lines.append(
        f"claim2 lemma25 : counterexample re-certified independently at 80 dps: violates={recert}, "
        f"matches stored values={agrees} (LHS {float(lhs):.8f} > RHS {float(rhs):.8f})"
    )
    ok &= recert and agrees

    route4 = _read_json(claim, "route4_focused_falsification.json")
    lines.append(
        f"claim2 route4  : {route4['n_restarts']} seeded restarts in the Lemma-25-failure region, "
        f"best margin {route4['best_margin']:.4g}, counterexample to Theorem 2 found="
        f"{route4['found_violation_of_E']}"
    )

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
    "claim2_theorem2": check_claim2,
    "claim3_theorem34": check_claim3,
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
        # A checker that falls off the end returns None. Treat anything that is
        # not a bool as a failed check rather than letting it crash the suite
        # (or, worse, be read as truthy) after hours of compute.
        if not isinstance(ok, bool):
            lines.append(f"{r['claim_id']}: checker returned {type(ok).__name__}, not bool "
                         f"-- treated as failure")
            ok = False
        checked.append({"claim_id": r["claim_id"], "ok": ok})
        all_ok &= ok
    return all_ok, {"ok": all_ok, "lines": lines, "checked": checked}
