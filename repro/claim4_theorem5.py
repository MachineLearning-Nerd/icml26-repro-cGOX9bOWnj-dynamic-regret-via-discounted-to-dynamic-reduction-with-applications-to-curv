"""Claim 4 - Theorem 5: clipped Adam under a RELAXED condition on beta_2.

EXACT STATEMENT UNDER TEST (Section 4.3.1, Theorem 5; proof Appendix D.6)
------------------------------------------------------------------------
Under Assumptions 1-4, run Algorithm 2 with beta = beta_1 and the discounted FTRL
update of Eq. (5) on D = {||Delta|| <= D}, with l_t(Delta) = <g_t, Delta> and
eta_t of Eq. (7) -- i.e. clipped Adam, Eq. (8). Set

    1 - (eps/(16(G+sigma)))^2 <= beta_1 < 1,
    D = (1-beta_1) sqrt(eps) / sqrt(48 c),
    gamma = beta_1 D / sqrt(1-beta_1),
    0 < nu <= G+sigma,
    max{ 1 - nu/(G+sigma), beta_1^4 } <= beta_2 < 1.                        (*)

If  T >= max{ (1/(1-beta_1)) max{16 F* sqrt(48c)/eps^{3/2}, 16(G+sigma)/eps},
              ln2/(1-beta_2) }
then  E[ ||grad F(xbar)||_c ] <= eps.

THE CLAIM'S NOVELTY, AND WHERE IT MUST THEREFORE BE TESTED
-----------------------------------------------------------
Prior analyses require beta_2 >= beta_1^2. Condition (*) with nu = G+sigma (its
largest permitted value) reduces to beta_2 >= beta_1^4, which admits the entire
band

    beta_1^4 <= beta_2 < beta_1^2

that the prior condition forbids. At eps = 1, G+sigma = 2 that band is
[0.996099, 0.998048) -- narrow, but nonempty and precisely the content of the
claim. Testing only at beta_2 = 0.999 (which satisfies both the old and the new
condition, as the rejected baseline did) exercises none of it. The sweep here is
therefore centred on beta2_choice in {lower, relaxed_mid, relaxed_top}, all of
which lie strictly inside the relaxed-only band, with {standard, high} included
as a comparison.

The O(F* G^2 c^{1/2} eps^{-7/2}) complexity is algebra on the T formula and is
checked symbolically (see adam_claims.complexity_symbolic), not by fitting a
slope -- the theorem asserts its T is sufficient, not that it is attained.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from . import adam_claims as AC
from . import o2nc
from .common import ClaimResult, Timer, banner, write_csv, write_json

CLAIM_ID = "claim4_theorem5"
VARIANT = "clipped"
PARAMS = o2nc.theorem5_params


def build_settings() -> list[dict[str, Any]]:
    settings = []
    # main grid: the relaxed-only band is the focus
    for objective in AC.OBJECTIVES:
        for d in (2, 5):
            for eps in (None,):
                for c in (0.5, 2.0):
                    for choice in ("lower", "relaxed_mid", "relaxed_top", "standard", "high"):
                        settings.append(AC.make_setting(VARIANT, objective, d, eps, c, 0.25, choice, PARAMS))
    # a harder, smaller-eps point (larger T) on two objectives
    for objective in ("two_well", "rastrigin_lipschitz"):
        for choice in ("lower", "relaxed_mid", "standard"):
            settings.append(AC.make_setting(VARIANT, objective, 3, None, 1.0, 0.25, choice, PARAMS, eps_ratio=0.45))
    return settings


def run() -> ClaimResult:
    banner("CLAIM 4 - Theorem 5: clipped Adam, relaxed beta_2 condition")
    with Timer() as timer:
        nw = AC.n_workers()
        settings = build_settings()
        print(f"settings: {len(settings)}   worker processes: {nw}")
        b1 = settings[0]["beta1"]
        print(f"relaxed-only band at eps={settings[0]['eps']}: "
              f"[beta_1^4, beta_1^2) = [{b1**4:.6f}, {b1**2:.6f})")

        # ---- assumption audit (numerical, not assumed) -----------------------
        print("\n  Assumption audit (Assumptions 1 and 4, checked numerically):")
        audits = []
        for objective in AC.OBJECTIVES:
            for d in (2, 3, 5):
                obj = o2nc.Objective(objective, d, 0.5, seed=17)
                a = o2nc.audit_assumptions(obj, n=30000)
                a.update({"objective": objective, "d": d})
                audits.append(a)
                print(f"    {objective:<22} d={d}  A1={a['A1_holds']}  "
                      f"A4 unbiased={a['A4_unbiased_within_mc_error']} "
                      f"var<=sigma^2={a['A4_variance_holds']} bounded={a['A4_bounded_holds']}")
        audit_ok = all(a["A1_holds"] and a["A4_variance_holds"] and a["A4_bounded_holds"]
                       and a["A4_unbiased_within_mc_error"] for a in audits)

        with mp.Pool(nw) as pool:
            # ---- main verification -------------------------------------------
            print("\n  Running Algorithm 2 at the theorem's own T and parameters:")
            rows = []
            for i, s in enumerate(settings):
                r = AC.evaluate_setting(s, pool)
                rows.append(r)
                print(f"    [{i+1:>3}/{len(settings)}] {r['objective']:<22} d={r['d']} eps={r['eps']} "
                      f"c={r['c']} b2={r['beta2']:.6f} ({r['beta2_band']:<14}) T={r['T']:<8} "
                      f"E<={r['E_grad_c_upper_2se']:.4f}  holds={r['conclusion_holds']}")

            relaxed = [r for r in rows if r["beta2_band"] == "relaxed_only"]
            both = [r for r in rows if r["beta2_band"] == "satisfies_both"]
            n_fail = sum(1 for r in rows if not r["conclusion_holds"])
            n_fail_relaxed = sum(1 for r in relaxed if not r["conclusion_holds"])
            print(f"\n  settings in the relaxed-only band : {len(relaxed)}  failures: {n_fail_relaxed}")
            print(f"  settings satisfying both conditions: {len(both)}  "
                  f"failures: {sum(1 for r in both if not r['conclusion_holds'])}")
            print(f"  total failures of the conclusion   : {n_fail} / {len(rows)}")

            # ---- negative controls -------------------------------------------
            print("\n  negative controls (each must make the guarantee FAIL):")
            controls = []
            base = AC.make_setting(VARIANT, "sharp_cone", 3, None, 1.0, 0.25, "relaxed_mid", PARAMS)

            # (a) T far below the theorem's threshold
            starved = {**base, "T": max(2, base["T_theoretical"] // 100000)}
            r_a = AC.evaluate_setting(starved, pool, n_runs=8)
            controls.append({
                "control": "T_far_below_threshold",
                "description": f"run only T={starved['T']} rounds instead of the theorem's "
                               f"T={base['T_theoretical']}",
                "expected": "E[||grad F||_c] exceeds eps",
                "measured_upper_2se": r_a["E_grad_c_upper_2se"], "eps": r_a["eps"],
                "behaved_as_designed": bool(not r_a["conclusion_holds"]),
            })

            # (b) Assumption 4 violated: real noise far larger than the sigma the
            #     parameter formulas were given
            # Violate the theorem's own condition 0 < nu <= G+sigma. nu enters the
            # step-size denominator, so nu = 200(G+sigma) shrinks every step by
            # ~200x and the theorem's T is no longer enough. This tests that the
            # nu condition is load-bearing rather than decorative.
            bad_nu = AC.make_setting(VARIANT, base["objective"], base["d"], base["eps"], base["c"],
                                     base["sigma"], base["beta2_choice"], PARAMS, nu_scale=2000.0)
            r_b = AC.evaluate_setting(bad_nu, pool, n_runs=8)
            controls.append({
                "control": "nu_condition_violated",
                "description": "set nu = 2000(G+sigma), violating the theorem's 0 < nu <= G+sigma, "
                               "while keeping every other parameter and T at the theorem's values",
                "expected": "E[||grad F||_c] exceeds eps",
                "measured_upper_2se": r_b["E_grad_c_upper_2se"], "eps": r_b["eps"],
                "behaved_as_designed": bool(not r_b["conclusion_holds"]),
            })

            # (c) the measurement itself must be able to report failure: demand a
            #     target 50x smaller at the same T
            strict_eps = base["eps"] / 50.0
            worst = max(r["E_grad_c_upper_2se"] for r in rows)
            controls.append({
                "control": "measurement_can_detect_failure",
                "description": f"re-test the SAME measured values against a target eps/50 = {strict_eps:.4f}",
                "expected": "at least one setting now fails, proving the acceptance test is not vacuous",
                "measured_max_upper_2se": worst, "eps": strict_eps,
                "behaved_as_designed": bool(worst > strict_eps),
            })
            for cc in controls:
                print(f"    {cc['control']:<32} measured={cc.get('measured_upper_2se', cc.get('measured_max_upper_2se')):.4f} "
                      f"vs eps={cc['eps']:.4f}  as_designed={cc['behaved_as_designed']}")

            # ---- exploratory: beta_2 BELOW the theorem's condition -------------
            # Outside the theorem's scope, so not a pass/fail control -- reported
            # because "is the relaxed condition necessary?" is the obvious next
            # question and the honest answer is 'the theorem does not say'.
            below = AC.make_setting(VARIANT, "two_well", 3, 2.0, 1.0, 0.5, "violating", PARAMS)
            r_below = AC.evaluate_setting(below, pool, n_runs=8)
            print(f"\n  exploratory (outside the theorem): beta_2={below['beta2']:.6f} < beta_1^4="
                  f"{below['beta1']**4:.6f} -> E<={r_below['E_grad_c_upper_2se']:.4f}, "
                  f"conclusion still holds={r_below['conclusion_holds']}")

            # ---- conservatism diagnostic (NOT the verification) ---------------
            print("\n  conservatism diagnostic (how far below the theorem's T does it still work?):")
            diag = AC.conservatism_diagnostic(base, pool)
            for row in diag["rows"]:
                print(f"    T_scale={row['T_scale']:<8} T={row['T']:<9} E<={row['upper_2se']:.4f} holds={row['holds']}")
            print(f"    approx conservatism factor: {diag['approx_conservatism_factor']}")

        # ---- complexity, symbolically -------------------------------------
        print("\n  iteration complexity (symbolic, from the T formula):")
        comp = AC.complexity_symbolic("theorem5")
        print(f"    1/(1-beta_1) = {comp['inv_one_minus_beta1']}")
        print(f"    branch1 / (F* (G+sigma)^2 c^(1/2) eps^(-7/2)) = {comp['branch1_over_target']} "
              f"(constant: {comp['branch1_is_constant_multiple']})")
        print(f"    branch2 / ((G+sigma)^3 eps^-3) = {comp['branch2_over_target']} "
              f"(constant: {comp['branch2_is_constant_multiple']})")
        comp_ok = (comp["branch1_is_constant_multiple"] and comp["branch2_is_constant_multiple"]
                   and comp["branch3_is_constant_multiple"])

    # ---- artifacts
    cols = [c for c in rows[0] if c != "per_run_means"]
    write_csv(CLAIM_ID, "settings_results.csv", [{k: r[k] for k in cols} for r in rows], fieldnames=cols)
    write_json(CLAIM_ID, "per_run_values.json", [{"objective": r["objective"], "d": r["d"],
               "eps": r["eps"], "c": r["c"], "beta2": r["beta2"], "beta2_band": r["beta2_band"],
               "per_run_means": r["per_run_means"]} for r in rows])
    write_json(CLAIM_ID, "assumption_audit.json", audits)
    write_json(CLAIM_ID, "negative_controls.json", controls)
    write_json(CLAIM_ID, "complexity_symbolic.json", comp)
    write_json(CLAIM_ID, "conservatism_diagnostic.json", diag)
    write_json(CLAIM_ID, "exploratory_beta2_below_condition.json", r_below)

    controls_ok = all(c["behaved_as_designed"] for c in controls)
    verdict = "VERIFIED" if (n_fail == 0 and audit_ok and comp_ok) else ("FALSIFIED" if n_fail > 0 else "BLOCKED")

    write_json(CLAIM_ID, "summary.json", {
        "n_settings": len(rows), "n_failures": n_fail,
        "n_relaxed_band_settings": len(relaxed), "n_relaxed_band_failures": n_fail_relaxed,
        "relaxed_band_at_eps2": {"beta1_4": settings[0]["beta1"] ** 4, "beta1_2": settings[0]["beta1"] ** 2},
        "assumption_audit_ok": audit_ok,
        "complexity_symbolic_ok": comp_ok,
        "worst_ratio_estimate_over_eps": max(r["ratio_estimate_over_eps"] for r in rows),
        "controls_ok": controls_ok,
        "conservatism_factor": diag["approx_conservatism_factor"],
        "verdict": verdict,
    })

    notes = [
        f"{len(relaxed)} of {len(rows)} settings sit strictly inside the relaxed-only band "
        f"[beta_1^4, beta_1^2) that prior beta_2 >= beta_1^2 conditions forbid; this is the content of the claim.",
        "Algorithm 2's Exp(1) step scaling and EMA output are implemented as written; the update is "
        "Eq. (8) exactly, not a plain-Adam stand-in.",
        "||grad F(x)||_c is replaced by a computable UPPER bound via Gaussian smoothing, and acceptance "
        "uses the upper end of a 2-standard-error interval, so every approximation is conservative.",
        "The theorem's T is a worst-case sufficient condition. The measured E[||grad F||_c] sits well below "
        f"eps (worst ratio {max(r['ratio_estimate_over_eps'] for r in rows):.3f}); the conservatism diagnostic "
        f"quantifies this (~{diag['approx_conservatism_factor']}x) and is reported separately from the verdict.",
        "Scope: this is scoped corroboration over four objective families, not a proof. The theorem quantifies "
        "over all F satisfying Assumptions 1-4, which no finite experiment can exhaust.",
    ]
    return ClaimResult(
        claim_id=CLAIM_ID,
        title="Theorem 5 - clipped Adam, relaxed beta_2 condition (Section 4.3.1)",
        verdict=verdict,
        ok=(verdict in ("VERIFIED", "FALSIFIED")) and controls_ok,
        headline={
            "n_settings": len(rows),
            "n_failures_of_conclusion": n_fail,
            "n_settings_in_relaxed_only_band": len(relaxed),
            "n_failures_in_relaxed_only_band": n_fail_relaxed,
            "relaxed_only_band_example": {"eps": settings[0]["eps"],
                                          "beta1_pow4": settings[0]["beta1"] ** 4,
                                          "beta1_pow2": settings[0]["beta1"] ** 2},
            "worst_ratio_E_over_eps": max(r["ratio_estimate_over_eps"] for r in rows),
            "assumption_audit_ok": audit_ok,
            "complexity_symbolic_ok": comp_ok,
            "negative_controls_all_failed_as_intended": controls_ok,
            "conservatism_factor_diagnostic": diag["approx_conservatism_factor"],
        },
        controls=controls,
        notes=notes,
        runtime_s=timer.elapsed,
    )
