"""Claim 5 - Theorem 7: clip-free Adam via a composite loss with damping.

EXACT STATEMENT UNDER TEST (Section 4.4, Theorem 7; proof Appendix D.8)
----------------------------------------------------------------------
Under Assumptions 1-4, run Algorithm 2 with beta = beta_1 and the discounted FTRL
update of Eq. (5) on D = R^d, with the COMPOSITE loss
l_t(Delta) = <g_t, Delta> + (mu/2)||Delta||^2 and eta_t of Eq. (7). This gives the
clip-free Adam update, Eq. (10):

    Delta_{t+1} = - gamma (1-beta_1) sum_{s<=t} beta_1^{t-s} g_s
                  / ( nu + gamma mu (1 - beta_1^t)
                      + sqrt((1-beta_2) sum_{s<=t} beta_2^{t-s} ||g_s||^2) ).

Set 1 - (eps/(16(G+sigma)))^2 <= beta_1 < 1, D = (1-beta_1) sqrt(eps)/sqrt(96 c),
gamma = beta_1 D / sqrt(1-beta_1), 0 < nu <= G+sigma, and

    max{ 1 - nu/(G+sigma), beta_1^2 } <= beta_2 < 1,                        (**)

i.e. back to the STANDARD beta_2 >= beta_1^2 -- that is the point of the theorem:
without clipping, ||Delta_t||^2 cannot be bounded, so the analysis needs
beta_1/(2 alpha_{t-1}) - 1/(2 alpha_t) <= 0. If

    T >= max{ (1/(1-beta_1)) max{16 F* sqrt(96c)/eps^{3/2}, 48(G+sigma)/eps},
              ln2/(1-beta_2) }

then E[ ||grad F(xbar)||_c ] <= eps.

WHAT THE PREVIOUS (REJECTED) ATTEMPT DID WRONG
----------------------------------------------
It ran an off-the-shelf Adam on f(x) = 0.5||x||^2 and checked that the loss went
down. That tests neither the composite loss, nor the damping term, nor the
beta_2 condition, nor the claimed rate. Here the update is Eq. (10) verbatim
including the gamma*mu*(1-beta_1^t) damping, the parameters are the theorem's
own, the objectives are nonconvex and satisfy Assumptions 1-4 (audited
numerically), and the conclusion tested is E[||grad F(xbar)||_c] <= eps.

DEVIATION, RECORDED
-------------------
Theorem 7 as printed leaves mu 'to be tuned' without fixing a value. The
companion Theorem 8 -- same algorithm, same D and gamma -- sets
mu = 24 c D / (1-beta_1)^2. That value is used here. Since Theorem 7 is an
existence-of-a-tuning statement, using the paper's own tuning from the adjacent
theorem is the faithful reading, but it is a choice and is flagged as such.

THE INTERESTING CONTRAST WITH CLAIM 4
-------------------------------------
Theorem 5 permits beta_2 down to beta_1^4 *because* clipping bounds ||Delta_t||;
Theorem 7 does not. Running clip-free Adam at beta_2 inside [beta_1^4, beta_1^2)
is therefore outside Theorem 7's scope and is reported as an exploratory contrast,
never as a pass or a failure of the claim.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

from . import adam_claims as AC
from . import o2nc
from .common import ClaimResult, Timer, banner, write_csv, write_json

CLAIM_ID = "claim5_theorem7"
VARIANT = "clipfree"
PARAMS = o2nc.theorem7_params


def build_settings() -> list[dict[str, Any]]:
    settings = []
    for objective in AC.OBJECTIVES:
        for d in (2, 5):
            for eps in (None,):
                for c in (0.5, 2.0):
                    for choice in ("lower", "standard", "high"):
                        settings.append(AC.make_setting(VARIANT, objective, d, eps, c, 0.25, choice, PARAMS))
    for objective in ("two_well", "sharp_cone"):
        for choice in ("lower", "standard"):
            settings.append(AC.make_setting(VARIANT, objective, 3, None, 1.0, 0.25, choice, PARAMS, eps_ratio=0.45))
    return settings


def run() -> ClaimResult:
    banner("CLAIM 5 - Theorem 7: clip-free Adam, composite loss with damping")
    with Timer() as timer:
        nw = AC.n_workers()
        settings = build_settings()
        print(f"settings: {len(settings)}   worker processes: {nw}")
        s0 = settings[0]
        print(f"clip-free condition (**): beta_2 >= max(1-nu/(G+sigma), beta_1^2) = {s0['beta2_lower_bound']:.6f}")
        print(f"damping term mu = 24 c D/(1-beta_1)^2 = {s0['mu']:.6g}  (gamma*mu = {s0['gamma']*s0['mu']:.6g})")

        print("\n  Assumption audit (Assumptions 1 and 4, checked numerically):")
        audits = []
        for objective in AC.OBJECTIVES:
            for d in (2, 3, 5):
                obj = o2nc.Objective(objective, d, 0.5, seed=17)
                a = o2nc.audit_assumptions(obj, n=30000)
                a.update({"objective": objective, "d": d})
                audits.append(a)
        audit_ok = all(a["A1_holds"] and a["A4_variance_holds"] and a["A4_bounded_holds"]
                       and a["A4_unbiased_within_mc_error"] for a in audits)
        print(f"    all {len(audits)} objective/dimension combinations satisfy Assumptions 1 and 4: {audit_ok}")

        with mp.Pool(nw) as pool:
            print("\n  Running Algorithm 2 (clip-free, Eq. 10) at the theorem's own T and parameters:")
            rows = AC.evaluate_settings(settings, pool)
            for i, r in enumerate(rows):
                print(f"    [{i+1:>3}/{len(settings)}] {r['objective']:<22} d={r['d']} eps={r['eps']} "
                      f"c={r['c']} b2={r['beta2']:.6f} T={r['T']:<8} "
                      f"E<={r['E_grad_c_upper_2se']:.4f}  holds={r['conclusion_holds']}")
            n_fail = sum(1 for r in rows if not r["conclusion_holds"])
            print(f"\n  failures of the conclusion: {n_fail} / {len(rows)}")

            print("\n  negative controls (each must make the guarantee FAIL):")
            controls = []
            base = AC.make_setting(VARIANT, "sharp_cone", 3, None, 1.0, 0.25, "standard", PARAMS)

            starved = {**base, "T": max(2, base["T_theoretical"] // 100000)}
            r_a = AC.evaluate_setting(starved, pool, n_runs=8)
            controls.append({
                "control": "T_far_below_threshold",
                "description": f"run only T={starved['T']} rounds instead of the theorem's T={base['T_theoretical']}",
                "expected": "E[||grad F||_c] exceeds eps",
                "measured_upper_2se": r_a["E_grad_c_upper_2se"], "eps": r_a["eps"],
                "behaved_as_designed": bool(not r_a["conclusion_holds"]),
            })

            # The obvious noise control has NO POWER here and was measured to
            # have none: injecting 40x the declared sigma still gave E = 0.084
            # against eps = 0.6, because clip-free Adam with the damping term is
            # genuinely noise-robust. A control that cannot fail proves nothing,
            # so it is replaced by violating a condition that IS load-bearing:
            # nu enters the step-size denominator, so nu = 2000(G+sigma) shrinks
            # every step by ~2000x and the theorem's T is no longer enough. Same
            # calibrated choice already verified to fire for Theorem 5.
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
                m = cc.get("measured_upper_2se", cc.get("measured_max_upper_2se"))
                print(f"    {cc['control']:<32} measured={m:.4f} vs eps={cc['eps']:.4f}  "
                      f"as_designed={cc['behaved_as_designed']}")

            # ---- exploratory contrasts (outside Theorem 7's scope) -------------
            print("\n  exploratory contrasts (reported, not scored):")
            b1 = base["beta1"]
            below = {**base, "beta2": 0.5 * (b1**4 + b1**2)}   # inside Thm 5's band, outside Thm 7's
            r_below = AC.evaluate_setting(below, pool, n_runs=8)
            print(f"    clip-free at beta_2={below['beta2']:.6f} in [beta_1^4, beta_1^2) "
                  f"(Theorem 5's band, outside Theorem 7's): E<={r_below['E_grad_c_upper_2se']:.4f}, "
                  f"holds={r_below['conclusion_holds']}")

            undamped = {**base, "mu": 0.0}  # removes the composite loss's damping
            r_undamped = AC.evaluate_setting(undamped, pool, n_runs=8)
            print(f"    damping removed (mu=0, i.e. no composite loss): E<={r_undamped['E_grad_c_upper_2se']:.4f}, "
                  f"holds={r_undamped['conclusion_holds']}  "
                  f"[gamma*mu was {base['gamma']*base['mu']:.4g} vs nu={base['nu']:.4g}]")

            print("\n  conservatism diagnostic (how far below the theorem's T does it still work?):")
            diag = AC.conservatism_diagnostic(base, pool)
            for row in diag["rows"]:
                print(f"    T_scale={row['T_scale']:<8} T={row['T']:<9} E<={row['upper_2se']:.4f} holds={row['holds']}")

        print("\n  iteration complexity (symbolic, from the T formula):")
        comp = AC.complexity_symbolic("theorem7")
        print(f"    branch1 / (F* (G+sigma)^2 c^(1/2) eps^(-7/2)) = {comp['branch1_over_target']} "
              f"(constant: {comp['branch1_is_constant_multiple']})")
        comp_ok = (comp["branch1_is_constant_multiple"] and comp["branch2_is_constant_multiple"]
                   and comp["branch3_is_constant_multiple"])

    cols = [c for c in rows[0] if c != "per_run_means"]
    write_csv(CLAIM_ID, "settings_results.csv", [{k: r[k] for k in cols} for r in rows], fieldnames=cols)
    write_json(CLAIM_ID, "per_run_values.json", [{"objective": r["objective"], "d": r["d"], "eps": r["eps"],
               "c": r["c"], "beta2": r["beta2"], "per_run_means": r["per_run_means"]} for r in rows])
    write_json(CLAIM_ID, "assumption_audit.json", audits)
    write_json(CLAIM_ID, "negative_controls.json", controls)
    write_json(CLAIM_ID, "complexity_symbolic.json", comp)
    write_json(CLAIM_ID, "conservatism_diagnostic.json", diag)
    write_json(CLAIM_ID, "exploratory_contrasts.json",
               {"clipfree_in_theorem5_band": r_below, "damping_removed_mu_zero": r_undamped,
                "note": "outside Theorem 7's stated condition; reported for context, not scored"})

    controls_ok = all(c["behaved_as_designed"] for c in controls)
    # controls_ok is part of the VERDICT, not just of the run gate. A sweep
    # whose negative controls never fail has no power to distinguish the theorem
    # from a weaker statement, and cannot support VERIFIED however many settings
    # passed. Claim 5 previously reported VERIFIED with a dead noise control.
    verdict = ("VERIFIED" if (n_fail == 0 and audit_ok and comp_ok and controls_ok)
               else ("FALSIFIED" if n_fail > 0 else "BLOCKED"))
    write_json(CLAIM_ID, "summary.json", {
        "n_settings": len(rows), "n_failures": n_fail, "assumption_audit_ok": audit_ok,
        "complexity_symbolic_ok": comp_ok, "controls_ok": controls_ok,
        "worst_ratio_estimate_over_eps": max(r["ratio_estimate_over_eps"] for r in rows),
        "mu_used": s0["mu"], "mu_source": "Theorem 8 (same algorithm), since Theorem 7 leaves mu unfixed",
        "conservatism_factor": diag["approx_conservatism_factor"], "verdict": verdict,
    })

    notes = [
        "The update is Eq. (10) verbatim, including the gamma*mu*(1-beta_1^t) damping that distinguishes "
        "clip-free Adam from plain Adam; the rejected baseline used an off-the-shelf optimiser on a quadratic.",
        "DEVIATION: Theorem 7 leaves mu 'to be tuned'. mu = 24 c D/(1-beta_1)^2 is taken from the companion "
        "Theorem 8, which shares the algorithm, D and gamma. This is a choice and is recorded as such.",
        "Objectives are nonconvex and Assumptions 1 and 4 are re-audited numerically for every "
        "objective/dimension pair rather than assumed from the construction.",
        "||grad F(x)||_c is upper-bounded by Gaussian smoothing and acceptance uses a 2-standard-error upper "
        "end, so all approximations are conservative.",
        "Scope: scoped corroboration over four objective families; the theorem quantifies over all F "
        "satisfying Assumptions 1-4, which no finite experiment can exhaust.",
    ]
    return ClaimResult(
        claim_id=CLAIM_ID,
        title="Theorem 7 - clip-free Adam with composite loss (Section 4.4)",
        verdict=verdict,
        # ok gates the RUN, not the science: BLOCKED is an honest outcome and
        # must not be reported as broken infrastructure. A DEAD instrument --
        # not one single control firing -- still fails, because then nothing was
        # actually tested.
        ok=any(c["behaved_as_designed"] for c in controls),
        headline={
            "n_settings": len(rows),
            "n_failures_of_conclusion": n_fail,
            "beta2_condition_lower_bound": s0["beta2_lower_bound"],
            "mu_used": s0["mu"],
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
