"""Lemma 25 -- a counterexample found while reconstructing the Theorem 2 derivation.

WHAT LEMMA 25 SAYS (Appendix, stated as Lemma G.2 of Jacobsen and Cutkosky 2024)
--------------------------------------------------------------------------------
Let beta in (0,1], lambda > 0, z_t in R^d. Define A_0 = lambda I and
A_t = z_t z_t^T + beta A_{t-1}. Then for ANY sequence c_1, c_2, ... in R:

    sum_{t=1..T} c_t^2 z_t^T A_t^{-1} z_t
        <= d ln(1/beta) sum_{t=1..T} c_t^2
         + { max_{t in [T]} c_t^2 } * d ln( 1 + (sum_{t=1..T} beta^{T-t}||z_t||^2)/(lambda d) ).   (L25)

Theorem 2's proof applies (L25) with c_t = y_t to convert the accumulated
stability terms into the log-determinant and discount-slack terms of the final
bound. It is the only step in that derivation that is neither a definition nor
Theorem 1 itself.

WHAT WE FOUND
-------------
(L25) is false as stated, and not marginally. A designed family of d = 1
instances violates it with LHS/RHS ratios from 2.2 up to 68.8, certified in
60-decimal-digit arithmetic (mpmath), so these are not floating-point artifacts.
A non-degenerate variant (every z_t and c_t strictly nonzero) violates it too.
The ratio is unbounded: it grows as beta -> 1 with lambda and T scaled to match.

WHY IT FAILS (the mechanism, not just the instance)
---------------------------------------------------
Because A_t contains the term z_t z_t^T, the ratio z_t^T A_t^{-1} z_t can be
arbitrarily close to 1 when the current feature dominates the discounted history.
A single round t* with c_{t*}^2 = max_t c_t^2 therefore contributes nearly
max_t c_t^2 to the left-hand side. The right-hand side allocates that round only

    ln(1/beta) * max c^2   +   max c^2 * d ln(1 + ...),

so whenever ln(1/beta) + d ln(1 + sum beta^{T-t}||z_t||^2/(lambda d)) < 1 and the
c sequence is spiky, (L25) is violated. In the certified instance those two terms
can be driven arbitrarily close to 0 (large lambda shrinks the log-determinant
term; beta -> 1 shrinks ln(1/beta)), while the LHS stays at ~1. The failure is
therefore structural, not a knife-edge accident.

A TRUE REPLACEMENT, AND WHY IT DOES NOT RESCUE THE STATED CONSTANTS
-------------------------------------------------------------------
Since z_t^T A_t^{-1} z_t >= 0, pulling the weight out gives the valid bound

    sum_t c_t^2 z_t^T A_t^{-1} z_t
        <= (max_t c_t^2) * [ d ln(1/beta) T + d ln(1 + sum_t beta^{T-t}||z_t||^2/(lambda d)) ],   (L25')

which this module verifies on the same instances that break (L25). But (L25')
carries a factor T in the first term where (L25) carries sum_t c_t^2, so
substituting it into the Theorem 2 derivation yields a discount-slack term of
order (1-beta)/beta * d T max_t y_t^2 rather than the stated
(1-beta)/beta * (d/2) sum_t y_t^2. The published constants do not follow from the
repaired lemma.

WHAT THIS DOES AND DOES NOT IMPLY
---------------------------------
It does NOT refute Theorem 2. The chain is

    D-Reg <= Eq.(12) = (beta lambda/2)||u_1||^2 + (1/2) sum_t lam_t + path,

and Theorem 2's bound (E) replaces (1/2) sum_t lam_t by the (L25)-derived terms.
Even where (L25) fails, Eq.(12) is itself slack relative to the true dynamic
regret, so (E) can still hold -- and in every configuration tested it does.
The finding is a gap in the derivation AS PRESENTED, and it is what motivates
the dedicated falsification route against (E) itself (see claim2_theorem2.py).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

CERTIFIED_PRECISION_DPS = 60


def lemma25_sides(z: list[float], c: list[float], beta: float, lam: float, dps: int = CERTIFIED_PRECISION_DPS):
    """Both sides of (L25) for d = 1, in arbitrary-precision arithmetic."""
    from mpmath import log as mlog
    from mpmath import mp, mpf

    mp.dps = dps
    T = len(z)
    A = mpf(lam)
    lhs = mpf(0)
    for t in range(T):
        A = mpf(z[t]) ** 2 + mpf(beta) * A  # A_t = z_t z_t^T + beta A_{t-1}
        lhs += mpf(c[t]) ** 2 * mpf(z[t]) ** 2 / A
    S = sum(mpf(beta) ** (T - 1 - t) * mpf(z[t]) ** 2 for t in range(T))
    rhs = mlog(1 / mpf(beta)) * sum(mpf(x) ** 2 for x in c) + max(mpf(x) ** 2 for x in c) * mlog(
        1 + S / mpf(lam)
    )
    return lhs, rhs, rhs - lhs


def lemma25_repaired_sides(z: list[float], c: list[float], beta: float, lam: float, dps: int = CERTIFIED_PRECISION_DPS):
    """Both sides of the repaired bound (L25'), which pulls max c^2 out of both terms."""
    from mpmath import log as mlog
    from mpmath import mp, mpf

    mp.dps = dps
    T = len(z)
    A = mpf(lam)
    lhs = mpf(0)
    for t in range(T):
        A = mpf(z[t]) ** 2 + mpf(beta) * A
        lhs += mpf(c[t]) ** 2 * mpf(z[t]) ** 2 / A
    S = sum(mpf(beta) ** (T - 1 - t) * mpf(z[t]) ** 2 for t in range(T))
    cmax = max(mpf(x) ** 2 for x in c)
    rhs = cmax * (mlog(1 / mpf(beta)) * T + mlog(1 + S / mpf(lam)))
    return lhs, rhs, rhs - lhs


def constructed_family(delta: float = 0.0) -> list[dict[str, Any]]:
    """A designed counterexample family, which refutes (L25) far more decisively
    than random search and shows the violation ratio is unbounded.

    The construction follows directly from the failure mechanism. Put the peak in
    the LAST round and make the history negligible:

        z_s = delta (s < T),  z_T = 1,     c_s = delta (s < T),  c_T = 1,
        lambda large,  T large enough that lambda * beta^T is negligible.

    Then A_T ~ 1, so the single term c_T^2 z_T^2 / A_T ~ 1 dominates the LHS.
    Meanwhile sum_t beta^{T-t}||z_t||^2 ~ 1, so the RHS is only
    ln(1/beta) + ln(1 + 1/lambda), which can be made as small as desired by
    taking lambda large and beta close to 1 (with T grown to match). The ratio
    LHS/RHS therefore diverges.

    delta = 0 gives the cleanest instance; delta > 0 gives a NON-DEGENERATE one
    (every z_t and c_t strictly nonzero) so the refutation cannot be dismissed as
    an artifact of vanishing features.
    """
    out = []
    for beta, T, lam in [(0.7, 50, 10.0), (0.9, 100, 100.0), (0.95, 200, 1000.0),
                         (0.99, 1000, 1e4), (0.999, 10000, 1e6)]:
        z = [delta] * (T - 1) + [1.0]
        c = [delta] * (T - 1) + [1.0]
        lhs, rhs, slack = lemma25_sides(z, c, beta, lam)
        out.append({
            "beta": beta, "T": T, "lambda": lam, "delta": delta, "d": 1,
            "lhs": float(lhs), "rhs": float(rhs), "slack": float(slack),
            "violation_ratio_lhs_over_rhs": float(lhs / rhs),
            "violates_L25": bool(slack < 0),
        })
    return out


def search_counterexample(n_trials: int = 40000, seed: int = 5) -> dict[str, Any]:
    """Constructed family (primary evidence) plus a random search for context."""
    constructed = constructed_family(delta=0.0)
    constructed_nondegenerate = constructed_family(delta=0.01)

    rng = random.Random(seed)
    worst = None
    n_violating = 0
    for _ in range(n_trials):
        T = rng.randint(2, 12)
        beta = rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        lam = rng.choice([0.1, 1.0, 10.0])
        z = [rng.gauss(0, 1) for _ in range(T)]
        c = [rng.gauss(0, 1) for _ in range(T)]
        _, _, slack = lemma25_sides(z, c, beta, lam, dps=25)  # fast pass
        if slack < 0:
            n_violating += 1
        if worst is None or slack < worst[0]:
            worst = (slack, T, beta, lam, z, c)

    # The certified counterexample is the CONSTRUCTED one: random search hits this
    # region too rarely to be the primary evidence (it is reported for context).
    bc = min(constructed, key=lambda r: r["violation_ratio_lhs_over_rhs"] * -1)
    T, beta, lam = bc["T"], bc["beta"], bc["lambda"]
    z = [bc["delta"]] * (T - 1) + [1.0]
    c = [bc["delta"]] * (T - 1) + [1.0]
    lhs, rhs, s_exact = lemma25_sides(z, c, beta, lam, dps=CERTIFIED_PRECISION_DPS)
    rlhs, rrhs, r_slack = lemma25_repaired_sides(z, c, beta, lam, dps=CERTIFIED_PRECISION_DPS)

    # the mechanism: the two RHS allocation factors for the peak round
    from mpmath import log as mlog
    from mpmath import mp, mpf

    mp.dps = CERTIFIED_PRECISION_DPS
    S = sum(mpf(beta) ** (T - 1 - t) * mpf(z[t]) ** 2 for t in range(T))
    log_beta_term = float(mlog(1 / mpf(beta)))
    log_det_term = float(mlog(1 + S / mpf(lam)))

    best_constructed = min(constructed, key=lambda r: r["slack"])
    return {
        "constructed_family": constructed,
        "constructed_family_nondegenerate": constructed_nondegenerate,
        "constructed_all_violate": all(r["violates_L25"] for r in constructed),
        "constructed_nondegenerate_all_violate": all(r["violates_L25"] for r in constructed_nondegenerate),
        "max_violation_ratio": max(r["violation_ratio_lhs_over_rhs"] for r in constructed),
        "best_constructed": best_constructed,
        "n_random_trials": n_trials,
        "n_violating_random_instances": n_violating,
        "violation_rate": n_violating / n_trials,
        "random_search_note": ("random sampling reaches this region rarely; the constructed family "
                               "above is the primary evidence and the certified instance is drawn from it"),
        "certified_counterexample": {
            "d": 1, "T": T, "beta": beta, "lambda": lam,
            "z": [float(v) for v in z], "c": [float(v) for v in c],
            "lhs": str(lhs), "rhs": str(rhs), "slack_rhs_minus_lhs": str(s_exact),
            "lhs_float": float(lhs), "rhs_float": float(rhs), "slack_float": float(s_exact),
            "precision_dps": CERTIFIED_PRECISION_DPS,
            "violates_L25": bool(s_exact < 0),
        },
        "mechanism": {
            "ln_one_over_beta": log_beta_term,
            "log_determinant_factor": log_det_term,
            "sum_of_allocation_factors": log_beta_term + log_det_term,
            "explanation": (
                "z_t^T A_t^{-1} z_t can approach 1 because A_t contains z_t z_t^T, so a single "
                "peak round contributes nearly max_t c_t^2 to the LHS while the RHS allocates it "
                "only (ln(1/beta) + d ln(1 + ...)) * max_t c_t^2. Whenever that sum is below 1 and "
                "c is spiky, (L25) fails."
            ),
            "sum_below_one": bool(log_beta_term + log_det_term < 1.0),
        },
        "repaired_bound_L25prime": {
            "lhs_float": float(rlhs), "rhs_float": float(rrhs), "slack_float": float(r_slack),
            "holds_on_the_same_instance": bool(r_slack >= 0),
            "why_it_does_not_rescue_theorem2": (
                "(L25') carries max_t c_t^2 * d ln(1/beta) * T where (L25) carries d ln(1/beta) sum_t c_t^2. "
                "Substituting it into the Appendix C.1 derivation gives a discount-slack term of order "
                "(1-beta)/beta * d T max_t y_t^2, not the stated (1-beta)/beta * (d/2) sum_t y_t^2."
            ),
        },
    }


def verify_repaired_bound(n_trials: int = 20000, seed: int = 909) -> dict[str, Any]:
    """(L25') must hold everywhere (it follows from z^T A^{-1} z >= 0 and the c=1 case)."""
    rng = random.Random(seed)
    worst = None
    fails = 0
    for _ in range(n_trials):
        T = rng.randint(2, 14)
        beta = rng.choice([0.3, 0.5, 0.7, 0.9, 0.99])
        lam = rng.choice([0.01, 0.1, 1.0, 10.0, 100.0])
        z = [rng.gauss(0, 1) * rng.choice([0.1, 1, 10]) for _ in range(T)]
        c = [rng.gauss(0, 1) * rng.choice([0.1, 1, 10]) for _ in range(T)]
        _, _, s = lemma25_repaired_sides(z, c, beta, lam, dps=25)
        if s < 0:
            fails += 1
        if worst is None or s < worst:
            worst = s
    return {"n_trials": n_trials, "violations_of_repaired_bound": fails,
            "worst_slack": float(worst), "repaired_bound_holds": bool(fails == 0)}
