import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dynamic Regret via Discounted-to-Dynamic Reduction — reproduction

    arXiv `2602.08372` / OpenReview `cGOX9bOWnj`.

    This notebook **recomputes** the two results that carry the most weight, live, when you run
    it. Nothing is loaded from a results file — every number below is produced by the cells you
    can see.

    1. **Theorem 1's proof certificate** — the theorem is an algebraic implication, so it is
       *proved*, not simulated.
    2. **The counterexample to Lemma 25** — the appendix lemma that the published proof of
       Theorem 2 routes through, which is false as stated.

    It calls the repository's own verifiers (`repro.claim1_theorem1`, `repro.lemma25`) rather
    than re-deriving the algebra inline. That is deliberate. An earlier draft of this notebook
    hand-rewrote both computations "for readability" and got **both wrong** — the certificate
    identity failed on 400 of 400 instances and the Lemma 25 family showed no violation at all,
    because the reconstruction had silently dropped the path and drift terms. `marimo check`
    passed the whole time: linting is not correctness. Running the audited code is the point.

    Run from the repository root:

    ```bash
    uv run --with marimo marimo edit notebooks/reproduction.py
    ```
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    # import the repository's verifiers, whatever directory the notebook runs from
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from repro import claim1_theorem1 as C1
    from repro import lemma25 as L25

    return C1, L25, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Theorem 1 — a certificate, not a simulation

    Theorem 1 says: if for every `t` and every `u` the *hypothesis*

    $$\sum_{s\le t}\beta^{-s} f_s(x_s) - \sum_{s\le t}\beta^{-s} f_s(u)
      \;\le\; \varphi_t(u) + \sum_{s\le t}\Lambda_s$$

    holds, then the dynamic regret obeys the theorem's bound.

    No amount of sampling establishes a statement quantified over *all* sequences. But the
    implication is algebraic, so it admits a **certificate**. Writing the hypothesis slack as

    $$S_t(u) \;=\; \beta^t\varphi_t(u) + \beta^t\sum_{s\le t}\Lambda_s - \mathrm{Reg}_{t;\beta}(u)
      \;\ge\; 0 ,$$

    the claim is the identity

    $$\text{RHS} - \text{LHS} \;=\; (1-\beta)\sum_{t=1}^{T} S_t(u_t) \;+\; \beta\, S_T(u_T).$$

    Both coefficients are non-negative *precisely* on the stated domain $\beta\in(0,1]$, so the
    identity together with $S_t\ge0$ proves the theorem. Two things must hold, and they are
    checked separately:

    - the **identity** is an algebraic fact about the expressions, true whether or not the
      hypothesis holds;
    - the **hypothesis** $S_t\ge0$ is what turns that identity into the inequality.

    Below, route A verifies the identity symbolically for every $T$ from 1 to 20 — no numeric
    substitution, free symbols throughout.
    """)
    return


@app.cell
def _(C1):
    # both verifier entry points return (all_ok, rows)
    route_a_ok, route_a = C1.route_a_exhaustive_symbolic()
    route_a
    return route_a, route_a_ok


@app.cell(hide_code=True)
def _(mo, route_a, route_a_ok):
    _n = len(route_a)
    _ok = sum(1 for r in route_a if r["residual_is_identically_zero"])
    mo.md(
        f"""
        **Result.** The certificate residual simplified to **identically zero** for
        **{_ok} of {_n}** horizons $T = 1 \\dots {_n}$, symbolically.

        {"✅ the identity holds for every $T$ checked" if (_ok == _n and route_a_ok) else "❌ the identity FAILED somewhere — that would refute the certificate"}

        Route A covers $T \\le 20$ exhaustively. The full verifier adds route B, which proves the
        same identity for **all** $T$ by matching coefficients across five symbol classes with
        $T$, $s$, $k$ left as free integer symbols, and route C, which cross-checks 4000
        instances in exact rational arithmetic to catch a mis-transcription of the theorem. Route
        C's worst margin is exactly `0`, so the bound is **tight**, not merely true.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The negative controls

    An identity that holds is only meaningful if a *damaged* identity would fail. Each control
    below weakens one term of the theorem's bound; every one must break, and at the smallest
    horizon where the damaged term first matters.
    """)
    return


@app.cell
def _(C1):
    controls_ok, controls = C1.negative_controls()
    controls
    return controls, controls_ok


@app.cell(hide_code=True)
def _(controls, controls_ok, mo):
    _body = "\n".join(
        f"| `{c['control']}` | {c['description']} | {c['broke_at_T']} | "
        f"{'✅ broke, as designed' if c['behaved_as_designed'] else '❌ did NOT break'} |"
        for c in controls
    )
    _all = controls_ok and all(c["behaved_as_designed"] for c in controls)
    mo.md(
        f"""
        | control | what it damages | first $T$ at which it fails | outcome |
        |---|---|---|---|
        {_body}

        {"✅ every control breaks — the check has the power to detect a false certificate" if _all else "❌ a control survived: the check would not detect a false certificate"}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Lemma 25 is false as stated

    The published proof of Theorem 2 passes through an appendix lemma (attributed to Lemma G.2
    of Jacobsen & Cutkosky 2024) of the elliptical-potential form

    $$\sum_{t=1}^{T} c_t^2\, z_t^{\top} A_t^{-1} z_t
      \;\;\le\;\; \Big(\ln\tfrac{1}{\beta} + d\ln\big(1 + \cdot\big)\Big)\max_t c_t^2 ,
      \qquad A_t = \lambda\beta^{t} I + \sum_{s\le t}\beta^{t-s} z_s z_s^{\top}.$$

    **Why it fails.** $A_t$ already contains $z_t z_t^{\top}$, so $z_t^{\top}A_t^{-1}z_t \to 1$
    when $z_t$ dominates the accumulated matrix. A single peak round therefore contributes about
    $\max_t c_t^2$ to the left-hand side, while the right-hand side allocates it only
    $(\ln\frac1\beta + d\ln(1+\cdot))\max_t c_t^2$ — and large $\lambda$ together with
    $\beta\to1$ drives that factor toward **zero**.

    The family below has one round of magnitude 1 and the rest of magnitude $\delta$, with
    $\lambda$ large and $\beta$ approaching 1. It is evaluated in high-precision arithmetic, so
    the violation is not a floating-point artifact.
    """)
    return


@app.cell
def _(L25):
    family = L25.constructed_family(delta=0.0)
    nondegenerate = L25.constructed_family(delta=0.01)
    family
    return family, nondegenerate


@app.cell(hide_code=True)
def _(family, mo, nondegenerate):
    _body = "\n".join(
        f"| {r['T']} | {r['beta']} | {r['lambda']:g} | {float(r['lhs']):.6f} | "
        f"{float(r['rhs']):.6f} | **{float(r['violation_ratio_lhs_over_rhs']):.2f}×** | "
        f"{'❌ violated' if r['violates_L25'] else 'holds'} |"
        for r in family
    )
    _worst = max(float(r["violation_ratio_lhs_over_rhs"]) for r in family)
    _all_deg = all(r["violates_L25"] for r in family)
    _all_non = all(r["violates_L25"] for r in nondegenerate)
    mo.md(
        f"""
        | T | β | λ | LHS | RHS | LHS/RHS | Lemma 25 |
        |---|---|---|---|---|---|---|
        {_body}

        Worst ratio here: **{_worst:.2f}×**, growing as $\\beta\\to1$ and $\\lambda$ grows —
        and unbounded in the limit.

        - all rows of the constructed family violate the lemma: **{_all_deg}**
        - all rows of a *non-degenerate* variant (every $z_t$, $c_t$ nonzero, so this is not an
          artifact of setting entries to zero) violate it: **{_all_non}**

        ### What this does and does not mean

        It does **not** refute Theorem 2. Eq. (12) is itself slack relative to true dynamic
        regret, so the theorem can hold where its published intermediate step does not — and
        across 987 configurations, plus a falsification route seeded *specifically inside the
        region where Lemma 25 fails*, the theorem's bound was never broken.

        What it means is that Theorem 2's **published derivation does not close**. Under the
        non-circularity rule — a universally quantified theorem cannot be marked VERIFIED on
        finite corroboration when its proof has a gap — the verdict is **BLOCKED**: not
        VERIFIED, and not FALSIFIED either.

        A repaired bound (pulling $\\max_t c_t^2$ out of both terms) does hold, but it carries a
        factor $T$ and so does not reproduce Theorem 2's stated constants.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running the full reproduction

    This notebook covers two results. The complete suite — all five claims, the negative
    controls, the adversarial calibration and the independent checker — is one fixed command:

    ```bash
    bash scripts/run.sh
    ```

    See `README.md` for the experiment log, and `report/report.md` for the full write-up. The
    limitations section is the part worth reading closely: three of Theorem 3's six weakened
    bounds cannot be broken by any instrument built here, so parts of that theorem are
    **untested** rather than confirmed.
    """)
    return


if __name__ == "__main__":
    app.run()
