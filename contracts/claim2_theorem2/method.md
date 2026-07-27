# Method — Theorem 2 — discounted VAW dynamic regret

Paper: arXiv `2602.08372` / OpenReview `cGOX9bOWnj`.
Claim id: `claim2_theorem2`. Fixed run command: `bash scripts/run.sh`.

## What is being tested

The explicit bound (E) of Theorem 2 for the discounted Vovk-Azoury-Warmuth forecaster,
quantified over all horizons, discount factors, dimensions, regularisers and comparator
sequences satisfying the stated norm bounds.

`P_T^beta` is Eq. (4), measured in **loss differences** with normalised geometric weights
`p^beta_{t,s} = beta^{t-s} / sum_{tau<=t} beta^{t-tau}` and `f_0 := phi`. It is explicitly
**not** the Zinkevich path length `sum ||u_{t+1} - u_t||`; a control substitutes the latter
and must break.

## Numerical method — the single most important decision here

The literal `Lambda_t = beta^{-2t} y_t^2 z_t^T A_t^{-1} z_t` **overflows**: at `beta = 0.99`,
`t = 3000` it is around `1e26`. A verifier that computes it directly reports `inf`/`nan`
and then "passes" **vacuously**, because every comparison against `inf` succeeds. All
quantities are therefore computed in `beta^t`-rescaled form:

    lam_t := y_t^2 z_t^T A_t^{-1} z_t ,   A_t = lambda beta^t I + sum_{s<=t} beta^{t-s} z_s z_s^T

with the recursions `S_t = beta S_{t-1} + z_t z_t^T` and `b_t = beta b_{t-1} + y_t z_t`.

## Design

- **987 configurations** across 9 data generators x 7 comparator strategies, arranged into
  regimes A–E so that a *different* term of the right-hand side is the binding one in each.
  An earlier draft had several regimes producing negative dynamic regret, which is
  uninformative; `unit_labels` data and an `oracle_tracking` comparator were added so each
  term binds somewhere.
- **Derivation links** L1–L5 checked per configuration, so the *route* to the bound is
  tested, not only its endpoint.
- **Independent algorithm check**: 5 configurations re-run through a from-scratch VAW
  implementation using no recursion; 0 dynamic-regret mismatches.
- **Calibration**: an adversarial search maximising (regret - bound), run against the true
  bound and against 5 weakened bounds at an identical budget. Search is boxed at |v| <= 100
  because an unconstrained optimiser escapes to |v| ~ 1e150, where squaring overflows and
  the margin reads -1e307 — a floating-point artifact, not a counterexample.
- **Route 4** (mandatory falsification route): 320 restarts seeded specifically inside the
  region where Lemma 25 fails.
- **Rate sub-claim** derived ALGEBRAICALLY from (E) via `min_q (P/q + qV) = 2 sqrt(PV)`.
  Deliberately not measured as an empirical slope: fitting a slope predicted by the formula
  under test would be circular.

## Result and verdict rule

(E) survived everything: 0 violations in 987 configurations, adversarial best margin
1.44e-26, max tightness 0.9283. Links L1, L2, L4, L5 hold everywhere.

**L3 — which is Lemma 25 — is false as stated**, certified at 60–80 decimal digits, with a
non-degenerate variant also violating and the ratio unbounded as `beta -> 1`.

VERIFIED requires (E) to survive AND every derivation link to hold. The second fails, so
the verdict is **BLOCKED** — not FALSIFIED, because (E) itself was never broken, and Eq.(12)
is slack relative to true dynamic regret so the theorem can hold where its published
intermediate step does not.
