# Method — Theorem 5 — clipped Adam, relaxed beta_2 condition

Paper: arXiv `2602.08372` / OpenReview `cGOX9bOWnj`.
Claim id: `claim4_theorem5`. Fixed run command: `bash scripts/run.sh`.

## What is being tested

That Algorithm 2 (Exponentiated O2NC) with clipped Adam (Eq. 8), run at the theorem's own
`T` and with parameters from the theorem's own formulas, achieves
`E_{t ~ Unif[T]} ||grad F(xbar_t)||_c <= eps`.

The **novelty** is the relaxed condition `beta_2 in [beta_1^4, beta_1^2)` — a band that
prior `beta_2 >= beta_1^2` analyses forbid. 52 of 86 settings sit inside it.

## Measurement

`||grad F(x)||_c = inf over couplings P with E[y]=x of ||E grad F(y)|| + c E||y-x||^2` is an
**infimum**, so it is upper-bounded here by Gaussian smoothing. By Jensen that is
**conservative**: the test is harder than the theorem requires, never easier.

Acceptance is **mean + 2 standard errors**, not the mean.

## Non-vacuity

`eps = 0.6 * ||grad F(x_0)||_c`, and every setting must **start above** `eps`
(min ratio 1.667, required > 1). This is load-bearing: an earlier draft started at
approximately stationary points and "passed" at `T = 2` as easily as at the theorem's `T`,
with `E` around 0.0002 regardless. The wells were widened and start points moved into
high-gradient regions.

## Assumption audit

Assumptions 1 and 4 are re-checked **numerically** per objective rather than assumed from
construction. This caught a real error: `asym_valley`'s Lipschitz constant had been
eyeballed and **failed its own audit** until `G_F` was computed properly from `||Q||_op`,
`sqrt(d)` and the numerically maximised `|h(z)|`.

## Negative controls

- `T_far_below_threshold` — run `T = 10` instead of the theorem's `T`.
- `nu_condition_violated` — set `nu = 2000(G+sigma)`, violating `0 < nu <= G+sigma`, which
  shrinks every step ~2000x. The obvious noise control was **measured to have no power**
  (40x sigma still passed, because clipped Adam is genuinely noise-robust) and was replaced;
  a control that cannot fail proves nothing.
- `measurement_can_detect_failure` — re-test the same values against `eps/50`.

All three fire.

## Calibration

The conservatism diagnostic answers the "orders of magnitude" objection directly: the
conclusion **fails** at `T_scale = 0.01` and holds at `0.1`. The theorem's own `T` is needed
to within about 10x — this is not a vacuous test passed by an enormous margin.

## Guard

`beta2_choice` values meant to be admissible (`lower`, `standard`, `high`) are checked at
construction and raise if they fall below `max(1-nu/(G+sigma), beta_1^2)`. A hardcoded
`high = 0.999` previously put settings outside the theorem's own precondition, where they
were scored as confirmations of a theorem that does not cover them.
