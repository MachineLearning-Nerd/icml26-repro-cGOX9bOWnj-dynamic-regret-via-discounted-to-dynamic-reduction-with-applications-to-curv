# Method — Theorem 7 — clip-free Adam with composite loss

Paper: arXiv `2602.08372` / OpenReview `cGOX9bOWnj`.
Claim id: `claim5_theorem7`. Fixed run command: `bash scripts/run.sh`.

## What is being tested

As Claim 4, for the **clip-free** Adam update (Eq. 10) with the `gamma*mu*(1-beta_1^t)`
damping term, keeping the standard condition
`beta_2 >= max(1 - nu/(G+sigma), beta_1^2)`.

52 settings at the theorem's own `T`, up to `T = 23,866,836`.

## Recorded deviation

Theorem 7 leaves `mu` unfixed, so `mu = 24cD/(1-beta_1)^2` is taken from **Theorem 8** (same
algorithm). 16 distinct `mu` values are used, all > 0. This is a deviation from the paper as
written and is recorded as such rather than presented as the paper's own choice.

## Measurement, non-vacuity, assumption audit

Identical to Claim 4: Gaussian-smoothing upper bound on an infimum (conservative by
Jensen), acceptance at mean + 2 SE, `eps = 0.6 ||grad F(x_0)||_c` with every setting
starting above `eps` (min ratio 1.667), and Assumptions 1 and 4 audited numerically across
12 objective/dimension pairs.

## Negative controls — and one that had to be replaced

The original `assumption4_violated` control injected **40x** the declared sigma and still
gave `E = 0.084` against `eps = 0.6`. Clip-free Adam with the damping term is genuinely
noise-robust, so that control had **no power** — and the claim had reported VERIFIED with it
dead. Two fixes followed:

1. it was replaced by the calibrated `nu_condition_violated` control already shown to fire
   for Theorem 5;
2. `controls_ok` became part of the **verdict**, not merely the run gate. A sweep whose
   controls never fail cannot distinguish the theorem from a weaker statement and cannot
   support VERIFIED however many settings passed.

All three controls now fire.

## Calibration

Conservatism diagnostic: conclusion **fails** at `T_scale = 0.01`, holds at `0.1`. The
theorem's `T` is needed to within about 10x.

## Exploratory contrasts — reported, not scored

- clip-free at `beta_2` inside Theorem 5's relaxed band (outside Theorem 7's condition):
  still holds, `E <= 0.0221`.
- damping removed entirely (`mu = 0`, i.e. no composite loss): still holds, `E <= 0.0198`.

Both are outside the theorem's stated hypotheses and are therefore reported as context,
never counted as confirmations.
