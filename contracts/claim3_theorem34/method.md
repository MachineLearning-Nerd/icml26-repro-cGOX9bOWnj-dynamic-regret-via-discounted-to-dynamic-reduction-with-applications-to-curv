# Method — Theorems 3 and 4 — discounted AIOLI and the Algorithm 1 ensemble

Paper: arXiv `2602.08372` / OpenReview `cGOX9bOWnj`.
Claim id: `claim3_theorem34`. Fixed run command: `bash scripts/run.sh`.

## What is being tested

The explicit bound (E3) of Theorem 3, and the asymptotic rate of Theorem 4 for the
Algorithm 1 two-layer ensemble. Also the comparative claim: (E3) depends on `B` only
through `(1+BR)`, i.e. **linearly**, avoiding the `e^B` degradation of proper ONS.

## The algorithm is the real one

The Section 3.2 update is **implicit**, and it is solved as the genuine implicit problem by
damped Newton with backtracking and **per-round convergence certification** (worst relative
gradient norm 5.1e-08 across all rounds). The optimism term `h_t`, the second-order
surrogate `fhat_s` and the discounting are all present.

The rejected 3/10 attempt used "logistic SGD as an AIOLI proxy" and checked only that
sigmoid outputs were finite and in (0,1) — true by construction, and true of *any*
implementation whatsoever.

Undamped Newton **diverged** here (dynamic regret 1.4e8, L1 slack -7.2e7). That was a
solver bug, not a finding, which is why convergence is now certified per round.

## Calibration, and its measured limits

A 680-configuration grid alone is **not** a test of Theorem 3: term4 `((1-beta)/beta)d(1+BR)T`
dominated the right-hand side almost everywhere, and 5 of 6 weakened bounds survived it
untouched. An adversarial search (2336 evaluations: random exploration plus 3 hill-climbing
rounds) was therefore pointed at the true bound **and** at each weakened variant under an
identical budget — one AIOLI run scores all seven targets, so calibration costs no more
than testing the bound alone.

Each control reports a **power ratio** = (bound removed) / (slack available); it must exceed
1 to fire.

| control | power | fires |
|---|---|---|
| `drop_log_term` | 118.5% | yes |
| `drop_last_term` | 108.4% | yes |
| `drop_path_term` | 103.6% | yes |
| `drop_B_dependence` | 98.9% | **no** |
| `zinkevich_path` | 93.4% | **no** |
| `halve_log_term` | 51.1% | **no** |

The three that do not fire are dead for a **structural** reason, measured directly: making
dynamic regret large requires a moving comparator, which makes
`term3 = (beta/(1-beta)) P_T^beta` dominate by three orders of magnitude (181482 vs 457 for
the log term at `d=5, B=8, beta=0.999`); making the log term dominate requires a static
comparator, against which **improper** AIOLI beats the comparator outright and dynamic
regret goes negative (-92). The two requirements are incompatible.

Escalation was tried and reported rather than hidden: 2.4x the search budget moved
`drop_B_dependence` only 96.7% -> 98.9%, and a hand-seeded focused scan reached only 33%.

## Consequence

(E3) is corroborated only **up to the constant in its log term and its stated
B-dependence**. Those parts are UNTESTED by this instrument, not confirmed by it. Hence
**BLOCKED**, never VERIFIED.

## B-dependence

Stated in **levels**, not growth ratios: AIOLI is improper, so its regret against a
ball-constrained comparator legitimately goes negative (-135.96 at B=0.5), which makes
ratio statistics meaningless. Over `B in [0.5, 10]` at `T=400, d=3`, AIOLI stays at or below
3.81 and inside (E3) at every `B`, while proper ONS rises to 235.65.

LIMITATION: this separates the algorithms but does **not** resolve exponential-vs-polynomial
asymptotics in `B` — the range is too short and constants dominate. The `e^B` statement
concerns the proper-learning lower bound of Hazan et al. (2014) and is not measured here.
