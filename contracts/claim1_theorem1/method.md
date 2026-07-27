# Method — Theorem 1 — modular D2D reduction

Paper: arXiv `2602.08372` / OpenReview `cGOX9bOWnj`.
Claim id: `claim1_theorem1`. Fixed run command: `bash scripts/run.sh`.

## What is being tested

Theorem 1 is an **algebraic implication**: given a hypothesis (H) that holds at every
round `t` and for every comparator `u`, the dynamic-regret conclusion (C) follows. It is
quantified over all `T`, all `beta in (0,1]`, and all sequences — an infinite domain.

## Why this is not a simulation

Sampling cannot establish a universally quantified statement. But an algebraic implication
admits a **proof certificate**. Writing the hypothesis slack as

    S_t(u) = beta^t phi_t(u) + beta^t sum_{s<=t} Lambda_s - Reg_{t;beta}(u)   ( >= 0 is exactly H )

the theorem is equivalent to the identity

    RHS(C) - LHS(C)  ==  (1 - beta) * sum_{t=1..T} S_t(u_t)  +  beta * S_T(u_T).

Both coefficients, `(1-beta)` and `beta`, are non-negative *precisely* on the stated domain
`beta in [0,1]`. So the identity plus `S_t >= 0` yields the inequality. Two facts are
verified **separately**, because they are different kinds of fact:

- the **identity** is true regardless of whether H holds (pure algebra);
- **H** is what turns the identity into the bound.

## Routes

| route | method | coverage |
|---|---|---|
| A | exact symbolic simplification, free symbols, no numeric substitution | every `T` in 1..20, exhaustively |
| B | coefficient matching over 5 symbol classes with `T`, `s`, `k` free integer symbols | **all** `T >= 1` |
| C | 4000 instances in exact rational arithmetic (`fractions.Fraction`, no floating point) | guards against mis-transcription of the statement |

Route B is what makes this a proof rather than a bounded check: the identity is linear in
its free symbols, the symbol classes are enumerated exhaustively, and each closed-form
coefficient identity is verified with `T`, `s`, `k` symbolic.

Route C's worst margin is **exactly 0**, attained. The bound is therefore *tight*, not
merely true — strong evidence the statement was transcribed faithfully.

## Negative controls

Six, each damaging one term of the certificate; every one must break, and does, at the
smallest horizon where the damaged term first matters (`T = 1` or `T = 2`). One control,
`hypothesis_only_at_T`, assumes H at `t = T` only and produces an explicit rational
counterexample — confirming the proof genuinely consumes H at every round.

## Verdict rule

VERIFIED requires routes A, B and C to agree and all six controls to break. Anything less
is BLOCKED.

## Known limitation

None material. The residual risk is mis-transcription of the theorem from the paper, which
route C is specifically designed to catch, and the observation that the reduction never
uses `Lambda_s >= 0` or `phi_t >= 0` is recorded as a note rather than a defect.
