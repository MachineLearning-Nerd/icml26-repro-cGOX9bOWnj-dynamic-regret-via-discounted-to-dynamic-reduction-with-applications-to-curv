# Source audit

## Retrieval

| Field | Value |
| --- | --- |
| Paper | Dynamic Regret via Discounted-to-Dynamic Reduction with Applications to Curved Losses and Adam Optimizer |
| OpenReview | cGOX9bOWnj |
| arXiv | 2602.08372 |
| Primary source | `https://ar5iv.labs.arxiv.org/html/2602.08372` |
| Retrieved (UTC) | 2026-07-26T15:40:21Z |
| User-Agent | `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36` |
| SHA-256 (ar5iv HTML) | `a9843091cac480a52e744171eb378e75632afeda60140fcddd518da41b057fc3` |
| Secondary source | `https://arxiv.org/abs/2602.08372` |
| SHA-256 (abs HTML) | `5cae6c5b21d653e9aa8a72d47f885debec2df4da34445bddbdb209118bacebfe` |

## Setting and notation

At each round `t in [T]` the learner picks `x_t in X ⊆ R^d`, then suffers convex
`f_t(x_t)`. Dynamic regret against a comparator sequence is

    D-Reg_T(u_{1:T}) = sum_{t=1..T} f_t(x_t) - sum_{t=1..T} f_t(u_t).

Discounted regret is `Reg_{t;beta}(u) = sum_{s=1..t} beta^{t-s} (f_s(x_s) - f_s(u))`.

Comparator variation is measured by

    P_T^beta = sum_{t=1..T-1} sum_{s=0..t} p^beta_{t,s} [ f_s(u_{t+1}) - f_s(u_t) ]_+,
    p^beta_{t,s} = beta^{t-s} / sum_{tau=0..t} beta^{t-tau},   f_0(u) := phi(u).

Note `P_T^beta` is measured in **loss** differences, not in decision-space
distance; it is not the Zinkevich path length `P_T = sum ||u_{t+1} - u_t||`.
Conflating the two is the easiest way to test the wrong quantity, so every
verifier in this repository computes `P_T^beta` from the definition above.

## Claim-by-claim anchors and exact quantifiers

### Claim 1 - Theorem 1 (Section 2.2), proof Appendix B.2

Hypothesis, for **every** `t in [T]` and **every** `u in X`:

    sum_{s=1..t} beta^{-s} f_s(x_s) - sum_{s=1..t} beta^{-s} f_s(u) <= phi_t(u) + sum_{s=1..t} Lambda_s

with `beta in (0,1]`, `Lambda_s >= 0`, `phi_t(.) >= 0`. Conclusion, for **every**
`u_1..u_T in X`:

    D-Reg_T <= beta*phi_1(u_1) + sum_t beta^t Lambda_t
             + beta*sum_{t=1..T-1} (F_t(u_{t+1}) - F_t(u_t))
             + beta*sum_{t=1..T-1} beta^t (phi_{t+1}(u_{t+1}) - phi_t(u_{t+1})),
    F_t(u) = beta^t phi_t(u) + sum_{s=1..t} beta^{t-s} f_s(u).

**Type: algebraic implication.** No probability, no algorithm, no data. The
quantification "for every t" in the hypothesis is load-bearing (see the
`hypothesis_only_at_T` negative control).

### Claim 2 - Theorem 2 (Section 3.1), extended version + proof Appendix C.1

Algorithm: **discounted VAW forecaster**, `X = R^d` (unconstrained),
`f_t(x) = 1/2 (x^T z_t - y_t)^2`,

    x_t = argmin_x { (lambda beta^t / 2)||x||^2 + 1/2 (x^T z_t)^2
                     + 1/2 sum_{s=1..t-1} beta^{t-s} (x^T z_s - y_s)^2 }.

Explicit bound, for `beta in (0,1)`, `lambda > 0`, and **any** `u_1..u_T in R^d`:

    D-Reg_T <= (beta*lambda/2)||u_1||^2
             + (d/2)(max_t y_t^2) ln(1 + (sum_t beta^{T-t}||z_t||^2)/(lambda d))
             + (beta/(1-beta)) P_T^beta
             + ((1-beta)/beta) (d/2) sum_t y_t^2.

The appendix's extended version replaces the third term by `gamma/(1-gamma) P_T^gamma`
for any `0 < beta <= gamma < 1`; the main-text form is the case `gamma = beta`.
A separate, **asymptotic** sentence claims a two-layer ensemble achieving
`O(d log T + sqrt(d T P_T^{beta*}))`. These are two different claims and are
reported separately.

Derivation chain (each link independently checkable):
L1 Lemma 2 applied to the rescaled sequence `z~_t = beta^{-t/2} z_t`, `y~_t = beta^{-t/2} y_t`
   gives hypothesis (H) with `phi_t(u) = (lambda/2)||u||^2` and
   `Lambda_t = beta^{-2t} y_t^2 z_t^T (lambda I + sum_{s<=t} beta^{-s} z_s z_s^T)^{-1} z_t`;
L2 Theorem 1 (Claim 1, already proved);
L3 Lemma 25 (log-determinant bound) on `sum_t y_t^2 z_t^T (lambda beta^t I + sum_{s<=t} beta^{t-s} z_s z_s^T)^{-1} z_t`;
L4 `ln(1/beta) <= (1-beta)/beta`;
L5 Lemma 18: `beta sum_{t<T} (F_t(u_{t+1}) - F_t(u_t)) <= gamma/(1-gamma) P_T^gamma`.

### Claim 3 - Theorems 3 and 4 (Section 3.2), proofs Appendix C.2-C.3

Logistic loss `l(y^, y) = ln(1 + exp(-y y^))`, labels in `{+1,-1}`,
`||z_t|| <= R`, comparators `||u_t|| <= B`, decisions unconstrained in `R^d`.

**Discounted AIOLI**:

    x_t = argmin_x { (beta^t lambda / 2)||x||^2 + h_t(x) + sum_{s=1..t-1} beta^{t-s} fhat_s(x) },
    h_t(x) = l(x^T z_t, +1) + l(x^T z_t, -1),
    fhat_t(x) = f_t(x_t) + <grad f_t(x_t), x - x_t> + (eta_t/2) <grad f_t(x_t), x - x_t>^2,
    eta_t = exp(y_t yhat_t) / (1 + B R).

Theorem 3 (explicit), for any `u_1..u_T` with `||u_t|| <= B`:

    D-Reg_T <= beta*lambda*||u_1||^2
             + d(1+BR) log(1 + R^2 (sum_t beta^{T-t}) / (d lambda (1+BR)))
             + (beta/(1-beta)) P_T^beta
             + ((1-beta)/beta) d(1+BR) T.

Theorem 4 (asymptotic), Algorithm 1 two-layer ensemble over a geometric grid of
`beta_i = eta_i/(1+eta_i)`, `eta_i = 2^{i-1} eta_min`,
`eta_min = sqrt(d(1+BR)/(CB))`, `eta_max = dT`, `C = max{1, 2R}`, `lambda = 1/B^2`:

    static-vs-dynamic regret <= O( d B log(BT) + sqrt(d B T P_T^{beta*}) ),
    beta* = sqrt(d(1+BR)T) / ( sqrt(d(1+BR)T) + sqrt(P_T^{beta*}) ).

The "avoids exponential dependence on B" part is a comparative claim against the
`O(d e^B log T)` that proper ONS incurs.

### Claim 4 - Theorem 5 (Section 4.3.1), proof Appendix D.6

Assumptions 1-4: `F` differentiable and `G`-Lipschitz; `F(x_0) - inf F <= F*`;
the gradient-integral identity; stochastic gradients unbiased, variance `<= sigma^2`,
almost surely bounded by `G`.

Algorithm 2 (exponentiated O2NC) with `beta = beta_1`, `l_t(Delta) = <g_t, Delta>`,
`D = {||Delta|| <= D}`, and `eta_t` of Eq. (7), giving clipped Adam Eq. (8):

    Delta_{t+1} = Clip_D[ -gamma (1-beta_1) sum_{s<=t} beta_1^{t-s} g_s
                          / ( nu + sqrt((1-beta_2) sum_{s<=t} beta_2^{t-s} ||g_s||^2) ) ].

Parameter settings: `1 - (eps/(16(G+sigma)))^2 <= beta_1 < 1`,
`D = (1-beta_1) sqrt(eps) / sqrt(48 c)`, `gamma = beta_1 D / sqrt(1-beta_1)`,
`0 < nu <= G+sigma`, and the **relaxed condition**

    max{ 1 - nu/(G+sigma), beta_1^4 } <= beta_2 < 1.

If `T >= max{ (1/(1-beta_1)) max{ 16 F* sqrt(48c) / eps^{3/2}, 16(G+sigma)/eps }, ln2/(1-beta_2) }`
then `E[ ||grad F(xbar)||_c ] <= eps`.

The novelty is that the prior literature requires `beta_2 >= beta_1^2`; taking
`nu = G+sigma` makes the condition exactly `beta_2 >= beta_1^4`, which admits the
whole band `beta_1^4 <= beta_2 < beta_1^2` that the prior condition excludes.
That band is where the claim must be tested.

`||grad F(x)||_c = inf over distributions P with mean x of ||E grad F(y)|| + c E||y-x||^2`
is an infimum, so `||grad F(x)||_c <= ||grad F(x)||_2` (take `P = delta_x`).
For **verifying** an upper-bound conclusion, bounding `E||grad F(xbar)||_2 <= eps`
suffices and is computable; that is the route used. It is a conservative test:
it can only make the theorem look worse, never better.

### Claim 5 - Theorem 7 (Section 4.4), proof Appendix D.8

Clip-free Adam: `l_t(Delta) = <g_t, Delta> + (mu/2)||Delta||^2`, `D = R^d`, giving

    Delta_{t+1} = - gamma (1-beta_1) sum_{s<=t} beta_1^{t-s} g_s
                  / ( nu + gamma mu (1 - beta_1^t)
                      + sqrt((1-beta_2) sum_{s<=t} beta_2^{t-s} ||g_s||^2) ).

Settings: `1 - (eps/(16(G+sigma)))^2 <= beta_1 < 1`,
`D = (1-beta_1) sqrt(eps)/sqrt(96 c)`, `gamma = beta_1 D/sqrt(1-beta_1)`,
`0 < nu <= G+sigma`, and `max{1 - nu/(G+sigma), beta_1^2} <= beta_2 < 1` -- i.e.
back to the **standard** `beta_2 >= beta_1^2`, which is the point of the theorem.
If `T >= max{ (1/(1-beta_1)) max{ 16 F* sqrt(96c)/eps^{3/2}, 48(G+sigma)/eps }, ln2/(1-beta_2) }`
then `E[||grad F(xbar)||_c] <= eps`.

Theorem 7 as printed does not fix `mu`; the companion Theorem 8 (same algorithm,
margin-style condition) sets `mu = 24 c D / (1-beta_1)^2`. That value is used and
the deviation is recorded in the limitations section of the claim's method note.

## Shared complexity statement (Appendix D.2 discussion)

Theorems 5-8 are stated to attain `O(max{c^{1/2} eps^{-7/2}, eps^{-3}}) = O(c^{1/2} eps^{-7/2})`
iterations to a `(c, eps)`-stationary point, matching the `Omega(F* G^2 c^{1/2} eps^{-7/2})`
lower bound of Zhang and Cutkosky (2024). This is an **algebraic property of the
stated T formula**, and is checked as such, separately from the convergence
guarantee itself. Measuring an empirical minimum-T and fitting a slope would not
test the theorem: the theorem asserts sufficiency of its own T, not tightness.
