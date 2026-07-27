"""Algorithm 2 (exponentiated O2NC) with clipped and clip-free Adam.

Implements Section 4 of the paper exactly:
  * Algorithm 2, including the Exp(1) step scaling on line 4 and the EMA output
    on line 7 -- both are load-bearing, not cosmetic;
  * clipped Adam, Eq. (8), from l_t(Delta) = <g_t, Delta> on D = {||Delta|| <= D};
  * clip-free Adam, Eq. (10), from l_t(Delta) = <g_t, Delta> + (mu/2)||Delta||^2
    on D = R^d, with the extra gamma*mu*(1 - beta_1^t) damping in the denominator.

Objectives satisfy Assumptions 1-4 by construction, and `audit_assumptions`
re-checks them numerically rather than taking the construction on trust.

Measuring the conclusion
------------------------
The theorems conclude E[ ||grad F(xbar)||_c ] <= eps, where

    ||grad F(x)||_c = inf_{P: E[y]=x} || E[grad F(y)] || + c E||y - x||^2

is an infimum over distributions, hence not directly computable. But any single
distribution P gives an UPPER bound, and an upper bound is exactly what is needed
to verify an upper-bound conclusion. We use isotropic Gaussians P = N(x, s^2 I),
for which E||y-x||^2 = s^2 d, and minimise over a grid of s that includes s = 0
(recovering the trivial bound ||grad F(x)||_2):

    ||grad F(x)||_c <= min_s ( || E_{y~N(x,s^2 I)} grad F(y) || + c s^2 d ).

The expectation inside is estimated by Monte Carlo. Note the direction of the
error is safe: E||sample mean|| >= ||true mean|| by Jensen, so the estimate is
biased UPWARD, making the test conservative. It can only make the theorem look
worse than it is, never better.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# objectives satisfying Assumptions 1-4
# --------------------------------------------------------------------------
class Objective:
    """F with ||grad F|| <= G_F everywhere, bounded below, differentiable.

    The stochastic gradient is grad F(x) + xi with xi uniform on the sphere of
    radius sigma, so it is unbiased, has variance exactly sigma^2, and is almost
    surely bounded by G_F + sigma. Declaring G = G_F + sigma therefore satisfies
    Assumption 4's ||grad f(x;xi)|| <= G exactly, not approximately.
    """

    def __init__(self, kind: str, d: int, sigma: float, seed: int = 0):
        self.kind = kind
        self.d = d
        self.sigma = sigma
        rng = np.random.default_rng(seed + 5150)
        if kind == "two_well":
            # Sum of two inverted Gaussian wells: nonconvex, two local minima,
            # gradient bounded, bounded below. The width w matters: with w too
            # small the function is numerically FLAT a few units out, every far
            # start point is already stationary, and the whole test goes vacuous.
            self.w = 2.5
            self.m1 = -2.0 * np.ones(d) / np.sqrt(d)
            self.m2 = 2.0 * np.ones(d) / np.sqrt(d)
            self.A1, self.A2 = 3.0, 2.0
            self.G_F = float(np.sqrt(d) * (self.A1 + self.A2) * np.exp(-0.5) / self.w)
        elif kind == "rastrigin_lipschitz":
            # highly nonconvex with many stationary points; gradient a*sin(w x)
            self.a, self.w = 0.5, 2.0
            self.G_F = float(self.a * np.sqrt(d))
        elif kind == "sharp_cone":
            # differentiable but with unbounded curvature as delta -> 0: the
            # 'nonsmooth' regime the paper targets, while staying within
            # Assumption 1 (F must be differentiable)
            self.delta = 1e-3
            self.G_F = 1.0
        elif kind == "asym_valley":
            # nonconvex, non-separable, asymmetric; gradient bounded via tanh.
            # grad F(x) = h(Qx)^T Q with h(z) = tanh(z) - 0.3(1 - tanh^2 z), so
            # ||grad F|| <= ||Q||_op * sqrt(d) * max_z |h(z)|. Both factors are
            # computed rather than guessed -- an eyeballed Lipschitz constant
            # here silently breaks Assumption 1, which is what the audit caught.
            self.Q = rng.standard_normal((d, d)) / np.sqrt(d)
            zs = np.linspace(-30, 30, 600001)
            h_max = float(np.max(np.abs(np.tanh(zs) - 0.3 * (1 - np.tanh(zs) ** 2))))
            self.G_F = float(np.linalg.norm(self.Q, 2) * np.sqrt(d) * h_max)
        else:
            raise ValueError(kind)
        self.G = self.G_F + sigma

    # -- F and grad F --------------------------------------------------------
    def F(self, x: np.ndarray) -> float:
        k = self.kind
        if k == "two_well":
            return float(
                self.A1 + self.A2
                - self.A1 * np.exp(-0.5 * np.sum((x - self.m1) ** 2) / self.w**2)
                - self.A2 * np.exp(-0.5 * np.sum((x - self.m2) ** 2) / self.w**2)
            )
        if k == "rastrigin_lipschitz":
            return float(np.sum(self.a * (1 - np.cos(self.w * x)) / self.w))
        if k == "sharp_cone":
            return float(np.sqrt(np.sum(x**2) + self.delta**2) - self.delta)
        if k == "asym_valley":
            z = self.Q @ x
            return float(np.sum(np.log(np.cosh(z)) - 0.3 * np.tanh(z)))
        raise ValueError(k)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """grad F, vectorised over a leading batch axis if present."""
        k = self.kind
        single = x.ndim == 1
        X = x[None, :] if single else x
        if k == "two_well":
            w2 = self.w**2
            e1 = np.exp(-0.5 * np.sum((X - self.m1) ** 2, axis=1) / w2)[:, None]
            e2 = np.exp(-0.5 * np.sum((X - self.m2) ** 2, axis=1) / w2)[:, None]
            g = (self.A1 * e1 * (X - self.m1) + self.A2 * e2 * (X - self.m2)) / w2
        elif k == "rastrigin_lipschitz":
            g = self.a * np.sin(self.w * X)
        elif k == "sharp_cone":
            g = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True) + self.delta**2)
        elif k == "asym_valley":
            Z = X @ self.Q.T
            g = (np.tanh(Z) - 0.3 * (1 - np.tanh(Z) ** 2)) @ self.Q
        else:
            raise ValueError(k)
        return g[0] if single else g

    def stoch_grad(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """grad F(x) + xi, xi uniform on the sphere of radius sigma.

        Vectorised over a leading batch axis: given x of shape (S, d) this draws
        S independent noise vectors, one per row, so S independent runs of the
        algorithm can share a single Python-level loop iteration.
        """
        single = x.ndim == 1
        X = x[None, :] if single else x
        xi = rng.standard_normal(X.shape)
        n = np.linalg.norm(xi, axis=1, keepdims=True)
        xi = np.divide(xi, n, out=np.zeros_like(xi), where=n > 0) * self.sigma
        out = self.grad(X) + xi
        return out[0] if single else out

    def F_star(self, x0: np.ndarray) -> float:
        """A valid F* = F(x0) - inf F (Assumption 2), computed exactly."""
        k = self.kind
        if k == "two_well":
            inf_F = 0.0  # attained in the limit only when the bumps coincide;
            # a valid lower bound: F >= A1 + A2 - A1 - A2 = 0
        elif k == "rastrigin_lipschitz":
            inf_F = 0.0
        elif k == "sharp_cone":
            inf_F = 0.0
        elif k == "asym_valley":
            # log cosh(z) - 0.3 tanh(z) >= min over z, per coordinate
            zs = np.linspace(-20, 20, 400001)
            inf_F = float(self.d * np.min(np.log(np.cosh(zs)) - 0.3 * np.tanh(zs)))
        else:
            raise ValueError(k)
        return float(self.F(x0) - inf_F)


def audit_assumptions(obj: Objective, n: int = 40000, seed: int = 11) -> dict[str, float | bool]:
    """Numerically re-check Assumptions 1 and 4 instead of trusting construction."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, obj.d)) * 4.0
    Gn = np.linalg.norm(obj.grad(X), axis=1)
    # unbiasedness and variance of the stochastic gradient at a few points
    biases, variances, sup_norm = [], [], 0.0
    for i in range(6):
        x = X[i]
        S = np.stack([obj.stoch_grad(x, rng) for _ in range(4000)])
        biases.append(float(np.linalg.norm(S.mean(axis=0) - obj.grad(x))))
        variances.append(float(np.mean(np.sum((S - obj.grad(x)) ** 2, axis=1))))
        sup_norm = max(sup_norm, float(np.max(np.linalg.norm(S, axis=1))))
    return {
        "A1_max_grad_norm_observed": float(Gn.max()),
        "A1_declared_G_F": obj.G_F,
        "A1_holds": bool(Gn.max() <= obj.G_F + 1e-9),
        "A4_max_bias_norm": float(max(biases)),
        "A4_unbiased_within_mc_error": bool(max(biases) < 0.05 * obj.sigma + 1e-12),
        "A4_max_variance_observed": float(max(variances)),
        "A4_declared_sigma_sq": obj.sigma**2,
        "A4_variance_holds": bool(max(variances) <= obj.sigma**2 * (1 + 1e-6)),
        "A4_max_stoch_grad_norm": sup_norm,
        "A4_declared_G": obj.G,
        "A4_bounded_holds": bool(sup_norm <= obj.G + 1e-9),
    }


# --------------------------------------------------------------------------
# Algorithm 2 with clipped / clip-free Adam
# --------------------------------------------------------------------------
def run_o2nc(
    obj: Objective,
    x0: np.ndarray,
    T: int,
    beta1: float,
    beta2: float,
    gamma: float,
    nu: float,
    D: float,
    variant: str,
    mu: float = 0.0,
    seed: int = 0,
    n_snapshots: int = 48,
) -> dict[str, np.ndarray]:
    """Run Algorithm 2 for T rounds; return snapshots of the EMA iterate xbar_t.

    `variant` is 'clipped' (Eq. 8) or 'clipfree' (Eq. 10). Snapshot rounds are
    drawn uniformly WITHOUT replacement, so averaging over them is an unbiased
    estimator of E_{t ~ Unif[T]} of any function of xbar_t -- which is exactly
    the expectation the theorems' output distribution induces (Algorithm 2 line 9).
    """
    rng = np.random.default_rng(seed)
    d = obj.d
    x = x0.astype(float).copy()
    xbar = x0.astype(float).copy()
    m = np.zeros(d)
    v = 0.0
    delta = np.zeros(d)  # Delta_1 = 0 (empty sum in Eq. 5)

    snap_idx = np.sort(rng.choice(np.arange(1, T + 1), size=min(n_snapshots, T), replace=False))
    snaps = np.empty((len(snap_idx), d))
    k = 0
    beta = beta1  # Algorithm 2 is run with beta = beta_1

    for t in range(1, T + 1):
        s_t = rng.exponential(1.0)  # line 4: s_t ~ Exp(1)
        x = x + s_t * delta
        g = obj.stoch_grad(x, rng)

        # line 7: xbar_t = (beta - beta^t)/(1 - beta^t) xbar_{t-1} + (1-beta)/(1-beta^t) x_t
        bt = beta**t
        denom = 1.0 - bt
        if denom <= 1e-300:  # t = 1 with beta -> 1; the update is xbar_1 = x_1
            xbar = x.copy()
        else:
            xbar = ((beta - bt) / denom) * xbar + ((1.0 - beta) / denom) * x

        # Adam state: m_t = sum_s beta_1^{t-s} g_s,  v_t = sum_s beta_2^{t-s}||g_s||^2
        m = beta1 * m + g
        v = beta2 * v + float(g @ g)

        num = gamma * (1.0 - beta1) * m
        root = np.sqrt((1.0 - beta2) * v)
        if variant == "clipped":
            cand = -num / (nu + root)
            nrm = np.linalg.norm(cand)
            delta = cand * (min(nrm, D) / nrm) if nrm > 0 else cand
        elif variant == "clipfree":
            delta = -num / (nu + gamma * mu * (1.0 - beta1**t) + root)
        else:
            raise ValueError(variant)

        if k < len(snap_idx) and t == snap_idx[k]:
            snaps[k] = xbar
            k += 1

    return {"snapshots": snaps[:k], "snapshot_rounds": snap_idx[:k], "x_final": x, "xbar_final": xbar}


def run_o2nc_batch(
    obj: Objective,
    x0: np.ndarray,
    T: int,
    beta1: float,
    beta2: float,
    gamma: float,
    nu: float,
    D: float,
    variant: str,
    n_runs: int,
    mu: float = 0.0,
    seed: int = 0,
    n_snapshots: int = 48,
) -> list[dict[str, np.ndarray]]:
    """Run n_runs INDEPENDENT copies of Algorithm 2 in one Python loop.

    Identical mathematics to run_o2nc, executed with a leading run axis so that
    T iterations of Python overhead are paid once instead of n_runs times. That
    overhead was the whole cost: at T = 3.9M and 86 settings x 8 seeds the
    scalar version projected past eight hours, essentially all of it interpreter
    time on length-2 to length-5 vectors.

    The runs remain independent -- separate noise, separate Exp(1) scalings,
    separate snapshot rounds. What differs from calling run_o2nc n_runs times is
    only WHICH pseudo-random numbers each run receives, since they are now drawn
    from one generator in a different order. That changes no distribution, and
    the run is still fully determined by `seed`.
    """
    rng = np.random.default_rng(seed)
    d, S = obj.d, n_runs
    # x0 may be a single point shared by every run, or one start point per run.
    # Per-run start points matter: _start_point draws a random direction, so
    # collapsing them would quietly average over less than the scalar path did.
    X0 = np.asarray(x0, dtype=float)
    x = np.tile(X0, (S, 1)) if X0.ndim == 1 else X0.copy()
    if x.shape != (S, d):
        raise ValueError(f"x0 must be (d,) or ({S}, {d}), got {X0.shape}")
    xbar = x.copy()
    m = np.zeros((S, d))
    v = np.zeros(S)
    delta = np.zeros((S, d))          # Delta_1 = 0 (empty sum in Eq. 5)

    # Independent snapshot rounds per run, exactly as the scalar version draws
    # them, indexed by round so the inner loop only pays a dict lookup.
    snap_idx = np.stack([np.sort(rng.choice(np.arange(1, T + 1),
                                            size=min(n_snapshots, T), replace=False))
                         for _ in range(S)])
    due: dict[int, list[int]] = {}
    for r in range(S):
        for t_ in snap_idx[r]:
            due.setdefault(int(t_), []).append(r)
    snaps = np.empty((S, snap_idx.shape[1], d))
    filled = np.zeros(S, dtype=int)

    beta = beta1                      # Algorithm 2 is run with beta = beta_1
    g1 = gamma * (1.0 - beta1)
    root_scale = np.sqrt(1.0 - beta2)

    for t in range(1, T + 1):
        s_t = rng.exponential(1.0, size=S)             # line 4: s_t ~ Exp(1)
        x = x + s_t[:, None] * delta
        g = obj.stoch_grad(x, rng)

        # line 7: xbar_t = (beta - beta^t)/(1 - beta^t) xbar_{t-1} + (1-beta)/(1-beta^t) x_t
        bt = beta**t
        denom = 1.0 - bt
        if denom <= 1e-300:           # t = 1 with beta -> 1; the update is xbar_1 = x_1
            xbar = x.copy()
        else:
            xbar = ((beta - bt) / denom) * xbar + ((1.0 - beta) / denom) * x

        m = beta1 * m + g
        v = beta2 * v + np.einsum("sd,sd->s", g, g)

        num = g1 * m
        root = (root_scale * np.sqrt(v))[:, None]
        if variant == "clipped":
            cand = -num / (nu + root)
            nrm = np.linalg.norm(cand, axis=1, keepdims=True)
            scale = np.divide(np.minimum(nrm, D), nrm,
                              out=np.ones_like(nrm), where=nrm > 0)
            delta = cand * scale
        elif variant == "clipfree":
            delta = -num / (nu + gamma * mu * (1.0 - beta1**t) + root)
        else:
            raise ValueError(variant)

        rows = due.get(t)
        if rows is not None:
            for r in rows:
                snaps[r, filled[r]] = xbar[r]
                filled[r] += 1

    return [
        {"snapshots": snaps[r, : filled[r]], "snapshot_rounds": snap_idx[r, : filled[r]],
         "x_final": x[r], "xbar_final": xbar[r]}
        for r in range(S)
    ]


# --------------------------------------------------------------------------
# upper bound on ||grad F(x)||_c
# --------------------------------------------------------------------------
def grad_norm_c_upper(
    obj: Objective, x: np.ndarray, c: float, n_mc: int = 192, seed: int = 0,
    s_grid: np.ndarray | None = None
) -> float:
    """min over s of || E_{y~N(x, s^2 I)} grad F(y) || + c s^2 d  >= ||grad F(x)||_c."""
    rng = np.random.default_rng(seed)
    if s_grid is None:
        s_grid = np.concatenate([[0.0], np.geomspace(1e-3, 3.0, 9)])
    best = np.inf
    for s in s_grid:
        if s == 0.0:
            val = float(np.linalg.norm(obj.grad(x)))
        else:
            Y = x[None, :] + s * rng.standard_normal((n_mc, obj.d))
            val = float(np.linalg.norm(obj.grad(Y).mean(axis=0)) + c * (s**2) * obj.d)
        best = min(best, val)
    return best


# --------------------------------------------------------------------------
# theorem parameter settings
# --------------------------------------------------------------------------
def theorem5_params(eps: float, c: float, G: float, sigma: float, F_star: float,
                    nu: float | None = None, beta2: float | None = None) -> dict[str, float]:
    """Exactly the settings of Theorem 5 (clipped Adam, relaxed beta_2)."""
    beta1 = 1.0 - (eps / (16.0 * (G + sigma))) ** 2
    D = (1.0 - beta1) * np.sqrt(eps) / np.sqrt(48.0 * c)
    gamma = beta1 * D / np.sqrt(1.0 - beta1)
    nu = (G + sigma) if nu is None else nu
    lo = max(1.0 - nu / (G + sigma), beta1**4)  # the RELAXED condition
    beta2 = lo if beta2 is None else beta2
    T = max(
        (1.0 / (1.0 - beta1)) * max(16.0 * F_star * np.sqrt(48.0 * c) / eps**1.5, 16.0 * (G + sigma) / eps),
        np.log(2.0) / (1.0 - beta2),
    )
    return {"beta1": beta1, "beta2": beta2, "beta2_lower_bound": lo, "D": D, "gamma": gamma,
            "nu": nu, "T": float(T), "T_int": int(np.ceil(T)), "mu": 0.0}


def theorem7_params(eps: float, c: float, G: float, sigma: float, F_star: float,
                    nu: float | None = None, beta2: float | None = None) -> dict[str, float]:
    """Exactly the settings of Theorem 7 (clip-free Adam, standard beta_2 >= beta_1^2).

    Theorem 7 as printed leaves mu 'to be tuned'. The companion Theorem 8, same
    algorithm and same D and gamma, fixes mu = 24 c D / (1 - beta_1)^2; that value
    is used here and the deviation is recorded in the claim's limitations.
    """
    beta1 = 1.0 - (eps / (16.0 * (G + sigma))) ** 2
    D = (1.0 - beta1) * np.sqrt(eps) / np.sqrt(96.0 * c)
    gamma = beta1 * D / np.sqrt(1.0 - beta1)
    nu = (G + sigma) if nu is None else nu
    lo = max(1.0 - nu / (G + sigma), beta1**2)  # the STANDARD condition
    beta2 = lo if beta2 is None else beta2
    mu = 24.0 * c * D / (1.0 - beta1) ** 2
    T = max(
        (1.0 / (1.0 - beta1)) * max(16.0 * F_star * np.sqrt(96.0 * c) / eps**1.5, 48.0 * (G + sigma) / eps),
        np.log(2.0) / (1.0 - beta2),
    )
    return {"beta1": beta1, "beta2": beta2, "beta2_lower_bound": lo, "D": D, "gamma": gamma,
            "nu": nu, "T": float(T), "T_int": int(np.ceil(T)), "mu": mu}
