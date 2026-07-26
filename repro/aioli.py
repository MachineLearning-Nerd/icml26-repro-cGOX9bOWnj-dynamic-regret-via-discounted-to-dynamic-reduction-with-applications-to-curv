"""Discounted AIOLI for online logistic regression, and the Algorithm 1 ensemble.

Implements Section 3.2 and Appendix C.2 exactly:

    x_t in argmin_x { (beta^t lambda / 2)||x||^2 + h_t(x) + sum_{s<t} beta^{t-s} fhat_s(x) }

with the OPTIMISM term h_t(x) = l(x^T z_t, +1) + l(x^T z_t, -1) using the observed
feature z_t, and the surrogate

    fhat_s(x) = f_s(x_s) + <g_s, x - x_s> + (eta_s/2) <g_s, x - x_s>^2,
    eta_s = exp(y_s x_s^T z_s) / (1 + B R).

This is an IMPLICIT update: h_t is not quadratic, so x_t has no closed form. The
rejected baseline replaced the whole thing with "simple logistic SGD as AIOLI
proxy", which shares neither the optimism term, nor the second-order surrogate,
nor the discounting -- i.e. none of the mechanisms the theorem is about.

Numerical form
--------------
Everything is kept in beta^t-scaled coordinates, as for VAW, because the paper's
A_t = lambda I + sum_s beta^{-s} eta_s g_s g_s^T overflows for any useful horizon.
Writing P_t = sum_{s<=t} beta^{t-s} eta_s g_s g_s^T (recursion P_t = beta P_{t-1}
+ eta_t g_t g_t^T), the scaled matrix is

    Atil_s = lambda beta^s I + P_s,       so   beta^t A_s^{-1} scales to beta^{t-s} Atil_s^{-1},

and the scaled stability term of Lemma 3 is

    lam_s = (1 + B R) * eta_s * g_s^T Atil_s^{-1} g_s.

The objective splits as a quadratic in x plus a convex function of the scalar
x^T z_t, so Newton's method converges in a handful of iterations per round.
"""

from __future__ import annotations

import numpy as np


def logistic_loss(v: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    """l(v, y) = ln(1 + exp(-y v)), computed stably."""
    z = -np.asarray(y) * np.asarray(v)
    return np.logaddexp(0.0, z)


def _h(v: float) -> float:
    """h(v) = l(v,+1) + l(v,-1) = ln(1+e^{-v}) + ln(1+e^{v})."""
    return float(np.logaddexp(0.0, -v) + np.logaddexp(0.0, v))


def _dh(v: float) -> float:
    """h'(v) = tanh(v/2)."""
    return float(np.tanh(v / 2.0))


def _d2h(v: float) -> float:
    """h''(v) = (1/2) sech^2(v/2) = (1 - tanh^2(v/2)) / 2."""
    th = np.tanh(v / 2.0)
    return float((1.0 - th * th) / 2.0)


def discounted_aioli(
    Z: np.ndarray, y: np.ndarray, beta: float, lam: float, B: float, R: float,
    newton_iters: int = 200, tol: float = 1e-11,
) -> dict[str, np.ndarray]:
    """Run discounted AIOLI. Returns iterates, gradients, eta's and scaled stability terms."""
    T, d = Z.shape
    P = np.zeros((d, d))     # sum_{s<=t} beta^{t-s} eta_s g_s g_s^T
    q = np.zeros(d)          # sum_{s<=t} beta^{t-s} (1 - eta_s <g_s, x_s>) g_s
    eye = np.eye(d)
    X = np.empty((T, d))
    G = np.empty((T, d))
    etas = np.empty(T)
    lam_terms = np.empty(T)
    newton_used = np.empty(T, dtype=int)
    newton_gnorm = np.empty(T)
    newton_scale = np.empty(T)

    for t in range(T):
        bt = beta ** (t + 1)
        # quadratic part: (1/2) x^T M x + b^T x  (constants dropped)
        M = lam * bt * eye + beta * P
        b = beta * q
        z = Z[t]

        # DAMPED Newton with backtracking on  (1/2)x^T M x + b^T x + h(x^T z).
        # M = lambda beta^t I + beta P is positive definite, so the minimiser
        # always exists -- but lambda beta^t decays geometrically, so by moderate
        # T the problem is extremely ill-conditioned and undamped Newton
        # overshoots and diverges. Convergence is verified per round rather than
        # assumed: an unconverged solve is a wrong x_t, and a wrong x_t silently
        # invalidates every downstream number.
        def obj(xx: np.ndarray) -> float:
            return float(0.5 * xx @ (M @ xx) + b @ xx + _h(float(xx @ z)))

        x = np.linalg.solve(M, -b)
        used = 0
        gnorm = np.inf
        for it in range(newton_iters):
            v = float(x @ z)
            grad = M @ x + b + _dh(v) * z
            gnorm = float(np.linalg.norm(grad))
            scale = max(1.0, float(np.linalg.norm(M @ x)), float(np.linalg.norm(b)))
            if gnorm <= tol * scale:
                break
            H = M + _d2h(v) * np.outer(z, z)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            f0 = obj(x)
            alpha = 1.0
            for _ in range(60):  # backtracking line search
                xn = x - alpha * step
                if obj(xn) <= f0 - 1e-4 * alpha * float(grad @ step):
                    break
                alpha *= 0.5
            x = x - alpha * step
            used = it + 1
        newton_used[t] = used
        newton_gnorm[t] = gnorm
        newton_scale[t] = max(1.0, float(np.linalg.norm(M @ x)), float(np.linalg.norm(b)))

        X[t] = x
        yhat = float(x @ z)
        # g_t = grad f_t(x_t) = -y_t sigma(-y_t x^T z) z
        m = -y[t] * yhat
        sig = 1.0 / (1.0 + np.exp(-m)) if m < 500 else 1.0
        g = -y[t] * sig * z
        G[t] = g
        eta = float(np.exp(np.clip(y[t] * yhat, -700, 700)) / (1.0 + B * R))
        etas[t] = eta

        # scaled stability term of Lemma 3 (see module docstring)
        Ptil = beta * P + eta * np.outer(g, g)      # = P_t
        Atil = lam * bt * eye + Ptil
        lam_terms[t] = (1.0 + B * R) * eta * float(g @ np.linalg.solve(Atil, g))

        P = Ptil
        q = beta * q + (1.0 - eta * float(g @ x)) * g

    rel_gnorm = newton_gnorm / np.maximum(newton_scale, 1e-300)
    return {"X": X, "G": G, "etas": etas, "lam_terms": lam_terms,
            "newton_iters": newton_used, "newton_rel_gradnorm": rel_gnorm,
            "solver_converged": bool(np.all(rel_gnorm <= 1e-6)),
            "worst_rel_gradnorm": float(rel_gnorm.max())}


# --------------------------------------------------------------------------
# regret, comparator variation, and the Theorem 3 bound
# --------------------------------------------------------------------------
def losses(X: np.ndarray, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(logistic_loss(np.einsum("td,td->t", X, Z), y))


def dynamic_regret(X: np.ndarray, U: np.ndarray, Z: np.ndarray, y: np.ndarray) -> float:
    return float(losses(X, Z, y).sum() - losses(U, Z, y).sum())


def loss_matrix(U: np.ndarray, Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """L[s, k] = f_s(u_{k+1}); row 0 is f_0(u) = (lambda/2)||u||^2 per Theorem 4."""
    T = Z.shape[0]
    L = np.empty((T + 1, T))
    L[0] = 0.5 * lam * np.einsum("kd,kd->k", U, U)
    L[1:] = np.asarray(logistic_loss(U @ Z.T, y[None, :])).T
    return L


def P_T_beta(U: np.ndarray, Z: np.ndarray, y: np.ndarray, beta: float, lam: float) -> float:
    """Eq. (4): comparator variation in LOSS differences, not decision distance."""
    T = Z.shape[0]
    if T < 2:
        return 0.0
    L = loss_matrix(U, Z, y, lam)
    total = 0.0
    for t in range(1, T):
        diff = L[0 : t + 1, t] - L[0 : t + 1, t - 1]
        np.maximum(diff, 0.0, out=diff)
        w = beta ** (t - np.arange(t + 1))
        total += float(diff @ w) / float(w.sum())
    return total


def theorem3_rhs(U, Z, y, beta: float, lam: float, B: float, R: float,
                 perturb: str | None = None) -> dict[str, float]:
    """The four terms of Theorem 3's explicit dynamic regret bound."""
    T, d = Z.shape
    t1 = beta * lam * float(U[0] @ U[0])
    geo = float(np.sum(beta ** (T - 1 - np.arange(T))))
    t2 = d * (1.0 + B * R) * np.log(1.0 + (R**2) * geo / (d * lam * (1.0 + B * R)))
    P = P_T_beta(U, Z, y, beta, lam)
    t3 = beta / (1.0 - beta) * P
    t4 = (1.0 - beta) / beta * d * (1.0 + B * R) * T

    if perturb == "drop_last_term":
        t4 = 0.0
    elif perturb == "drop_path_term":
        t3 = 0.0
    elif perturb == "drop_log_term":
        t2 = 0.0
    elif perturb == "halve_log_term":
        t2 *= 0.5
    elif perturb == "zinkevich_path":
        t3 = beta / (1.0 - beta) * float(np.linalg.norm(np.diff(U, axis=0), axis=1).sum()) if T > 1 else 0.0
    elif perturb == "drop_B_dependence":
        # replace (1+BR) by 1 everywhere: tests that the B-dependence is real
        t2 = d * np.log(1.0 + (R**2) * geo / (d * lam))
        t4 = (1.0 - beta) / beta * d * T
    elif perturb is not None:
        raise ValueError(perturb)

    return {"term1": t1, "term2": t2, "term3": t3, "term4": t4, "P_T_beta": P,
            "rhs": t1 + t2 + t3 + t4}


def derivation_links(run: dict[str, np.ndarray], U, Z, y, beta: float, lam: float,
                     B: float, R: float) -> dict[str, float]:
    """Check Lemma 3 (the Theorem 1 hypothesis) and the Theorem 1 application."""
    T, d = Z.shape
    X, lam_terms = run["X"], run["lam_terms"]
    f_alg = losses(X, Z, y)
    L = loss_matrix(U, Z, y, lam)

    # L1: Lemma 3 in beta^t-scaled form, at every t against comparator u_t
    l1_min = np.inf
    for t in range(1, T + 1):
        k = t - 1
        w = beta ** (t - np.arange(1, t + 1))
        reg = float(w @ (f_alg[:t] - L[1 : t + 1, k]))
        bound = (beta**t) * 0.5 * lam * float(U[k] @ U[k]) + float(w @ lam_terms[:t])
        l1_min = min(l1_min, bound - reg)

    # L2: Theorem 1 applied -> intermediate bound
    Lm = L
    path_raw = 0.0
    if T > 1:
        for t in range(1, T):
            w = beta ** (t - np.arange(1, t + 1))
            F_next = (beta**t) * 0.5 * lam * float(U[t] @ U[t]) + float(Lm[1 : t + 1, t] @ w)
            F_cur = (beta**t) * 0.5 * lam * float(U[t - 1] @ U[t - 1]) + float(Lm[1 : t + 1, t - 1] @ w)
            path_raw += F_next - F_cur
        path_raw *= beta
    dreg = dynamic_regret(X, U, Z, y)
    eq_mid = beta * 0.5 * lam * float(U[0] @ U[0]) + float(lam_terms.sum()) + path_raw

    # L3: the log-determinant step for AIOLI's stability sum
    geo = float(np.sum(beta ** (T - 1 - np.arange(T))))
    logdet_rhs = d * (1.0 + B * R) * np.log(1.0 + (R**2) * geo / (d * lam * (1.0 + B * R))) + \
        (1.0 - beta) / beta * d * (1.0 + B * R) * T
    return {
        "L1_lemma3_min_slack": float(l1_min),
        "L2_theorem1_applied_slack": float(eq_mid - dreg),
        "L3_logdet_slack": float(logdet_rhs - float(lam_terms.sum())),
        "sum_lam_terms": float(lam_terms.sum()),
        "dynamic_regret": float(dreg),
        "max_newton_iters": int(run["newton_iters"].max()),
        "solver_converged": bool(run["solver_converged"]),
        "worst_rel_gradnorm": float(run["worst_rel_gradnorm"]),
    }


# --------------------------------------------------------------------------
# ONS reference: the e^B baseline the claim says AIOLI avoids
# --------------------------------------------------------------------------
def ons_logistic(Z: np.ndarray, y: np.ndarray, B: float, R: float, gamma: float | None = None) -> np.ndarray:
    """Online Newton Step on the ball ||x|| <= B.

    Its exp-concavity constant on that ball degrades like e^{BR}, which is the
    source of the O(d e^B log T) static regret the paper contrasts against. It is
    included so the 'avoids exponential dependence on B' claim is measured against
    a real alternative rather than asserted.
    """
    T, d = Z.shape
    # exp-concavity parameter of the logistic loss on ||x||<=B, ||z||<=R
    alpha = np.exp(-B * R) / 2.0
    gamma = 0.5 * min(1.0 / (4.0 * max(R, 1e-12) * B if B > 0 else 1.0), alpha) if gamma is None else gamma
    eps = 1.0 / (gamma**2 * max(B, 1e-9) ** 2) if B > 0 else 1.0
    A = eps * np.eye(d)
    x = np.zeros(d)
    X = np.empty((T, d))
    for t in range(T):
        X[t] = x
        v = float(x @ Z[t])
        m = -y[t] * v
        sig = 1.0 / (1.0 + np.exp(-m)) if m < 500 else 1.0
        g = -y[t] * sig * Z[t]
        A = A + np.outer(g, g)
        x = x - (1.0 / gamma) * np.linalg.solve(A, g)
        n = np.linalg.norm(x)
        if n > B:  # projection onto the ball (the A-norm projection is approximated by Euclidean)
            x = x * (B / n)
    return X
