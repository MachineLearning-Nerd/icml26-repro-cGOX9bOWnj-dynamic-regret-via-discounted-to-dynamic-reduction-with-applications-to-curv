"""Discounted Vovk-Azoury-Warmuth forecaster and the Theorem 2 bound.

Numerical note (this is the crux of implementing the paper faithfully)
---------------------------------------------------------------------
The paper's analysis is written in *rescaled* coordinates: z~_t = beta^{-t/2} z_t,
and the stability term is

    Lambda_t = beta^{-2t} y_t^2 z_t^T ( lambda I + sum_{s<=t} beta^{-s} z_s z_s^T )^{-1} z_t.

Evaluating that literally overflows for any interesting horizon: beta^{-2t} with
beta = 0.99 and t = 3000 is about 1e26, and beta^{-t} inside the inverse is worse.
A verifier that computes it directly reports inf/nan and then "passes" vacuously.

Every quantity below is therefore evaluated in the beta^t-scaled coordinates
where the algebra is identical but the magnitudes are O(1). Writing

    A_t = lambda beta^t I + sum_{s<=t} beta^{t-s} z_s z_s^T,

the identity ( lambda I + sum_{s<=t} beta^{-s} z_s z_s^T )^{-1} = beta^s A_s^{-1}
(applied at s = t) gives the scaled stability term

    lam_t := y_t^2 z_t^T A_t^{-1} z_t,      so that   beta^t Lambda_s = beta^{t-s} lam_s.

In particular sum_t beta^t Lambda_t = sum_t lam_t, which is exactly the middle
term of Theorem 2's intermediate bound (Eq. 12). All formulas below use lam_t.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# the algorithm
# --------------------------------------------------------------------------
def discounted_vaw(Z: np.ndarray, y: np.ndarray, beta: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the discounted VAW forecaster of Section 3.1.

        x_t = argmin_x { (lambda beta^t / 2)||x||^2 + (1/2)(x^T z_t)^2
                         + (1/2) sum_{s<t} beta^{t-s} (x^T z_s - y_s)^2 }

    The first-order condition gives x_t = A_t^{-1} (beta * b_{t-1}) with
    A_t = lambda beta^t I + sum_{s<=t} beta^{t-s} z_s z_s^T  (the s=t term being
    z_t z_t^T, i.e. the VAW "look at the feature before predicting" trick) and
    b_t = sum_{s<=t} beta^{t-s} y_s z_s. Both admit the O(d^2) recursions
    S_t = beta S_{t-1} + z_t z_t^T and b_t = beta b_{t-1} + y_t z_t.

    Returns (X, lam_terms) where X[t] = x_{t+1} and lam_terms[t] = lam_{t+1}.
    """
    T, d = Z.shape
    S = np.zeros((d, d))
    b = np.zeros(d)
    X = np.empty((T, d))
    lam_terms = np.empty(T)
    eye = np.eye(d)

    for t in range(T):  # t is 0-based; paper's round index is t+1
        z = Z[t]
        beta_pow = beta ** (t + 1)
        # A_t uses S_{t-1} discounted by beta, plus the current outer product.
        A = lam * beta_pow * eye + beta * S + np.outer(z, z)
        rhs = beta * b
        X[t] = np.linalg.solve(A, rhs)
        # lam_t = y_t^2 z_t^T A_t^{-1} z_t, the scaled stability term.
        lam_terms[t] = (y[t] ** 2) * float(z @ np.linalg.solve(A, z))
        # advance the recursions to include round t
        S = beta * S + np.outer(z, z)
        b = beta * b + y[t] * z
    return X, lam_terms


def sq_loss(pred: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (pred - y) ** 2


def algorithm_losses(X: np.ndarray, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f_t(x_t) for each t."""
    return sq_loss(np.einsum("td,td->t", X, Z), y)


def comparator_losses(U: np.ndarray, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f_t(u_t) for each t (the diagonal), used for dynamic regret."""
    return sq_loss(np.einsum("td,td->t", U, Z), y)


def dynamic_regret(X: np.ndarray, U: np.ndarray, Z: np.ndarray, y: np.ndarray) -> float:
    """TRUE dynamic regret: sum_t f_t(x_t) - sum_t f_t(u_t), comparator-per-round.

    This is deliberately not static regret against a single best fixed vector.
    """
    return float(algorithm_losses(X, Z, y).sum() - comparator_losses(U, Z, y).sum())


# --------------------------------------------------------------------------
# comparator variation P_T^beta  (Eq. 4)
# --------------------------------------------------------------------------
def loss_matrix(U: np.ndarray, Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """L[s, k] = f_s(u_{k+1}) for s = 0..T, k = 0..T-1, with f_0(u) = (lambda/2)||u||^2.

    Row 0 is the analysis quantity phi = f_0; rows 1..T are the real losses.
    """
    T = Z.shape[0]
    preds = U @ Z.T  # preds[k, s] = u_{k+1}^T z_{s+1}
    L = np.empty((T + 1, T))
    L[0] = 0.5 * lam * np.einsum("kd,kd->k", U, U)
    L[1:] = (0.5 * (preds - y[None, :]) ** 2).T  # -> [s, k]
    return L


def P_T_beta(U: np.ndarray, Z: np.ndarray, y: np.ndarray, beta: float, lam: float) -> float:
    """P_T^beta = sum_{t=1..T-1} sum_{s=0..t} p^beta_{t,s} [ f_s(u_{t+1}) - f_s(u_t) ]_+

    with p^beta_{t,s} = beta^{t-s} / sum_{tau=0..t} beta^{t-tau}.

    This is comparator variation measured in LOSS differences. It is NOT the
    Zinkevich path length sum ||u_{t+1} - u_t||; substituting that is a way to
    test the wrong theorem, and is used only as a negative control.
    """
    T = Z.shape[0]
    if T < 2:
        return 0.0
    L = loss_matrix(U, Z, y, lam)
    total = 0.0
    for t in range(1, T):  # paper index t = 1..T-1
        diff = L[0 : t + 1, t] - L[0 : t + 1, t - 1]  # f_s(u_{t+1}) - f_s(u_t), s=0..t
        np.maximum(diff, 0.0, out=diff)
        s = np.arange(t + 1)
        w = beta ** (t - s)
        denom = w.sum()  # = sum_{tau=0..t} beta^{t-tau}
        total += float(diff @ w) / denom
    return total


def zinkevich_path_length(U: np.ndarray) -> float:
    """sum_t ||u_{t+1} - u_t||_2 -- the WRONG quantity, kept for a negative control."""
    if U.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(U, axis=0), axis=1).sum())


# --------------------------------------------------------------------------
# the Theorem 2 bound and its derivation links
# --------------------------------------------------------------------------
def theorem2_rhs(
    U: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    beta: float,
    lam: float,
    perturb: str | None = None,
) -> dict[str, float]:
    """The four terms of Theorem 2's explicit bound (main-text form, gamma = beta)."""
    T, d = Z.shape
    t1 = 0.5 * beta * lam * float(U[0] @ U[0])
    weights = beta ** (T - 1 - np.arange(T))  # beta^{T-t} for t = 1..T
    t2 = 0.5 * d * float(np.max(y**2)) * np.log1p(float(weights @ np.einsum("td,td->t", Z, Z)) / (lam * d))
    P = P_T_beta(U, Z, y, beta, lam)
    t3 = beta / (1.0 - beta) * P
    t4 = (1.0 - beta) / beta * 0.5 * d * float((y**2).sum())

    # Negative controls: each weakens the bound, so a violation MUST become findable.
    if perturb == "halve_log_term":
        t2 *= 0.5
    elif perturb == "drop_last_term":
        t4 = 0.0
    elif perturb == "drop_path_term":
        t3 = 0.0
    elif perturb == "zinkevich_path":
        t3 = beta / (1.0 - beta) * zinkevich_path_length(U)
    elif perturb == "drop_log_term":
        t2 = 0.0
    elif perturb is not None:
        raise ValueError(f"unknown perturbation {perturb!r}")

    return {
        "term1_beta_lambda_u1": t1,
        "term2_logdet": t2,
        "term3_path": t3,
        "term4_discount_slack": t4,
        "P_T_beta": P,
        "rhs": t1 + t2 + t3 + t4,
    }


def F_t(U: np.ndarray, Z: np.ndarray, y: np.ndarray, beta: float, lam: float) -> np.ndarray:
    """F_t^{beta,phi}(u_k) for the t appearing in the path term, evaluated at u_t and u_{t+1}.

    Returns an array G of shape (T-1, 2) with G[t-1, 0] = F_t(u_t), G[t-1, 1] = F_t(u_{t+1}).
    F_t(u) = beta^t phi(u) + sum_{s=1..t} beta^{t-s} f_s(u), phi(u) = (lambda/2)||u||^2.
    """
    T = Z.shape[0]
    L = loss_matrix(U, Z, y, lam)  # L[s, k], s=0..T (row 0 = phi), k = u_{k+1}
    out = np.empty((max(T - 1, 0), 2))
    for t in range(1, T):
        s = np.arange(1, t + 1)
        w = beta ** (t - s)
        for j, k in enumerate((t - 1, t)):  # u_t then u_{t+1}
            out[t - 1, j] = (beta**t) * L[0, k] + float(L[1 : t + 1, k] @ w)
    return out


def derivation_links(
    X: np.ndarray,
    U: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    lam_terms: np.ndarray,
    beta: float,
    lam: float,
    gamma: float | None = None,
) -> dict[str, float]:
    """Check each link of the Appendix C.1 derivation independently.

    Returns slack values (>= 0 means the link holds). Reconstructing the proof
    link by link is what distinguishes 'the final inequality happened to hold'
    from 'the stated derivation is sound'.
    """
    T, d = Z.shape
    gamma = beta if gamma is None else gamma
    f_alg = algorithm_losses(X, Z, y)
    L = loss_matrix(U, Z, y, lam)

    # ---- L1: the rescaled-regret hypothesis (Lemma 2 on rescaled data), in
    # beta^t-scaled form, checked at every t against the comparator u_t:
    #   sum_{s<=t} beta^{t-s}(f_s(x_s) - f_s(u)) <= beta^t (lambda/2)||u||^2
    #                                              + sum_{s<=t} beta^{t-s} lam_s
    l1_min = np.inf
    for t in range(1, T + 1):
        k = t - 1
        s = np.arange(1, t + 1)
        w = beta ** (t - s)
        reg = float(w @ (f_alg[:t] - L[1 : t + 1, k]))
        bound = (beta**t) * L[0, k] + float(w @ lam_terms[:t])
        l1_min = min(l1_min, bound - reg)

    # ---- L2app: Theorem 1 applied, i.e. the intermediate bound Eq. (12):
    #   D-Reg <= beta*(lambda/2)||u_1||^2 + sum_t lam_t + beta*sum_{t<T}(F_t(u_{t+1})-F_t(u_t))
    G = F_t(U, Z, y, beta, lam)
    path_raw = beta * float((G[:, 1] - G[:, 0]).sum()) if T > 1 else 0.0
    dreg = dynamic_regret(X, U, Z, y)
    eq12 = 0.5 * beta * lam * float(U[0] @ U[0]) + float(lam_terms.sum()) + path_raw
    l2_slack = eq12 - dreg

    # ---- L3: Lemma 25 (log-determinant bound) on sum_t lam_t
    weights = beta ** (T - 1 - np.arange(T))
    logdet_rhs = d * np.log(1.0 / beta) * float((y**2).sum()) + float(np.max(y**2)) * d * np.log1p(
        float(weights @ np.einsum("td,td->t", Z, Z)) / (lam * d)
    )
    l3_slack = logdet_rhs - float(lam_terms.sum())

    # ---- L4: ln(1/beta) <= (1-beta)/beta
    l4_slack = (1.0 - beta) / beta - np.log(1.0 / beta)

    # ---- L5: Lemma 18, beta*sum(F_t(u_{t+1})-F_t(u_t)) <= gamma/(1-gamma) P_T^gamma
    l5_slack = gamma / (1.0 - gamma) * P_T_beta(U, Z, y, gamma, lam) - path_raw

    return {
        "L1_rescaled_hypothesis_min_slack": float(l1_min),
        "L2_theorem1_applied_slack": float(l2_slack),
        "L3_logdet_slack": float(l3_slack),
        "L4_log_inequality_slack": float(l4_slack),
        "L5_path_to_P_slack": float(l5_slack),
        "eq12_intermediate_bound": float(eq12),
        "dynamic_regret": float(dreg),
    }
