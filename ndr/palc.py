"""
Pseudo-Arclength Continuation (PALC).

Unlike natural continuation, PALC parametrises the branch by arc-length so it can turn folds

Algorithm (Keller 1977):
  Predictor: (x_pred, p_pred) = (x_k + ds*t_x, p_k + ds*t_p)
  Corrector: Newton on [F(x,p); t_x.(x-x_k) + theta*t_p.(p-p_k) - ds] = 0

The weight theta balances x (large) vs p (small) in the arclength norm:
  ||(dx, dp)||_theta^2 = ||dx||_2^2 + theta * dp^2
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg


@dataclass
class PALCPoint:
    x: np.ndarray
    p: float
    t_x: np.ndarray
    t_p: float
    s: float
    step: int


@dataclass
class PALCBranch:
    sol: list = field(default_factory=list)

    @property
    def params(self):
        return [pt.p for pt in self.sol]

    @property
    def param(self):
        """Alias for compatibility with NaturalBranch interface."""
        return self.params

    @property
    def solutions(self):
        return [pt.x for pt in self.sol]


def _fd_fp(F, x, p, eps=1e-7):
    """Finite-difference dF/dp"""
    return (F(x, p + eps) - F(x, p - eps)) / (2 * eps)


def _compute_tangent(J_mat, Fp_vec, t_prev_x, t_prev_p, theta=1.0, x_weight=1.0):
    """Compute unit tangent by solving J t_x = -F_p, normalise with theta, orient"""
    try:
        t_x = scipy.linalg.solve(J_mat, -Fp_vec)
        t_p = 1.0
    except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
        return t_prev_x.copy(), t_prev_p

    norm = np.sqrt(x_weight**2 *np.dot(t_x, t_x) + theta * t_p**2)
    if norm < 1e-30:
        return t_prev_x.copy(), t_prev_p
    t_x /= norm
    t_p /= norm

    if x_weight ** 2 * np.dot(t_x, t_prev_x) + theta * t_p * t_prev_p < 0:
        t_x, t_p = -t_x, -t_p
    return t_x, t_p


def palc_continue(F, J, x0, p0, *, max_steps=300, ds=0.01, ds_min=1e-7, ds_max=0.1,
                  newton_tol=1e-10, newton_max_iter=20,
                  F_p=None, fp_eps=1e-7, orient=+1.0, theta=1.0,
                  p_min=None, p_max=None, feasible=(lambda x: x[-1]<=0.0), verbosity=0):
    """
    PALC from (x0, p0)
    F(x, p) residual, J(x, p)  Jacobian
    F_p(x, p) (uses finite differences if None)
    orient: +1 for increasing p initially, -1 for decreasing
    theta: arclength scaling weight for p (large theta => p moves less per step)
    p_min, p_max: optional bounds on the continuation parameter
    """
    x, p = x0.copy(), float(p0)
    n = len(x)
    x_weight = 1.0/np.sqrt(max(n,1))
    ds = float(ds)

    Fp = F_p if F_p is not None else lambda xx, pp: _fd_fp(F, xx, pp, fp_eps)

    # initial tangent
    J0 = J(x, p)
    Fp0 = Fp(x, p)
    try:
        t_x = scipy.linalg.solve(J0, -Fp0)
        t_p = 1.0
        norm = np.sqrt(x_weight**2 *np.dot(t_x, t_x) + theta * t_p**2)
        t_x /= norm
        t_p /= norm
    except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
        t_x = np.zeros(n)
        t_p = 1.0
    if t_p * orient < 0:
        t_x, t_p = -t_x, -t_p

    branch = PALCBranch()
    s = 0.0
    branch.sol.append(PALCPoint(x=x.copy(), p=p, t_x=t_x.copy(), t_p=t_p, s=s, step=0))

    for step in range(1, max_steps + 1):
        # predictor
        xp = x + ds * t_x
        pp = p + ds * t_p

        # corrector
        xc, pc = xp.copy(), float(pp)
        converged = False
        rn = np.inf

        for _ in range(newton_max_iter):
            Fv = F(xc, pc)
            arc = x_weight**2 *np.dot(t_x, xc - x) + theta * t_p * (pc - p) - ds
            G = np.append(Fv, arc)
            rn = np.linalg.norm(G)
            if not np.isfinite(rn):
                break
            if rn < newton_tol:
                converged = True
                break

            Jm = J(xc, pc)
            Fpv = Fp(xc, pc)
            if not np.all(np.isfinite(Jm)) or not np.all(np.isfinite(Fpv)):
                break
            dG = np.empty((n+1, n+1))
            dG[:n, :n] = Jm
            dG[:n, n] = Fpv
            dG[n, :n] = x_weight**2 *t_x
            dG[n, n] = theta * t_p

            try:
                dy = scipy.linalg.solve(dG, -G)
            except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
                break
            if not np.all(np.isfinite(dy)):
                break
            #backtrack to reject nonphysical positive imaginary freq
            alpha = 1.0
            best_x= xc+dy[:n]
            best_p = pc+float(dy[n])
            best_rn = np.inf
            accepted = False
            for _ls in range(12):
                xt = xc + alpha * dy[:n]
                pt = pc + alpha * float(dy[n])
                if not feasible(xt):
                    alpha *= 0.5
                    continue
                Ft = F(xt, pt)
                arct = x_weight ** 2 * np.dot(t_x, xt - x) + theta * t_p * (pt - p) - ds
                nt = np.linalg.norm(np.append(Ft, arct))
                if np.isfinite(nt) and nt < best_rn:
                    best_rn, best_x, best_p = nt, xt.copy(), float(pt)
                if np.isfinite(nt) and nt < rn:
                    accepted = True
                    best_rn = nt
                    best_x = xt.copy()
                    best_p = float(pt)
                    break
                alpha *= 0.5
            if not np.isfinite(best_rn):
                break
            xc, pc = best_x, best_p
            if not accepted and best_rn >= rn:
                break

        if converged:
            # check parameter bounds
            if p_min is not None and pc < p_min:
                break
            if p_max is not None and pc > p_max:
                break

            t_x_new, t_p_new = _compute_tangent(J(xc, pc), Fp(xc, pc), t_x, t_p, theta=theta, x_weight=x_weight)
            s += ds
            x, p, t_x, t_p = xc, pc, t_x_new, t_p_new
            branch.sol.append(PALCPoint(x=x.copy(), p=p, t_x=t_x.copy(), t_p=t_p, s=s, step=step))
            ds = min(ds * 1.2, ds_max)
            if verbosity > 0:
                print(f"  [PALC] step={step:4d}  p={p:+.6f}  ds={ds:.3e}  |G|={rn:.1e}")
        else:
            ds *= 0.5
            if ds < ds_min:
                warnings.warn(f"PALC: ds < {ds_min:.1e} at step {step}, stopping.")
                break
            if verbosity > 0:
                print(f"  [PALC] step={step:4d}  FAIL  ds -> {ds:.2e}")

    return branch
