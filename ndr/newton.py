# Newton solvers: plain and deflated.
# Plain Newton with Armijo backtracking line search.
# Deflated Newton uses Sherman-Morrison to avoid forming the deflated Jacobian.

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.linalg as nla
from scipy.linalg import solve, LinAlgWarning


@dataclass
class NewtonResult:
    x: np.ndarray
    converged: bool
    residual_norm: float
    n_iter: int


def newton_solve(F, J, x0, tol=1e-10, max_iter=20, linesearch=True, verbose=False, FJ=None, **_kw):
    """Standard Newton with optional Armijo backtracking.
    If FJ(x) -> (residual, jacobian) is provided, uses it to avoid
    redundant kernel builds (F and J share the same K assembly)"""
    x = x0.copy()

    # initial evaluation
    if FJ is not None:
        f, jac_ready = FJ(x)
    else:
        f = F(x)
        jac_ready = None
    rn = nla.norm(f)

    if verbose:
        print(f"  Newton 0: |F| = {rn:.4e}")
    if not np.isfinite(rn):
        return NewtonResult(x, False, float('inf'), 0)
    if rn < tol:
        return NewtonResult(x, True, rn, 0)

    for k in range(1, max_iter + 1):
        # get Jacobian: from initial FJ, or fresh FJ/J call
        if jac_ready is not None:
            jac = jac_ready
            jac_ready = None
        elif FJ is not None:
            f, jac = FJ(x)
            rn = nla.norm(f)
            if rn < tol:
                return NewtonResult(x, True, rn, k - 1)
        else:
            jac = J(x)

        if not np.all(np.isfinite(jac)):
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                dx = solve(jac, -f)
        except nla.LinAlgError:
            break
        if not np.all(np.isfinite(dx)):
            break

        # line search uses F only, no Jacobian needed
        xt = x + dx
        ft = F(xt)
        nt = nla.norm(ft)
        accepted = True

        if linesearch and (not np.isfinite(nt) or nt > rn):
            best_n, best_x, best_f = (nt, xt.copy(), ft.copy()) if np.isfinite(nt) else (float('inf'), None, None)
            alpha = 1.0
            for _ in range(10):
                alpha *= 0.5
                xt = x + alpha * dx
                ft = F(xt)
                nt = nla.norm(ft)
                if np.isfinite(nt) and nt < best_n:
                    best_n, best_x, best_f = nt, xt.copy(), ft.copy()
                if np.isfinite(nt) and nt < rn:
                    break
            if best_x is not None:
                xt, ft, nt = best_x, best_f, best_n
            else:
                accepted = False
        elif not np.isfinite(nt):
            accepted = False

        if not accepted:
            break

        x, f, rn = xt, ft, nt
        if verbose:
            print(f"  Newton {k}: |F| = {rn:.4e}")
        if rn < tol:
            return NewtonResult(x, True, rn, k)

    return NewtonResult(x, False, rn, max_iter)


# Deflation inspired/based on Julia library BifurcationKit.jl
class DeflationOperator:
    """
    D(x) = prod_i (||x - xi||^(-2p) + alpha)
    Suppresses convergence to known roots xi
    """
    def __init__(self, roots, p=2, alpha=1.0):
        self.roots = [r.copy() for r in roots]
        self.p = p
        self.alpha = alpha

    def eval(self, x):
        """Returns (D, grad_D)"""
        factors, grads = [], []
        for xi in self.roots:
            d = x - xi
            d2 = np.dot(d, d)
            if d2 < 1e-30:
                return 1e30, np.zeros_like(x)
            fi = d2**(-self.p) + self.alpha
            gi = -2 * self.p * d2**(-self.p - 1) * d
            factors.append(fi)
            grads.append(gi)

        D = 1.0
        for fi in factors:
            D *= fi

        gD = np.zeros_like(x)
        for j, (fj, gj) in enumerate(zip(factors, grads)):
            gD += gj * (D / fj)

        return D, gD


def deflated_newton_solve(F, J, x0, known_roots, deflate_p=2, deflate_alpha=1.0,
                          tol=1e-11, max_iter=28, linesearch=True, verbose=False, **_kw):
    """
    Deflated Newton
    After deflated convergence do plain Newton but checks it
    didnt snap back to known root
    """
    if not known_roots:
        return newton_solve(F, J, x0, tol=tol, max_iter=max_iter,
                            linesearch=linesearch, verbose=verbose)

    defl = DeflationOperator(known_roots, p=deflate_p, alpha=deflate_alpha)

    x = x0.copy()
    f = F(x)
    rn = nla.norm(f)
    M0, _ = defl.eval(x)
    dr = abs(M0) * rn

    if verbose:
        print(f"  Defl 0: |F|={rn:.4e}  |MF|={dr:.4e}")
    if not np.isfinite(rn):
        return NewtonResult(x, False, float('inf'), 0)

    for k in range(1, max_iter + 1):
        jac = J(x)
        if not np.all(np.isfinite(jac)):
            break

        Mv, Mg = defl.eval(x)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                h = solve(jac, f)
        except nla.LinAlgError:
            break
        if not np.all(np.isfinite(h)):
            break

        denom = Mv + np.dot(Mg, h)
        if abs(denom) < 1e-30:
            break

        dx = -(Mv / denom) * h
        if not np.all(np.isfinite(dx)):
            break

        # line search on deflated residual
        xn = x + dx
        fn = F(xn)
        rn_new = nla.norm(fn)
        Mn, _ = defl.eval(xn)
        dr_new = abs(Mn) * rn_new

        if linesearch and (not np.isfinite(dr_new) or dr_new > dr):
            alpha = 1.0
            best_dr = dr_new if np.isfinite(dr_new) else float('inf')
            best_x, best_f, best_rn = (xn.copy(), fn.copy(), rn_new) if np.isfinite(dr_new) else (None, None, None)
            for _ in range(10):
                alpha *= 0.5
                xt = x + alpha * dx
                ft = F(xt)
                rt = nla.norm(ft)
                if np.isfinite(rt):
                    Mt, _ = defl.eval(xt)
                    dt = abs(Mt) * rt
                    if dt < best_dr:
                        best_dr, best_x, best_f, best_rn = dt, xt.copy(), ft.copy(), rt
                    if dt < dr:
                        break
            if best_x is not None:
                xn, fn, rn_new, dr_new = best_x, best_f, best_rn, best_dr
            else:
                break

        x, f, rn, dr = xn, fn, rn_new, dr_new
        if not np.isfinite(rn):
            break

        if verbose:
            print(f"  Defl {k}: |F|={rn:.4e}  |MF|={dr:.4e}")

        if dr < tol:
            # polish with plain Newton, but guard against snapping back
            pol = newton_solve(F, J, x, tol=tol, max_iter=10, linesearch=True)
            if pol.converged:
                snapped = False
                for xi in known_roots:
                    if nla.norm(pol.x - xi) / max(nla.norm(xi), 1.0) < 1e-4:
                        snapped = True
                        break
                if not snapped:
                    x, rn = pol.x, pol.residual_norm
                elif verbose:
                    print("  Defl: snapped back, keeping prev sol")
            if verbose:
                print(f"  Defl: |F|={rn:.4e}")
            return NewtonResult(x, True, rn, k)

    return NewtonResult(x, False, rn, max_iter)
