# Natural parameter continuation
# Natural continuation steps N monotonically and solves Newton at each step
# Step size adapts: success -> grow, failure -> shrink

import warnings
from dataclasses import dataclass, field

import numpy as np

from .quadrature import KernelPrecomp
from .state import pack_state, unpack_state, normalize_power, enforce_gauge, enforce_component_gauge
from .residuals import (
    residual_reduced, jacobian_reduced,
    residual_reduced_component_gauge, jacobian_reduced_component_gauge,
    residual_full_imperfect, jacobian_full_imperfect,
    residual_and_jacobian_full_imperfect,
    param_deriv_full_imperfect_N, param_deriv_full_imperfect_delta,
)
from .theory import theory_dimer
from .newton import newton_solve
from .branch_switching import linear_scan_mode_2d
from .palc import palc_continue, PALCBranch, PALCPoint


@dataclass
class BranchPoint:
    x: np.ndarray
    p: float
    step: int


@dataclass
class NaturalBranch:
    sol: list = field(default_factory=list)
    param: list = field(default_factory=list)


def natural_continue(prob_builder, x_start, N_start, N_max,
                     dN=0.0, dN_max=0.01, dN_min=1e-7,
                     newton_tol=1e-10, newton_max_iter=20, verbosity=1):

    if dN <= 0:
        dN = max(0.2 * N_start, 5e-5)
    dN = min(dN, dN_max)

    branch = NaturalBranch()
    branch.sol.append(BranchPoint(x=x_start.copy(), p=N_start, step=0))
    branch.param.append(N_start)

    x = x_start.copy()
    x_prev = None
    N = N_start
    N_prev = None
    step = 0

    while N < N_max - 1e-14:
        N_try = min(N + dN, N_max)

        # secant predictor
        if x_prev is not None and N_prev is not None:
            dN_last = N - N_prev
            if abs(dN_last) > 1e-14:
                x_guess = x + ((N_try - N) / dN_last) * (x - x_prev)
            else:
                x_guess = x.copy()
        else:
            x_guess = x.copy()

        built = prob_builder(N_try, x_guess)
        if len(built) == 3:
            F, J, FJ = built
        else:
            F, J = built
            FJ = None

        result = newton_solve(F, J, x_guess, tol=newton_tol, max_iter=newton_max_iter,
                              linesearch=True, FJ=FJ)

        if verbosity > 0:
            tag = "OK" if result.converged else "FAIL"
            print(f"  [cont] N={N:.4e} -> {N_try:.4e}  dN={dN:.2e}  {tag}  |F|={result.residual_norm:.2e}")

        if result.converged:
            step += 1
            x_prev = x.copy()
            N_prev = N
            x = result.x
            N = N_try
            branch.sol.append(BranchPoint(x=x.copy(), p=N, step=step))
            branch.param.append(N)
            dN = min(dN * 1.25, dN_max)
        else:
            dN *= 0.5
            if dN < dN_min:
                warnings.warn(f"Continuation: dN < {dN_min:.2e} at N={N:.4e}, stopping.")
                break

    return branch


def continue_branch_reduced(pre, sigma, tau, eta, beta, L, *,
                            N_min=0.2, N_max=6.0, dN=0.0, dN_max=0.01, dN_min=1e-7,
                            omega_prev=None, seed_u="theory", seed_from=None,
                            gauge="component", gauge_idx=-1,
                            newton_tol=1e-10, newton_max_iter=20, verbosity=1):
    """
    Seed and continue the symmetry-reduced branch in N
    Returns (branch, uref, info)
    """
    n0 = len(pre.W)
    th = theory_dimer(pre, L, tau=tau, eta=eta)

    if N_min <= 0 or not np.isfinite(N_min):
        N_min = max(1e-3 * th.Ncrit_beta1 / max(beta.real, 1e-16), 1e-8)

    # omega seed
    w_th = th.omega_even_star if sigma == +1 else th.omega_odd_star
    if omega_prev is not None:
        w0 = omega_prev
    else:
        w0 = complex(w_th, -0.06 * abs(w_th))

    #u seed
    if seed_u == "previous" and seed_from is not None:
        u0 = seed_from[0].copy().astype(complex)
        w0 = seed_from[1]
    else:
        phi = th.phi_even if sigma == +1 else th.phi_odd
        u0 = phi.astype(complex).copy()

    normalize_power(u0, pre.W, N_min / 2.0)

    #gauge setup
    uref = None
    if gauge == "component":
        if gauge_idx < 0:
            gauge_idx = int(np.argmax(np.abs(u0)))
        enforce_component_gauge(u0, gauge_idx)
    else:
        uref = u0.copy()
        enforce_gauge(u0, uref, pre.W)

    x0 = pack_state(u0, w0)

    #initial Newton
    p = np.array([N_min, L])
    if gauge == "component":
        F_at = lambda x: residual_reduced_component_gauge(x, p, pre, sigma, tau, eta, beta, gauge_idx)
        J_at = lambda x: jacobian_reduced_component_gauge(x, p, pre, sigma, tau, eta, beta, gauge_idx)
    else:
        F_at = lambda x: residual_reduced(x, p, pre, sigma, tau, eta, beta, uref)
        J_at = lambda x: jacobian_reduced(x, p, pre, sigma, tau, eta, beta, uref)

    sol0 = newton_solve(F_at, J_at, x0, tol=newton_tol, max_iter=newton_max_iter, linesearch=True)

    if not sol0.converged:
        if verbosity >= 0:
            warnings.warn(f"Initial Newton failed (sigma={sigma}, L={L:.4f}), retrying with omega-scan")
        w0, _, _ = linear_scan_mode_2d(
            pre, sigma, tau, eta, L, omega_center=w_th,
            delta_r=0.15, delta_i_min=0.02, delta_i_max=0.20, n_r=15, n_i=8)
        x0 = pack_state(u0, w0)
        if gauge == "component":
            F_at = lambda x: residual_reduced_component_gauge(x, p, pre, sigma, tau, eta, beta, gauge_idx)
            J_at = lambda x: jacobian_reduced_component_gauge(x, p, pre, sigma, tau, eta, beta, gauge_idx)
        sol0 = newton_solve(F_at, J_at, x0, tol=newton_tol, max_iter=newton_max_iter, linesearch=True)

    if not sol0.converged:
        raise RuntimeError(
            f"Initial Newton did not converge (sigma={sigma}, L={L:.4f}, "
            f"|F|={sol0.residual_norm:.2e}). Cannot start from unconverged seed.")

    x_start = sol0.x

    #continuation
    def prob_builder(N_val, x_guess):
        pv = np.array([N_val, L])
        if gauge == "component":
            F = lambda x: residual_reduced_component_gauge(x, pv, pre, sigma, tau, eta, beta, gauge_idx)
            J = lambda x: jacobian_reduced_component_gauge(x, pv, pre, sigma, tau, eta, beta, gauge_idx)
        else:
            F = lambda x: residual_reduced(x, pv, pre, sigma, tau, eta, beta, uref)
            J = lambda x: jacobian_reduced(x, pv, pre, sigma, tau, eta, beta, uref)
        return F, J

    branch = natural_continue(prob_builder, x_start, N_min, N_max,
                              dN=dN, dN_max=dN_max, dN_min=dN_min,
                              newton_tol=newton_tol, newton_max_iter=newton_max_iter,
                              verbosity=verbosity)

    return branch, uref, {"omega0": w0, "gauge_idx": gauge_idx}


def continue_full_imperfect_in_N(pre, tau, eta, beta, L_fixed, delta_fixed,
                                 N_start, N_max, x0, gauge_idx, *,
                                 dN=0.0, dN_max=0.02, dN_min=1e-8,
                                 newton_tol=1e-11, newton_max_iter=30, verbosity=1):
    """Natural continuation in N for the imperfect full-dimer at fixed (L, delta)"""
    if dN <= 0:
        dN = max(0.15 * N_start, 1e-4)
    dN = min(dN, dN_max)

    def prob_builder(N_val, x_guess):
        pv = np.array([N_val, L_fixed, delta_fixed])
        F = lambda x: residual_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)
        J = lambda x: jacobian_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)
        FJ = lambda x: residual_and_jacobian_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)
        return F, J, FJ

    return natural_continue(prob_builder, x0, N_start, N_max,
                            dN=dN, dN_max=dN_max, dN_min=dN_min,
                            newton_tol=newton_tol, newton_max_iter=newton_max_iter,
                            verbosity=verbosity)


def continue_full_imperfect_in_delta(pre, tau, eta, beta, N_fixed, L_fixed,
                                     delta_start, delta_end, x0, gauge_idx, *,
                                     d_delta=0.0, d_delta_max=5e-4, d_delta_min=1e-10,
                                     newton_tol=1e-11, newton_max_iter=30, verbosity=1):
    """Natural continuation in delta at fixed (N, L), here this is delta->0 homotopy idea"""
    direction = 1.0 if delta_end > delta_start else -1.0
    if d_delta <= 0:
        d_delta = max(0.25 * abs(delta_start), 1e-4)
    d_delta *= direction

    branch = NaturalBranch()
    branch.sol.append(BranchPoint(x=x0.copy(), p=delta_start, step=0))
    branch.param.append(delta_start)

    x = x0.copy()
    x_prev = None
    delta = delta_start
    delta_prev = None
    step = 0

    while (direction > 0 and delta < delta_end - 1e-14) or \
          (direction < 0 and delta > delta_end + 1e-14):
        dt = min(delta + d_delta, delta_end) if direction > 0 else max(delta + d_delta, delta_end)

        # secant predictor
        if x_prev is not None and delta_prev is not None and abs(delta - delta_prev) > 1e-14:
            x_guess = x + ((dt - delta) / (delta - delta_prev)) * (x - x_prev)
        else:
            x_guess = x.copy()

        pv = np.array([N_fixed, L_fixed, dt])
        F = lambda x: residual_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)
        J = lambda x: jacobian_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)
        FJ = lambda x: residual_and_jacobian_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)

        result = newton_solve(F, J, x_guess, tol=newton_tol, max_iter=newton_max_iter,
                              linesearch=True, FJ=FJ)

        if verbosity > 0:
            tag = "OK" if result.converged else "FAIL"
            print(f"  [delta] {delta:.3e} -> {dt:.3e}  dd={d_delta:.2e}  {tag}")

        if result.converged:
            step += 1
            x_prev = x.copy()
            delta_prev = delta
            x = result.x
            delta = dt
            branch.sol.append(BranchPoint(x=x.copy(), p=delta, step=step))
            branch.param.append(delta)
            d_delta = np.sign(d_delta) * min(abs(d_delta) * 1.25, d_delta_max)
        else:
            d_delta *= 0.5
            if abs(d_delta) < d_delta_min:
                warnings.warn(f"Delta homotopy: dd < {d_delta_min:.2e} at delta={delta:.4e}, stopping.")
                break

    return branch

# PALC wrappers
def _palc_to_natural_branch(palc_br):
    """Convert PALCBranch to NaturalBranch struct"""
    br = NaturalBranch()
    for pt in palc_br.sol:
        br.sol.append(BranchPoint(x=pt.x.copy(), p=pt.p, step=pt.step))
        br.param.append(pt.p)
    return br


def palc_full_imperfect_in_N(pre, tau, eta, beta, L_fixed, delta_fixed,
                             N_start, N_max, x0, gauge_idx, *,
                             ds=0.005, ds_max=0.02, ds_min=1e-8,
                             max_steps=500, theta=1.0,
                             newton_tol=1e-11, newton_max_iter=30, verbosity=1):
    """PALC continuation in N for the imperfectdimer at fixed (L, delta)
    Returns a NaturalBranch struct
    """
    def F(x, N_val):
        pv = np.array([N_val, L_fixed, delta_fixed])
        return residual_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)

    def J(x, N_val):
        pv = np.array([N_val, L_fixed, delta_fixed])
        return jacobian_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)

    def Fp(x, N_val):
        pv = np.array([N_val, L_fixed, delta_fixed])
        return param_deriv_full_imperfect_N(x, pv, pre, tau, eta, beta, gauge_idx)

    orient = +1.0 if N_max > N_start else -1.0
    p_lo = min(N_start, N_max)
    p_hi = max(N_start, N_max)

    palc_br = palc_continue(
        F, J, x0, N_start,
        max_steps=max_steps, ds=ds, ds_min=ds_min, ds_max=ds_max,
        newton_tol=newton_tol, newton_max_iter=newton_max_iter,
        F_p=Fp, orient=orient, theta=theta,
        p_min=max(0.0, p_lo), p_max=p_hi,
        verbosity=verbosity,
    )
    return _palc_to_natural_branch(palc_br)


def palc_full_imperfect_in_delta(pre, tau, eta, beta, N_fixed, L_fixed,
                                 delta_start, delta_end, x0, gauge_idx, *,
                                 ds=1e-4, ds_max=5e-4, ds_min=1e-10,
                                 max_steps=500, theta=1.0,
                                 newton_tol=1e-11, newton_max_iter=30, verbosity=1):
    """PALC continuation in delta at fixed (N, L), here for the delta->0 homotopy
    Returns NaturalBranch struct
    """
    orient = +1.0 if delta_end > delta_start else -1.0

    def F(x, delta_val):
        pv = np.array([N_fixed, L_fixed, delta_val])
        return residual_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)

    def J(x, delta_val):
        pv = np.array([N_fixed, L_fixed, delta_val])
        return jacobian_full_imperfect(x, pv, pre, tau, eta, beta, gauge_idx)

    def Fp(x, delta_val):
        pv = np.array([N_fixed, L_fixed, delta_val])
        return param_deriv_full_imperfect_delta(x, pv, pre, tau, eta, beta, gauge_idx)

    p_lo = min(delta_start, delta_end)
    p_hi = max(delta_start, delta_end)

    palc_br = palc_continue(
        F, J, x0, delta_start,
        max_steps=max_steps, ds=ds, ds_min=ds_min, ds_max=ds_max,
        newton_tol=newton_tol, newton_max_iter=newton_max_iter,
        F_p=Fp, orient=orient, theta=theta,
        p_min=p_lo, p_max=p_hi,
        verbosity=verbosity,
    )
    return _palc_to_natural_branch(palc_br)
