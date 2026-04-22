# Branch switching tools: linear scan, pitchfork indicator, asymmetric seeding.

import numpy as np
import numpy.linalg as nla
from .quadrature import KernelPrecomp
from .kernel import K_self, K_cross
from .state import normalize_power


def linear_scan_mode(pre, sigma, tau, eta, L,
                     omega_r_min=0.5, omega_r_max=6.0, omega_i_fixed=-0.08, n_scan=60):
    """
    1D scan over Re(omega) to find near-singularity of A = I - tau eta omega^2 K_eff
    Returns (omega0, v0, smin)
    """
    n0 = len(pre.W)
    best_s, best_w, best_v = np.inf, 0j, np.zeros(n0, dtype=complex)
    Id = np.eye(n0, dtype=complex)

    for wr in np.linspace(omega_r_min, omega_r_max, n_scan):
        w = complex(wr, omega_i_fixed)
        K = K_self(w, pre) + sigma * K_cross(w, L, pre)
        A = Id - tau * eta * w**2 * K
        _, s, Vh = nla.svd(A)
        if s[-1] < best_s:
            best_s = s[-1]
            best_w = w
            best_v = Vh[-1].conj()

    return best_w, best_v, best_s


def linear_scan_mode_2d(pre, sigma, tau, eta, L, omega_center,
                        delta_r=0.35, delta_i_min=0.01, delta_i_max=0.30,
                        n_r=20, n_i=10):
    """
    2D scan over (Re omega, Im omega) near omega_center
    Returns (omega0, v0, smin)
    """
    n0 = len(pre.W)
    wr_grid = np.linspace(omega_center*(1-delta_r), omega_center*(1+delta_r), n_r)
    wi_grid = np.linspace(-abs(omega_center)*delta_i_max, -abs(omega_center)*delta_i_min, n_i)
    Id = np.eye(n0, dtype=complex)

    best_s, best_w, best_v = np.inf, 0j, np.zeros(n0, dtype=complex)
    for wr in wr_grid:
        for wi in wi_grid:
            w = complex(wr, wi)
            K = K_self(w, pre) + sigma * K_cross(w, L, pre)
            A = Id - tau * eta * w**2 * K
            _, s, Vh = nla.svd(A)
            if s[-1] < best_s:
                best_s = s[-1]
                best_w = w
                best_v = Vh[-1].conj()

    return best_w, best_v, best_s


def odd_block_smin(u, omega, L, pre, tau, eta, beta, return_vec=True):
    """
    Smallest singular val of the odd-subspace PDE Jacobian along an even solution
    pitchfork indicator: when it crosses zero=> symmetry breaking
    Returns (smin, v_complex)
    """
    n0 = len(pre.W)
    K = K_self(omega, pre) - K_cross(omega, L, pre)  # odd sector

    absu2 = np.abs(u)**2
    dg = eta + 2 * beta * absu2
    dg_conj = beta * u**2

    Id = np.eye(n0, dtype=complex)
    c = tau * omega**2
    M1 = Id - c * (K * dg[None, :])
    M2 = -c * (K * dg_conj[None, :])

    A = M1 + M2
    B = M1 - M2

    J_pde = np.block([[A.real, -B.imag],
                      [A.imag,  B.real]])

    _, s, Vh = nla.svd(J_pde)
    smin = s[-1]

    if not return_vec:
        return smin, np.zeros(n0, dtype=complex)

    v = Vh[-1]
    return smin, v[:n0] + 1j * v[n0:]


def seed_asymmetric_from_even(pre, u_even, omega, v_odd, N, epsilon=1e-2):
    """
    Build full-dimer guess by perturbing even solution along odd direction:
    u1 = u_even + eps*v_odd, u2 = u_even - eps*v_odd
    """
    u_full = np.concatenate([u_even + epsilon*v_odd,
                             u_even - epsilon*v_odd])
    Wf = np.concatenate([pre.W, pre.W])
    normalize_power(u_full, Wf, N)
    return u_full


def asymmetry_measure(u_full, pre):
    """A = (P1 - P2)/(P1 + P2) where Pi = ||ui||^2_W"""
    n0 = len(pre.W)
    P1 = np.sum(np.abs(u_full[:n0])**2 * pre.W)
    P2 = np.sum(np.abs(u_full[n0:])**2 * pre.W)
    return (P1 - P2) / (P1 + P2 + 1e-16)
