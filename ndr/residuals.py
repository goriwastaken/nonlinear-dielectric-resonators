# Residuals and Jacobians for the nonlinear Lippmann-Schwinger system
# PDE:   F(u, omega) = u - tau omega^2 K(eta u + beta |u|^2 u) = 0
# plus normalization constraint and phase gauge
#
# Unknowns are realified: x = [Re(u), Im(u), Re(omega), Im(omega)]

import numpy as np
from .quadrature import KernelPrecomp
from .kernel import K_self, K_cross, dK_self, dK_cross, build_full_K, build_full_K_and_dK
from .state import unpack_state, inner_W


# Kerr nonlinearity
def _kerr_source(u, eta, beta):
    """
    g(u)  = eta* u + beta * |u|^2 * u
    dg/du = eta + 2*beta*|u|^2
    dg/du_conj = beta * u^2
    """
    absu2 = np.abs(u)**2
    g = eta * u + beta * absu2 * u
    dg = eta + 2 * beta * absu2
    dg_conj = beta * u**2
    return g, dg, dg_conj


def _pde_jac_blocks(K, dK, g, dg, dg_conj, u, omega, tau):
    """
    Build the 8 real blocks of the realified Jacobian

    The PDE is F = u - tau omega^2 K g(u)
    Returns J11, J12, J21, J22 (n x n each) and 4 omega-column vectors
    """
    n = len(u)
    Kg = K @ g
    dKg = dK @ g

    # omega column: dF/d(omega) = -tau (2 omega K g + omega^2 dK g)
    Fw = -tau * (2 * omega * Kg + omega**2 * dKg)

    Id = np.eye(n, dtype=complex)
    c = tau * omega**2
    M1 = Id - c * (K * dg[None, :])
    M2 = -c * (K * dg_conj[None, :])

    A = M1 + M2
    B = M1 - M2

    J11 = A.real
    J12 = -B.imag
    J21 = A.imag
    J22 = B.real

    return (J11, J12, J21, J22,
            Fw.real, -Fw.imag, Fw.imag, Fw.real)


#Reduced system (one sphere, sigma = +-1)

def residual_reduced(x, p, pre, sigma, tau, eta, beta, uref):
    """Reduced residual with innerprod gauge"""
    N, L = p[0], p[1]
    n0 = len(pre.W)
    u, omega = unpack_state(x, n0)

    K = K_self(omega, pre) + sigma * K_cross(omega, L, pre)
    g, _, _ = _kerr_source(u, eta, beta)
    Fu = u - tau * omega**2 * (K @ g)

    pw = np.sum(np.abs(u)**2 * pre.W)
    C = 2 * pw - N
    P = 2 * inner_W(u, uref, pre.W).imag

    r = np.empty(2*n0 + 2)
    r[:n0] = Fu.real
    r[n0:2*n0] = Fu.imag
    r[2*n0] = C
    r[2*n0+1] = P
    return r


def jacobian_reduced(x, p, pre, sigma, tau, eta, beta, uref):
    """Jacobian of residual_reduced"""
    N, L = p[0], p[1]
    n0 = len(pre.W)
    u, omega = unpack_state(x, n0)

    K = K_self(omega, pre) + sigma * K_cross(omega, L, pre)
    dK = dK_self(omega, pre) + sigma * dK_cross(omega, L, pre)
    g, dg, dg_conj = _kerr_source(u, eta, beta)

    (J11, J12, J21, J22,
     dRF_dor, dRF_doi, dIF_dor, dIF_doi) = _pde_jac_blocks(K, dK, g, dg, dg_conj, u, omega, tau)

    m = 2*n0 + 2
    J = np.zeros((m, m))
    J[:n0, :n0] = J11
    J[:n0, n0:2*n0] = J12
    J[n0:2*n0, :n0] = J21
    J[n0:2*n0, n0:2*n0] = J22
    J[:n0, 2*n0] = dRF_dor
    J[:n0, 2*n0+1] = dRF_doi
    J[n0:2*n0, 2*n0] = dIF_dor
    J[n0:2*n0, 2*n0+1] = dIF_doi

    # power row: d(2 sum W|u|^2)/d(Re u, Im u)
    J[2*n0, :n0] = 4 * u.real * pre.W
    J[2*n0, n0:2*n0] = 4 * u.imag * pre.W

    # gauge row: d(2 Im <u, uref>_W)
    J[2*n0+1, :n0] = -2 * uref.imag * pre.W
    J[2*n0+1, n0:2*n0] = 2 * uref.real * pre.W
    return J


#Reduced system with component gauge
def residual_reduced_component_gauge(x, p, pre, sigma, tau, eta, beta, gauge_idx):
    """component gauge Im(u[gauge_idx]) = 0"""
    N, L = p[0], p[1]
    n0 = len(pre.W)
    u, omega = unpack_state(x, n0)

    K = K_self(omega, pre) + sigma * K_cross(omega, L, pre)
    g, _, _ = _kerr_source(u, eta, beta)
    Fu = u - tau * omega**2 * (K @ g)

    pw = np.sum(np.abs(u)**2 * pre.W)
    C = 2 * pw - N
    P = u[gauge_idx].imag

    r = np.empty(2*n0 + 2)
    r[:n0] = Fu.real
    r[n0:2*n0] = Fu.imag
    r[2*n0] = C
    r[2*n0+1] = P
    return r


def jacobian_reduced_component_gauge(x, p, pre, sigma, tau, eta, beta, gauge_idx):
    """Jacobian of residual_reduced_component_gauge"""
    N, L = p[0], p[1]
    n0 = len(pre.W)
    u, omega = unpack_state(x, n0)

    K = K_self(omega, pre) + sigma * K_cross(omega, L, pre)
    dK = dK_self(omega, pre) + sigma * dK_cross(omega, L, pre)
    g, dg, dg_conj = _kerr_source(u, eta, beta)

    (J11, J12, J21, J22,
     dRF_dor, dRF_doi, dIF_dor, dIF_doi) = _pde_jac_blocks(K, dK, g, dg, dg_conj, u, omega, tau)

    m = 2*n0 + 2
    J = np.zeros((m, m))
    J[:n0, :n0] = J11
    J[:n0, n0:2*n0] = J12
    J[n0:2*n0, :n0] = J21
    J[n0:2*n0, n0:2*n0] = J22
    J[:n0, 2*n0] = dRF_dor
    J[:n0, 2*n0+1] = dRF_doi
    J[n0:2*n0, 2*n0] = dIF_dor
    J[n0:2*n0, 2*n0+1] = dIF_doi

    J[2*n0, :n0] = 4 * u.real * pre.W
    J[2*n0, n0:2*n0] = 4 * u.imag * pre.W

    J[2*n0+1, :] = 0
    J[2*n0+1, n0 + gauge_idx] = 1
    return J


#Full dimer (2 spheres, 2n0 unknowns)
def residual_full(x, p, pre, tau, eta, beta, uref_full):
    N, L = p[0], p[1]
    n0 = len(pre.W)
    n = 2 * n0
    u, omega = unpack_state(x, n)

    K = build_full_K(omega, L, pre)
    g, _, _ = _kerr_source(u, eta, beta)
    Fu = u - tau * omega**2 * (K @ g)

    Wf = np.concatenate([pre.W, pre.W])
    C = np.sum(np.abs(u)**2 * Wf) - N
    P = inner_W(u, uref_full, Wf).imag

    r = np.empty(2*n + 2)
    r[:n] = Fu.real
    r[n:2*n] = Fu.imag
    r[2*n] = C
    r[2*n+1] = P
    return r


def jacobian_full(x, p, pre, tau, eta, beta, uref_full):
    N, L = p[0], p[1]
    n0 = len(pre.W)
    n = 2 * n0
    u, omega = unpack_state(x, n)

    K, dK = build_full_K_and_dK(omega, L, pre)
    g, dg, dg_conj = _kerr_source(u, eta, beta)

    (J11, J12, J21, J22,
     dRF_dor, dRF_doi, dIF_dor, dIF_doi) = _pde_jac_blocks(K, dK, g, dg, dg_conj, u, omega, tau)

    m = 2*n + 2
    J = np.zeros((m, m))
    J[:n, :n] = J11
    J[:n, n:2*n] = J12
    J[n:2*n, :n] = J21
    J[n:2*n, n:2*n] = J22
    J[:n, 2*n] = dRF_dor
    J[:n, 2*n+1] = dRF_doi
    J[n:2*n, 2*n] = dIF_dor
    J[n:2*n, 2*n+1] = dIF_doi

    Wf = np.concatenate([pre.W, pre.W])
    J[2*n, :n] = 2 * u.real * Wf
    J[2*n, n:2*n] = 2 * u.imag * Wf
    J[2*n+1, :n] = -uref_full.imag * Wf
    J[2*n+1, n:2*n] = uref_full.real * Wf
    return J


def residual_full_component_gauge(x, p, pre, tau, eta, beta, gauge_idx):
    N, L = p[0], p[1]
    n0 = len(pre.W)
    n = 2 * n0
    u, omega = unpack_state(x, n)

    K = build_full_K(omega, L, pre)
    g, _, _ = _kerr_source(u, eta, beta)
    Fu = u - tau * omega**2 * (K @ g)

    Wf = np.concatenate([pre.W, pre.W])
    C = np.sum(np.abs(u)**2 * Wf) - N
    P = u[gauge_idx].imag

    r = np.empty(2*n + 2)
    r[:n] = Fu.real
    r[n:2*n] = Fu.imag
    r[2*n] = C
    r[2*n+1] = P
    return r


def jacobian_full_component_gauge(x, p, pre, tau, eta, beta, gauge_idx):
    N, L = p[0], p[1]
    n0 = len(pre.W)
    n = 2 * n0
    u, omega = unpack_state(x, n)

    K, dK = build_full_K_and_dK(omega, L, pre)
    g, dg, dg_conj = _kerr_source(u, eta, beta)

    (J11, J12, J21, J22,
     dRF_dor, dRF_doi, dIF_dor, dIF_doi) = _pde_jac_blocks(K, dK, g, dg, dg_conj, u, omega, tau)

    m = 2*n + 2
    J = np.zeros((m, m))
    J[:n, :n] = J11
    J[:n, n:2*n] = J12
    J[n:2*n, :n] = J21
    J[n:2*n, n:2*n] = J22
    J[:n, 2*n] = dRF_dor
    J[:n, 2*n+1] = dRF_doi
    J[n:2*n, 2*n] = dIF_dor
    J[n:2*n, 2*n+1] = dIF_doi

    Wf = np.concatenate([pre.W, pre.W])
    J[2*n, :n] = 2 * u.real * Wf
    J[2*n, n:2*n] = 2 * u.imag * Wf
    J[2*n+1, :] = 0
    J[2*n+1, n + gauge_idx] = 1
    return J


#Full dimer with imperfection delta (eta1 = eta(1+delta), eta2 = eta(1-delta))

def _eta_vec(n0, eta, delta):
    """Per  node eta for the imperfect dimer"""
    return np.concatenate([np.full(n0, eta*(1+delta)),
                           np.full(n0, eta*(1-delta))])


def residual_full_imperfect(x, p, pre, tau, eta, beta, gauge_idx):
    """Full dimer with symm breaking delta, component gauge p = [N, L, delta]"""
    N, L, delta = p[0], p[1], p[2]
    n0 = len(pre.W)
    n = 2 * n0
    u, omega = unpack_state(x, n)

    K = build_full_K(omega, L, pre)
    eta_v = _eta_vec(n0, eta, delta)
    absu2 = np.abs(u)**2
    g = eta_v * u + beta * absu2 * u
    Fu = u - tau * omega**2 * (K @ g)

    Wf = np.concatenate([pre.W, pre.W])
    C = np.sum(absu2 * Wf) - N
    P = u[gauge_idx].imag

    r = np.empty(2*n + 2)
    r[:n] = Fu.real
    r[n:2*n] = Fu.imag
    r[2*n] = C
    r[2*n+1] = P
    return r


def jacobian_full_imperfect(x, p, pre, tau, eta, beta, gauge_idx):
    """Jacobian of residual_full_imperfect"""
    N, L, delta = p[0], p[1], p[2]
    n0 = len(pre.W)
    n = 2 * n0
    u, omega = unpack_state(x, n)

    K, dK = build_full_K_and_dK(omega, L, pre)
    eta_v = _eta_vec(n0, eta, delta)
    absu2 = np.abs(u)**2
    g = eta_v * u + beta * absu2 * u
    dg = eta_v + 2 * beta * absu2
    dg_conj = beta * u**2

    (J11, J12, J21, J22,
     dRF_dor, dRF_doi, dIF_dor, dIF_doi) = _pde_jac_blocks(K, dK, g, dg, dg_conj, u, omega, tau)

    m = 2*n + 2
    J = np.zeros((m, m))
    J[:n, :n] = J11
    J[:n, n:2*n] = J12
    J[n:2*n, :n] = J21
    J[n:2*n, n:2*n] = J22
    J[:n, 2*n] = dRF_dor
    J[:n, 2*n+1] = dRF_doi
    J[n:2*n, 2*n] = dIF_dor
    J[n:2*n, 2*n+1] = dIF_doi

    Wf = np.concatenate([pre.W, pre.W])
    J[2*n, :n] = 2 * u.real * Wf
    J[2*n, n:2*n] = 2 * u.imag * Wf
    J[2*n+1, :] = 0
    J[2*n+1, n + gauge_idx] = 1
    return J


def param_deriv_full_imperfect_N(x, p, pre, tau, eta, beta, gauge_idx):
    """Analytical dH/dN for the imperfect dimer.  H_N = [0,...,0, -1, 0]"""
    n0 = len(pre.W)
    n = 2 * n0
    m = 2 * n + 2
    Hp = np.zeros(m)
    Hp[2 * n] = -1.0
    return Hp


def param_deriv_full_imperfect_delta(x, p, pre, tau, eta, beta, gauge_idx):
    """Analytical dH/ddelta for the imperfect dimer
    eta_v = [eta(1+d), ..., eta(1-d), ...], so d(eta_v)/d(delta) = [+eta,...,-eta,...]
    dg/ddelta = d(eta_v)/d(delta) * u, and dFu/ddelta = -tau w^2 K (dg/ddelta)
    Power and gauge rows do not depend on delta
    """
    N, L, delta = p[0], p[1], p[2]
    n0 = len(pre.W)
    n = 2 * n0
    u, omega = unpack_state(x, n)

    K = build_full_K(omega, L, pre)

    # d(eta_v)/d(delta): +eta on first sphere, -eta on second
    d_eta = np.concatenate([np.full(n0, +eta), np.full(n0, -eta)])
    dg_ddelta = d_eta * u
    dFu_ddelta = -tau * omega**2 * (K @ dg_ddelta)

    m = 2 * n + 2
    Hp = np.zeros(m)
    Hp[:n] = dFu_ddelta.real
    Hp[n:2*n] = dFu_ddelta.imag
    # power row: dC/ddelta = 0
    # gauge row: dP/ddelta = 0
    return Hp


def residual_and_jacobian_full_imperfect(x, p, pre, tau, eta, beta, gauge_idx):
    """Combined residual and Jacobian builds K+dK only once"""
    N, L, delta = p[0], p[1], p[2]
    n0 = len(pre.W)
    n = 2 * n0
    u, omega = unpack_state(x, n)

    K, dK = build_full_K_and_dK(omega, L, pre)
    eta_v = _eta_vec(n0, eta, delta)
    absu2 = np.abs(u)**2
    g = eta_v * u + beta * absu2 * u

    # residual (uses K)
    Kg = K @ g
    Fu = u - tau * omega**2 * Kg
    Wf = np.concatenate([pre.W, pre.W])

    r = np.empty(2*n + 2)
    r[:n] = Fu.real
    r[n:2*n] = Fu.imag
    r[2*n] = np.sum(absu2 * Wf) - N
    r[2*n+1] = u[gauge_idx].imag

    # Jacobian (reuses K, dK, g)
    dg = eta_v + 2 * beta * absu2
    dg_conj = beta * u**2

    (J11, J12, J21, J22,
     dRF_dor, dRF_doi, dIF_dor, dIF_doi) = _pde_jac_blocks(K, dK, g, dg, dg_conj, u, omega, tau)

    m = 2*n + 2
    J = np.zeros((m, m))
    J[:n, :n] = J11
    J[:n, n:2*n] = J12
    J[n:2*n, :n] = J21
    J[n:2*n, n:2*n] = J22
    J[:n, 2*n] = dRF_dor
    J[:n, 2*n+1] = dRF_doi
    J[n:2*n, 2*n] = dIF_dor
    J[n:2*n, 2*n+1] = dIF_doi

    J[2*n, :n] = 2 * u.real * Wf
    J[2*n, n:2*n] = 2 * u.imag * Wf
    J[2*n+1, :] = 0
    J[2*n+1, n + gauge_idx] = 1
    return r, J
