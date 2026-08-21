import numpy as np
from .quadrature import KernelPrecomp

try:
    import numba as _nb
    _HAS_NUMBA = True
except ImportError:
    _nb = None
    _HAS_NUMBA = False


# (compiled only if numba present)
if _HAS_NUMBA:
    @_nb.njit(cache=True)
    def _k_self_nb(wr, wi, Rself, W, a):
        omega = wr + 1j * wi
        n = Rself.shape[0]
        K = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            z = omega * a[i]
            if abs(z) < 1e-3:
                ai = a[i]
                K[i,i] = ai*ai/2 + 1j*omega*ai**3/3 - omega**2*ai**4/8
            else:
                ai = a[i]
                e = np.exp(1j * omega * ai)
                K[i, i] = e * ai / (1j *omega) + (e -1)/omega**2
            for j in range(n):
                if i != j:
                    r = Rself[i,j]
                    K[i, j] = np.exp(1j *omega *r) / (4* np.pi* r) * W[j]
        return K

    @_nb.njit(cache=True)
    def _dk_self_nb(wr, wi, Rself, W, a):
        omega = wr + 1j *wi
        n = Rself.shape[0]
        dK = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            ai = a[i]
            z = omega * ai
            if abs(z) < 1e-3:
                dK[i,i] = 1j*ai**3/3 - omega*ai**4/4 - 1j*omega**2*ai**5/10
            else:
                e = np.exp(1j * omega * ai)
                dK[i,i] = (ai**2*e/omega +2j*ai*e/omega**2
                            - 2*(e -1)/omega**3)
            for j in range(n):
                if i != j:
                    r = Rself[i, j]
                    dK[i, j] = 1j * np.exp(1j *omega *r) / (4*np.pi) * W[j]
        return dK

    @_nb.njit(cache=True)
    def _k_cross_nb(wr, wi, L, Dx, Dy, Dz, W):
        omega = wr + 1j * wi
        n = Dx.shape[0]
        K = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                dx = Dx[i,j] - L
                r = np.sqrt(dx*dx + Dy[i,j]**2 + Dz[i,j]**2)
                K[i, j] = np.exp(1j *omega *r) / (4*np.pi *r) * W[j]
        return K

    @_nb.njit(cache=True)
    def _dk_cross_nb(wr, wi, L, Dx, Dy, Dz, W):
        omega = wr + 1j * wi
        n = Dx.shape[0]
        dK = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                dx = Dx[i,j] - L
                r = np.sqrt(dx*dx + Dy[i,j]**2 + Dz[i,j]**2)
                dK[i,j] = 1j * np.exp(1j *omega* r) /(4*np.pi) * W[j]
        return dK

    @_nb.njit(cache=True)
    def _k_and_dk_self_nb(wr, wi, Rself, W, a):
        omega = wr + 1j * wi
        n = Rself.shape[0]
        K = np.empty((n, n), dtype=np.complex128)
        dK = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            ai = a[i]
            z = omega * ai
            if abs(z) < 1e-3:
                K[i,i] = ai*ai/2 + 1j*omega*ai**3/3 - omega**2*ai**4/8
                dK[i,i] = 1j*ai**3/3 - omega*ai**4/4 - 1j*omega**2*ai**5/10
            else:
                e = np.exp(1j * omega * ai)
                em1 = e - 1
                K[i,i] = e*ai/(1j*omega) + em1/omega**2
                dK[i,i] = ai**2*e/omega + 2j*ai*e/omega**2 -2*em1/omega**3
            for j in range(n):
                if i != j:
                    r = Rself[i, j]
                    eikr = np.exp(1j *omega *r)
                    K[i,j] = eikr /(4*np.pi *r) * W[j]
                    dK[i,j] = 1j *eikr / (4*np.pi) * W[j]
        return K, dK

    @_nb.njit(cache=True)
    def _k_and_dk_cross_nb(wr, wi, L, Dx, Dy, Dz, W):
        omega = wr + 1j * wi
        n = Dx.shape[0]
        K = np.empty((n, n), dtype=np.complex128)
        dK = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                dx = Dx[i,j] - L
                r = np.sqrt(dx*dx + Dy[i,j]**2 + Dz[i,j]**2)
                eikr = np.exp(1j *omega * r)
                K[i,j] = eikr / (4*np.pi *r) * W[j]
                dK[i,j] = 1j * eikr/(4*np.pi) * W[j]
        return K, dK

    @_nb.njit(cache=True)
    def _k_cross_base_nb(wr, wi, L, Dx, Dy, Dz):
        omega = wr + 1j * wi
        n = Dx.shape[0]
        G = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                dx = Dx[i, j] - L
                r = np.sqrt(dx*dx + Dy[i,j]**2 + Dz[i,j]**2)
                G[i,j] = np.exp(1j * omega *r) / (4*np.pi *r)
        return G

    @_nb.njit(cache=True)
    def _k_and_dk_cross_base_nb(wr, wi, L, Dx, Dy, Dz):
        omega = wr + 1j * wi
        n = Dx.shape[0]
        G = np.empty((n, n), dtype=np.complex128)
        dG = np.empty((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(n):
                dx = Dx[i,j] - L
                r = np.sqrt(dx*dx + Dy[i,j]**2 + Dz[i,j]**2)
                eikr = np.exp(1j *omega * r)
                G[i,j] = eikr / (4*np.pi *r)
                dG[i,j] = 1j *eikr/(4*np.pi)
        return G, dG


#self-cell integral
def _self_integral(omega, a):
    """Integral of exp(i omega r) r dr from 0 to a"""
    z = omega * a
    if abs(z) < 1e-3:
        return a**2/2 + 1j*omega*a**3/3 - omega**2*a**4/8
    e = np.exp(1j *omega *a)
    return e *a / (1j *omega) + np.expm1(1j*omega*a)/ omega**2


def _self_integral_domega(omega, a):
    """d/d(omega) of the self-cell integral"""
    z = omega * a
    if abs(z) < 1e-3:
        return 1j*a**3/3 - omega*a**4/4 - 1j*omega**2*a**5/10
    e = np.exp(1j *omega *a)
    return a**2*e/omega + 2j*a*e/omega**2 - 2*np.expm1(1j*omega*a)/omega**3

# mirror_d2 is flag to determine if we use the mirror-symmetric geometry (which should be the case generally) or just dimers with translation
def _mirror_check(pre):
    return pre.Sx if pre.mirror_d2 else pre.Dx

#public kernel functions

def K_self(omega, pre):
    """Self-interaction kernel for one sphere"""
    if _HAS_NUMBA:
        return _k_self_nb(omega.real, omega.imag, pre.Rself, pre.W, pre.a)
    R = pre.Rself
    n0 = R.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        G = np.where(R > 0, np.exp(1j*omega*R) / (4*np.pi*R), 0j)
    K = G * pre.W[None, :]
    for i in range(n0):
        K[i, i] = _self_integral(omega, pre.a[i])
    return K


def dK_self(omega, pre):
    """omega-derivative of K_self"""
    if _HAS_NUMBA:
        return _dk_self_nb(omega.real, omega.imag, pre.Rself, pre.W, pre.a)
    R = pre.Rself
    n0 = R.shape[0]
    dG = np.where(R > 0, 1j*np.exp(1j*omega*R) / (4*np.pi), 0j)
    dK = dG * pre.W[None, :]
    for i in range(n0):
        dK[i, i] = _self_integral_domega(omega, pre.a[i])
    return dK


def K_cross(omega, L, pre):
    """Cross-sphere kernel (sphere 1 sources, sphere 2 observation, separated by L along x)"""
    if _HAS_NUMBA:
        return _k_cross_nb(omega.real, omega.imag, L, _mirror_check(pre), pre.Dy, pre.Dz, pre.W)
    dx = _mirror_check(pre) - L
    R = np.sqrt(dx**2 + pre.Dy**2 + pre.Dz**2)
    return np.exp(1j*omega*R) / (4*np.pi*R) * pre.W[None, :]


def dK_cross(omega, L, pre):
    """omega-derivative of K_cross"""
    if _HAS_NUMBA:
        return _dk_cross_nb(omega.real, omega.imag, L, _mirror_check(pre), pre.Dy, pre.Dz, pre.W)
    dx = _mirror_check(pre) - L
    R = np.sqrt(dx**2 + pre.Dy**2 + pre.Dz**2)
    return 1j * np.exp(1j*omega*R) / (4*np.pi) * pre.W[None, :]


def build_full_K(omega, L, pre):
    """Full 2n0 x 2n0 dimer kernel [[K11, K12], [K21, K11]]
    Cross blocks share one base matrix: K21[i,j] = G[j,i] * W[j]"""
    n0 = len(pre.W)
    K11 = K_self(omega, pre)
    if _HAS_NUMBA:
        G = _k_cross_base_nb(omega.real, omega.imag, L, _mirror_check(pre), pre.Dy, pre.Dz)
    else:
        dx = _mirror_check(pre) - L
        R = np.sqrt(dx**2 + pre.Dy**2 + pre.Dz**2)
        G = np.exp(1j * omega * R) / (4 * np.pi * R)
    W = pre.W[None, :]
    K = np.empty((2*n0, 2*n0), dtype=np.complex128)
    K[:n0, :n0] = K11
    K[:n0, n0:] = G * W
    K[n0:, :n0] = G.T * W
    K[n0:, n0:] = K11
    return K


def build_full_K_and_dK(omega, L, pre):
    """K and dK/domega for the full dimer, fused where possible
    Cross blocks derived from one unweighted base matrix"""
    n0 = len(pre.W)
    if _HAS_NUMBA:
        K11, dK11 = _k_and_dk_self_nb(omega.real, omega.imag, pre.Rself, pre.W, pre.a)
        G, dG = _k_and_dk_cross_base_nb(omega.real, omega.imag, L, _mirror_check(pre), pre.Dy, pre.Dz)
    else:
        K11 = K_self(omega, pre)
        dK11 = dK_self(omega, pre)
        dx = _mirror_check(pre) - L
        R = np.sqrt(dx**2 + pre.Dy**2 + pre.Dz**2)
        eikr = np.exp(1j * omega * R)
        G = eikr / (4 * np.pi * R)
        dG = 1j * eikr / (4 * np.pi)
    W = pre.W[None, :]
    K = np.empty((2*n0, 2*n0), dtype=np.complex128)
    K[:n0, :n0] = K11
    K[:n0, n0:] = G * W
    K[n0:, :n0] = G.T * W
    K[n0:, n0:] = K11
    dK = np.empty((2*n0, 2*n0), dtype=np.complex128)
    dK[:n0, :n0] = dK11
    dK[:n0, n0:] = dG * W
    dK[n0:, :n0] = dG.T * W
    dK[n0:, n0:] = dK11
    return K, dK

def evaluate_volume_potential(omega, pre, density, x_eval):
    """
    Evaluate the volume potential:
        u(x) = sum_j G^omega(x - X0_j) * density_j * W_j
    at exterior points x_eval for dimension 3.

    Return u_ext : (M,) complex
    """
    if x_eval.shape[0] != pre.dim:
        raise ValueError(f"x_eval shape {x_eval.shape} incompatible with pre.dim={pre.dim}")
    dx = x_eval[0, :, None] - pre.X0[0, None, :]
    dy = x_eval[1, :, None] - pre.X0[1, None, :]
    dz = x_eval[2, :, None] - pre.X0[2, None, :]
    R = np.sqrt(dx**2 + dy**2 + dz**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        G = np.where(R > 0, np.exp(1j * omega * R) / (4 * np.pi * R), 0j)
    # weighted contraction:  u_ext[i] = sum_j G[i, j] * W[j] * density[j]
    return G @ (pre.W * density)
