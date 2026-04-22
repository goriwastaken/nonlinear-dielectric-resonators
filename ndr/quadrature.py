# vol quad on sphere + kernel precomputation

import numpy as np
from dataclasses import dataclass
from numpy.polynomial.legendre import leggauss


def volume_quadrature_sphere(R, n_radial=6, n_mu=6, n_phi=8):
    """
    Tensor-product quadrature on a ball of radius R
    Gauss-Legendre in r and cos(theta), trapezoidal in phi
    Returns (X, W, a) where X is (3, n), W are weights, a are cell radii
    """
    xi_r, w_r = leggauss(n_radial)
    r = (R / 2) * (xi_r + 1)
    dr_dxi = R / 2

    mu, w_mu = leggauss(n_mu)

    phi = np.linspace(0, 2*np.pi, n_phi + 1)[:-1]
    w_phi = 2 * np.pi / n_phi

    n = n_radial * n_mu * n_phi
    X = np.empty((3, n))
    W = np.empty(n)

    k = 0
    for ir in range(n_radial):
        ri = r[ir]
        for im in range(n_mu):
            sin_th = np.sqrt(max(0, 1 - mu[im]**2))
            for ip in range(n_phi):
                X[0, k] = ri * sin_th * np.cos(phi[ip])
                X[1, k] = ri * sin_th * np.sin(phi[ip])
                X[2, k] = ri * mu[im]
                W[k] = w_r[ir] * dr_dxi * ri**2 * w_mu[im] * w_phi
                k += 1

    # equivalent cell radius: (4/3) pi a^3 = W
    a = (3 * W / (4 * np.pi))**(1/3)
    return X, W, a


@dataclass
class KernelPrecomp:
    """Precomputed geometry for one sphere centred at the origin"""
    X0: np.ndarray# (3, n0) quad points
    W: np.ndarray# (n0,) quad weights
    a: np.ndarray# (n0,) cell radii
    Rself: np.ndarray # (n0, n0) pairwise distances
    Dx: np.ndarray# (n0, n0) x-diffs
    Dy: np.ndarray
    Dz: np.ndarray


def precompute_kernel(X0, W, a):
    """Build KernelPrecomp from quadrature nodes"""
    Dx = X0[0, :, None] - X0[0, None, :]
    Dy = X0[1, :, None] - X0[1, None, :]
    Dz = X0[2, :, None] - X0[2, None, :]
    Rself = np.sqrt(Dx**2 + Dy**2 + Dz**2)
    return KernelPrecomp(X0=X0, W=W, a=a, Rself=Rself, Dx=Dx, Dy=Dy, Dz=Dz)
