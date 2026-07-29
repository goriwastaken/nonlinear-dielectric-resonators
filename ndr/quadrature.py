# Volume quadrature for 3D resonators, plus kernel precomputation

import numpy as np
from dataclasses import dataclass
from numpy.polynomial.legendre import leggauss


def _radial_nodes(R, n_radial, radial_map, radial_beta):
    # GL nodes on [0, R],  "tanh" clusters them toward r = R, beta ~ |ln(d/R)| resolves a layer of thickness d.
    xi, w = leggauss(n_radial)
    if radial_map == "uniform":
        r = (R / 2) * (xi + 1)
        dr_dxi = np.full_like(xi, R / 2)
    elif radial_map == "tanh":
        eta = 0.5 * (xi + 1.0)
        denom = np.tanh(radial_beta)
        r = R * np.tanh(radial_beta * eta) / denom
        dr_dxi = (R / 2.0) * radial_beta / (np.cosh(radial_beta * eta) ** 2 * denom)
    else:
        raise ValueError(f"unknown radial_map: {radial_map!r}")
    return r, w * dr_dxi


def volume_quadrature_sphere(R, n_radial=6, n_mu=6, n_phi=8,
                             polar_axis="z", radial_map="uniform",
                             radial_beta=2.0):
    """
    Tensor quadrature on a ball of radius R,  polar_axis="x" puts the mu clustering at the +-x
    poles (gap faces of an x-aligned dimer); radial_map="tanh" clusters r
    toward the boundary.  return nodes X (3, n), weights W, and cell radii
    a = (3W/4pi)^(1/3)
    """
    r, wr = _radial_nodes(R, n_radial, radial_map, radial_beta)
    mu, w_mu = leggauss(n_mu)
    phi = np.linspace(0, 2 * np.pi, n_phi + 1)[:-1]
    w_phi = 2 * np.pi / n_phi
    n = n_radial * n_mu * n_phi
    X = np.empty((3, n))
    W = np.empty(n)

    if polar_axis not in ("x", "z"):
        raise ValueError(f"polar_axis must be 'x' or 'z', got {polar_axis!r}")

    k = 0
    for ir in range(n_radial):
        ri = r[ir]
        for im in range(n_mu):
            sin_th = np.sqrt(max(0.0, 1.0 - mu[im] ** 2))
            for ip in range(n_phi):
                cphi = np.cos(phi[ip])
                sphi = np.sin(phi[ip])
                if polar_axis == "z":
                    X[0, k] = ri * sin_th * cphi
                    X[1, k] = ri * sin_th * sphi
                    X[2, k] = ri * mu[im]
                else:  # "x"
                    X[0, k] = ri * mu[im]
                    X[1, k] = ri * sin_th * cphi
                    X[2, k] = ri * sin_th * sphi
                W[k] = wr[ir] * ri ** 2 * w_mu[im] * w_phi
                k += 1

    a = (3 * W / (4 * np.pi)) ** (1 / 3)
    return X, W, a


def volume_quadrature_ellipsoid(semi_axes, n_radial=6, n_mu=8, n_phi=12,
                                polar_axis="z", radial_map="uniform",
                                radial_beta=2.0):
    """
    Quadrature on ellipsoid {(x/a)^2 + (y/b)^2 + (z/c)^2 <= 1},
    semi_axes = (a, b, c): unit-ball nodes scaled by diag(a, b, c),
    weights by abc.  Returns (X, W, a) like volume_quadrature_sphere
    """
    ax, ay, az = semi_axes
    Xu, Wu, _ = volume_quadrature_sphere(1.0, n_radial, n_mu, n_phi,
                                         polar_axis, radial_map, radial_beta)
    X = np.vstack([ax * Xu[0], ay * Xu[1], az * Xu[2]])
    W = (ax * ay * az) * Wu
    a = (3 * W / (4 * np.pi)) ** (1 / 3)
    return X, W, a



def volume_quadrature_peanut(a, b, R_curv, n_radial=6, n_mu=8, n_phi=12,
                             theta0=np.pi / 2, radial_map="uniform", radial_beta=2.0):
    """
    Bent sphere (peanut): {(x/a)^2 + (y/b)^2 +
    (z/b)^2 <= 1}, a > b, wrapped onto an arc of radius R_curv in the x-y
    plane (x -> arc angle, y -> radius; bend Jacobian (R_curv+y)/R_curv)
    Recentred on the centroid and returns (X, W, a) like
    volume_quadrature_sphere
    """
    if a <= b:
        raise ValueError("peanut needs a > b (semi-major along the arc)")
    if R_curv <= b:
        raise ValueError("need R_curv > b so the bent radius stays positive")
    Xu, Wu, _ = volume_quadrature_sphere(1.0, n_radial, n_mu, n_phi,
                                         polar_axis="x", radial_map=radial_map,
                                         radial_beta=radial_beta)
    xs, ys, zs = a * Xu[0], b * Xu[1], b * Xu[2]
    psi = theta0 + xs / R_curv
    rho = R_curv + ys
    X = np.vstack([rho * np.cos(psi), rho * np.sin(psi), zs])
    W = Wu * (a * b * b) * (rho / R_curv)              # stretch a*b^2 x bend rho/R
    X = X - (X * W).sum(axis=1, keepdims=True) / W.sum()   # recentre on centroid
    a_cell = (3 * W / (4 * np.pi)) ** (1 / 3)
    return X, W, a_cell


@dataclass
class KernelPrecomp:
    """
    precomp geometry for one resonator centred at the origin
    X0: (3, n0) quad pts
    W: (n0,) quad weights
    a: (n0,) cell radiuses
    Rself: (n0, n0) pairwise distances within  reso
    Dx, Dy, Dz: (n0, n0) per-coordinate differences (used for K_cross blocks)
    mirror_d2 : if True, D_2 is the reflection of D_1 through the gap plane
                x = L/2 rather than original idea to translate by L e_x; node j of D_2 sits
                at (L - X0[0,j], X0[1,j], X0[2,j]).  Theta clustering toward
                the gap face of D_1 then lands on the gap face of D_2 as well
    Sx     : (n0, n0) x-sums X0[0,i] + X0[0,j], used by the mirrored-D2
             cross block
    """
    X0: np.ndarray
    W: np.ndarray
    a: np.ndarray
    Rself: np.ndarray
    Dx: np.ndarray
    Dy: np.ndarray
    Dz: np.ndarray
    dim: int = 3
    mirror_d2: bool = False
    Sx: "np.ndarray | None" = None


def precompute_kernel(X0, W, a, mirror_d2=False):
    """
    Build KernelPrecomp from quadrature nodes
    """
    if X0.shape[0] != 3:
        raise ValueError(f"expected 3D nodes, got X0 with shape {X0.shape}")
    Dx = X0[0, :, None] - X0[0, None, :]
    Dy = X0[1, :, None] - X0[1, None, :]
    Dz = X0[2, :, None] - X0[2, None, :]
    Sx = X0[0, :, None] + X0[0, None, :]
    Rself = np.sqrt(Dx**2 + Dy**2 + Dz**2)
    return KernelPrecomp(X0=X0, W=W, a=a, Rself=Rself, Dx=Dx, Dy=Dy, Dz=Dz,
                         mirror_d2=mirror_d2, Sx=Sx)
