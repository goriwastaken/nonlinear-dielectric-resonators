"""Leading-order single-resonator asymptotics for different shapes
(ball, ellipsoid, bent peanut).

Reference: [AL26] H. Ammari and B. Li. Dielectric scattering resonances for high-refractive
resonators with cubic nonlinearity. J. Differential Equations 475 (2026),
Paper No. 114459, 56.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ndr import (
    volume_quadrature_sphere, volume_quadrature_ellipsoid, volume_quadrature_peanut,
    precompute_kernel, newton_solve,
    pack_state, unpack_state, normalize_power, enforce_component_gauge,
)
from ndr.kernel import K_self, dK_self
from ndr.residuals import _kerr_source, _pde_jac_blocks

R = 1.0
TAU = 500.0
ETA = 1.0
BETA = 1.0
QUAD = (7, 9, 14)
OUTDIR = os.path.join(ROOT, "output_shape_asymptotics")


# single-resonator Newton solver (K_self only) solves the nonlinear Lippmann-Schwinger equation with the normalization constraint and the phase constr

def residual_single(x, N, pre, tau, eta, beta, gi):
    n0 = len(pre.W)
    u, om = unpack_state(x, n0)
    g, _, _ = _kerr_source(u, eta, beta)
    Fu = u - tau * om**2 * (K_self(om, pre) @ g)
    r = np.empty(2 * n0 + 2)
    r[:n0] = Fu.real
    r[n0:2*n0] = Fu.imag
    r[2*n0] = np.sum(np.abs(u)**2 * pre.W) - N
    r[2*n0+1] = u[gi].imag
    return r


def jacobian_single(x, N, pre, tau, eta, beta, gi):
    n0 = len(pre.W)
    u, om = unpack_state(x, n0)
    g, dg, dgc = _kerr_source(u, eta, beta)
    (J11, J12, J21, J22, dRr, dRi, dIr, dIi) = _pde_jac_blocks(
        K_self(om, pre), dK_self(om, pre), g, dg, dgc, u, om, tau)
    m = 2 * n0 + 2
    J = np.zeros((m, m))
    J[:n0, :n0] = J11; J[:n0, n0:2*n0] = J12
    J[n0:2*n0, :n0] = J21; J[n0:2*n0, n0:2*n0] = J22
    J[:n0, 2*n0] = dRr; J[:n0, 2*n0+1] = dRi
    J[n0:2*n0, 2*n0] = dIr; J[n0:2*n0, 2*n0+1] = dIi
    J[2*n0, :n0] = 2 * u.real * pre.W
    J[2*n0, n0:2*n0] = 2 * u.imag * pre.W
    J[2*n0+1, n0 + gi] = 1.0
    return J


def linear_eig(pre, tau, eta=ETA):
    # principal eigenpair of the Newtonian potential K_D
    # phi normalized in W, linear resonance om = 1/sqrt(tau lam0)
    W = pre.W; sq = np.sqrt(W); isq = 1.0 / sq
    A = (sq[:, None] * K_self(0j, pre).real) * isq[None, :]
    A = 0.5 * (A + A.T)
    vals, vecs = eigh(A)
    i = int(np.argmax(vals))
    lam0 = vals[i]
    phi = isq * vecs[:, i]
    if np.sum(phi * W) < 0:
        phi = -phi
    phi /= np.sqrt(np.sum(phi**2 * W))
    return 1.0 / np.sqrt(tau * eta * lam0), phi.astype(complex), lam0


def solve_single(pre, tau, N, eta, beta, om0, u0, tol=1e-11, max_iter=40):
    u = u0.copy()
    normalize_power(u, pre.W, N)
    gi = int(np.argmax(np.abs(u)))
    enforce_component_gauge(u, gi)
    res = newton_solve(
        lambda x: residual_single(x, N, pre, tau, eta, beta, gi),
        lambda x: jacobian_single(x, N, pre, tau, eta, beta, gi),
        pack_state(u, om0), tol=tol, max_iter=max_iter, linesearch=True)
    u_s, om_s = unpack_state(res.x, len(pre.W))
    return om_s, u_s, res.converged


def continue_in_N(pre, tau, N_values, eta, beta, om0, u0):
    oms, flags, om, u = [], [], om0, u0.copy()
    for N in N_values:
        om, u, conv = solve_single(pre, tau, N, eta, beta, om, u)
        oms.append(om); flags.append(conv)
    return np.array(oms), np.array(flags)


# leading-order predictions

def leading_order(pre, tau, eta=ETA):
    om_lin, phi, lam0 = linear_eig(pre, tau, eta)
    A4 = float(np.sum(np.abs(phi)**4 * pre.W))   # int |phi0|^4
    M = float(np.sum(phi.real * pre.W))          # int phi0
    return dict(
        om_lin=om_lin, lam0=lam0, phi=phi, A4=A4, M=M,
        # Kerr shift of Re(om) with N: from Eq 3.20 with N ~ a^2 (Remark 2, AL26)
        slope=-(A4 / 2.0) * om_lin,
        # term |Im om| = M^2/(8 pi lam0^2 tau) (Cor 3.4, AL26)
        im_const=M**2 / (8 * np.pi * lam0**2 * tau))


#shapes
ELLIP_AXES = (1.5 * R, 0.8 * R, 0.8 * R)
PEANUT_A, PEANUT_B, PEANUT_RCURV = 1.3 * R, 0.45 * R, 1.5 * R

SHAPES = ["sphere", "ellipsoid", "peanut"]
LABELS = {"sphere": "ball", "ellipsoid": "ellipsoid",
          "peanut": "peanut"}
COLOR = {"sphere": "tab:blue", "ellipsoid": "tab:orange", "peanut": "tab:green"}
MARKER = {"sphere": "o", "ellipsoid": "s", "peanut": "^"}


def shape_pre(name, quad=QUAD):
    nr, nm, nphi = quad
    if name == "sphere":
        X, W, a = volume_quadrature_sphere(R, nr, nm, nphi, polar_axis="x")
    elif name == "ellipsoid":
        X, W, a = volume_quadrature_ellipsoid(ELLIP_AXES, nr, nm, nphi, polar_axis="x")
    elif name == "peanut":
        X, W, a = volume_quadrature_peanut(PEANUT_A, PEANUT_B, PEANUT_RCURV,
                                           nr, nm, nphi)
    else:
        raise ValueError(name)
    return precompute_kernel(X, W, a)


def shape_outline(name):
    # z=0 slice
    t = np.linspace(0, 2 * np.pi, 240)
    if name == "sphere":
        return R * np.cos(t), R * np.sin(t)
    if name == "ellipsoid":
        return ELLIP_AXES[0] * np.cos(t), ELLIP_AXES[1] * np.sin(t)
    if name == "peanut":
        xs, ys = PEANUT_A * np.cos(t), PEANUT_B * np.sin(t)
        rho = PEANUT_RCURV + ys
        psi = np.pi / 2 + xs / PEANUT_RCURV
        ox, oy = rho * np.cos(psi), rho * np.sin(psi)
        return ox - ox.mean(), oy - oy.mean()
    raise ValueError(name)


def compute_shape(name, tau=TAU, eta=ETA, beta=BETA, quad=QUAD):
    pre = shape_pre(name, quad)
    P = leading_order(pre, tau, eta)
    n0 = len(pre.W)
    print(f"\n[{name}], quad points: {n0}")

    # nonlinear dispersion Re(om) vs N at fixed tau
    N_disp = np.linspace(0.01, 2.5, 24)
    om0 = complex(P["om_lin"], -0.02 * P["om_lin"])
    om_disp, disp_ok = continue_in_N(pre, tau, N_disp, eta, beta, om0, P["phi"])
    # leading-order slope on a small-N window
    N_sl = np.array([0.005, 0.01, 0.02, 0.03, 0.05])
    om_sl, _ = continue_in_N(pre, tau, N_sl, eta, beta, om0, P["phi"])
    slope_num = float(np.polyfit(N_sl, om_sl.real, 1)[0])

    # contrast sweep at fixed small N, in the scaled frequency om_hat =
    # sqrt(tau)*omega (Eq 2.22, AL26): Re om_hat -> 1/sqrt(lam0) (Prop 2.7) and
    # |Im om_hat| ~ (M^2/8pi lam0^2)*eps (Cor 3.4)
    taus = np.array([50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 25000], float)
    om_tau = []
    for t in taus:
        ol, ph, _ = linear_eig(pre, t, eta)
        om, _, _ = solve_single(pre, t, 0.01, eta, beta, complex(ol, -0.02 * ol), ph)
        om_tau.append(om)
    om_tau = np.array(om_tau)

    return dict(name=name, n0=n0, pre=pre, phi=P["phi"],
                om_lin=P["om_lin"], lam0=P["lam0"], A4=P["A4"], M=P["M"],
                slope_pred=P["slope"], slope_num=slope_num,
                im_pred=P["im_const"], im_num=float(abs(om_disp.imag[0])),
                N_disp=N_disp, om_disp=om_disp, disp_ok=disp_ok,
                taus=taus, om_tau=om_tau)


# plotting, all shapes overlaid

def _shape_legend(ax, loc, lo_label="leading order"):
    h = [Line2D([0], [0], color=COLOR[n], marker=MARKER[n], ls="none", ms=6,
                label=LABELS[n]) for n in SHAPES]
    h.append(Line2D([0], [0], color="k", ls=":", lw=1.4, label=lo_label))
    ax.legend(handles=h, fontsize=8, loc=loc)


def _shape_inset(ax, box):
    ins = ax.inset_axes(box)
    for n in SHAPES:
        x, y = shape_outline(n)
        ins.plot(x, y, "-", color=COLOR[n], lw=1.4)
    ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([])
    ins.margins(0.12)
    for s in ins.spines.values():
        s.set_edgecolor("0.6")


def plot_dispersion(results):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ylo, yhi = np.inf, -np.inf
    for d in results:
        ok = d["disp_ok"] & np.isfinite(d["om_disp"].real) & (d["om_disp"].real > 0)
        N, re = d["N_disp"][ok], d["om_disp"].real[ok]
        ax.plot(N, re, MARKER[d["name"]], color=COLOR[d["name"]], ms=4)
        ax.plot(N, d["om_lin"] + d["slope_pred"] * N, ":", color="k", lw=1.2)
        ylo, yhi = min(ylo, re.min()), max(yhi, d["om_lin"])
    ax.set(xlabel=r"normalization $\mathcal{N}$", ylabel=r"$\mathrm{Re}\,\omega$")
    ax.set_ylim(ylo - 0.008, yhi + 0.008)
    ax.grid(alpha=0.3)
    _shape_legend(ax, "lower left",
                  r"$\omega_*(1-\frac{1}{2}\int_D|\phi_j|^4\mathrm{d}x\,\mathcal{N})$")
    _shape_inset(ax, [0.66, 0.62, 0.30, 0.32])
    fig.tight_layout()
    path = os.path.join(OUTDIR, "asymptotics_dispersion.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_re_vs_contrast(results):
    # Re om_hat flattens to the constant om_hat_j = 1/sqrt(lam0) (Prop 2.7, AL26)
    taus = results[0]["taus"]
    eps = 1.0 / np.sqrt(taus)
    el = np.linspace(0, eps.max() * 1.03, 100)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    lv = []
    for d in results:
        wj = 1.0 / np.sqrt(d["lam0"])
        ax.plot(eps, d["om_tau"].real * np.sqrt(taus), MARKER[d["name"]],
                color=COLOR[d["name"]], ms=5)
        ax.plot(el, np.full_like(el, wj), ":", color="k", lw=1.2)
        lv.append(wj)
    ax.set(xlabel=r"$\varepsilon=1/\sqrt{\tau}$",
           ylabel=r"$\mathrm{Re}\,\hat\omega=\sqrt{\tau}\,\mathrm{Re}\,\omega$")
    ax.set_xlim(left=0.0)
    ax.set_ylim(min(lv) - 0.2, max(lv) + 0.2)
    ax.grid(alpha=0.3)
    _shape_legend(ax, "center left", r"slope $\hat{\omega}_j=1/\sqrt{\lambda_j}$")
    _shape_inset(ax, [0.63, 0.40, 0.33, 0.30])
    fig.tight_layout()
    path = os.path.join(OUTDIR, "asymptotics_re_vs_contrast.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")



def plot_im_vs_contrast(results):
    taus = results[0]["taus"]
    eps = 1.0 / np.sqrt(taus)
    el = np.linspace(0, eps.max() * 1.03, 100)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for d in results:
        slope = d["M"]**2 / (8 * np.pi * d["lam0"]**2)   # = o_j^4 int(phi0)^2 / 8pi
        ax.plot(eps, np.abs(d["om_tau"].imag) * np.sqrt(taus), MARKER[d["name"]],
                color=COLOR[d["name"]], ms=5)
        ax.plot(el, slope * el, ":", color="k", lw=1.2)
    ax.set(xlabel=r"$\varepsilon=1/\sqrt{\tau}$",
           ylabel=r"$|\mathrm{Im}\,\hat\omega|=\sqrt{\tau}\,|\mathrm{Im}\,\omega|$")
    ax.set_xlim(left=0.0); ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    _shape_legend(ax, "upper left",
                  r"slope $\hat{\omega}_j^{4}(\int_D\phi_j\,\mathrm{d}x)^2/8\pi$")
    _shape_inset(ax, [0.64, 0.12, 0.32, 0.32])
    fig.tight_layout()
    path = os.path.join(OUTDIR, "asymptotics_im_vs_contrast.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_mode_gallery(results):
    fig, ax = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.8))
    if len(results) == 1:
        ax = [ax]
    for a, d in zip(ax, results):
        X = d["pre"].X0
        sl = np.abs(X[2]) < 1e-9            # z=0 slice
        sc = a.scatter(X[0, sl], X[1, sl], c=np.abs(d["phi"][sl])**2, s=14, cmap="viridis")
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel(LABELS[d["name"]], fontsize=9)
        fig.colorbar(sc, ax=a, fraction=0.046, label=r"$|\varphi_0|^2$")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "mode_gallery.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")



def main():
    os.makedirs(OUTDIR, exist_ok=True)
    results = [compute_shape(name) for name in SHAPES]
    plot_dispersion(results)
    plot_re_vs_contrast(results)
    plot_im_vs_contrast(results)
    # plot_mode_gallery(results)


if __name__ == "__main__":
    main()
