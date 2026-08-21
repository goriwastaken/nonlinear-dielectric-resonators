"""
Nonlinear LS equation under a plane wave
"""
import sys
import os
from pathlib import Path
import numpy as np
import numpy.linalg as nla
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ndr import volume_quadrature_sphere, precompute_kernel, K_self, evaluate_volume_potential, newton_solve, palc_continue

R = 1.0
ETA = 1.0# Kerr coefficient

OUT = Path(ROOT)/"output_inc_field"
OUT.mkdir(exist_ok=True)

X0, W, a_cells = volume_quadrature_sphere(R, 9, 9, 12)
PRE = precompute_kernel(X0, W, a_cells)
N0 = len(W)

# principal Newtonian eigenpair (lam0, phi0), <phi0,phi0>_W = 1
sq = np.sqrt(W)
A = (sq[:, None] * K_self(0j, PRE).real) / sq[None, :]
vals, vecs = eigh(0.5 * (A + A.T))
LAM0 = float(vals[-1])
PHI0 = vecs[:, -1] / sq
if PHI0.sum() < 0:
    PHI0 = -PHI0
PHI0 /= np.sqrt(np.sum(W * PHI0 ** 2))
MJ = float(np.sum(W * PHI0))
print(f"nodes {N0}, lam0 {LAM0:.5f}")


def KM(om, tau):
    """K(om), M(om) of the amplitude equation (K-eta M |a|^2)a = f"""
    Kw = K_self(complex(om), PRE)
    c = tau * om ** 2
    return (1.0 - c * np.sum(W * PHI0 * (Kw @ PHI0)),
            c * np.sum(W * PHI0 * (Kw @ (PHI0 ** 2 * PHI0))))


def a_theory(Kc, Mc, f, I_prev=None):
    """a = f/(K - eta M I), I:=abs(a)^2  smallest positive root of I|K - eta M I|^2 = |f|^2(or the root nearest I_prev)."""
    em = ETA * Mc
    r = np.roots([abs(em) ** 2, -2.0 * (Kc * np.conj(em)).real,
                  abs(Kc) ** 2, -abs(f) ** 2])
    r = r.real[np.abs(r.imag) <= 1e-8 * (1.0 + np.abs(r.real))]
    r = np.sort(r[r > 0.0])
    if len(r) == 0:
        return 0j
    I = r[0] if I_prev is None or len(r) == 1 else r[np.argmin(np.abs(r - I_prev))]
    return f / (Kc - em * I)


def forced_system(om, tau):
    """F(x, alpha), J(x, alpha) of  u = alpha e^{i om z} + tau om^2 K^om[u + eta|u|^2 u] where x = (Re u, Im u)"""
    Kw = K_self(complex(om), PRE)
    c = tau * om ** 2
    einc = np.exp(1j * om * X0[2])

    def F(x, alpha):
        u = x[:N0] + 1j * x[N0:]
        Fu = u - alpha * einc - c * (Kw @ (u + ETA * np.abs(u) ** 2 * u))
        return np.concatenate([Fu.real, Fu.imag])

    def J(x, alpha):
        u = x[:N0] + 1j * x[N0:]
        M1 = np.eye(N0, dtype=complex) - c * (Kw * (1 + 2 * ETA * np.abs(u) ** 2)[None, :])
        M2 = -c * (Kw * (ETA * u ** 2)[None, :])
        Ar, Br = M1 + M2, M1 - M2
        Jm = np.empty((2 * N0, 2 * N0))
        Jm[:N0, :N0] = Ar.real; Jm[:N0, N0:] = -Br.imag
        Jm[N0:, :N0] = Ar.imag; Jm[N0:, N0:] = Br.real
        return Jm

    return F, J, einc


def solve_forced(om, alpha, tau, x0):
    """Newton from x0, returns (u, x)"""
    F, J, _ = forced_system(om, tau)
    res = newton_solve(lambda x: F(x, alpha), lambda x: J(x, alpha), x0,
                       tol=1e-11, max_iter=60, linesearch=True,
                       feasible=lambda x: True)
    if not res.converged:
        return None, x0
    return res.x[:N0] + 1j * res.x[N0:], res.x


def line_sweep(tau=200.0, a_peak=0.1, n_pts=61, half_width=6.0):
    """EXperim 1 numerical sol u_h  vs |a_th|^2."""
    om_j = 1.0 / np.sqrt(tau * LAM0)
    Gam = MJ ** 2 / (8.0 * np.pi * LAM0 ** 2 * tau)
    Kc0, _ = KM(om_j, tau)
    alpha = a_peak * abs(Kc0) / abs(np.sum(W * PHI0 * np.exp(1j * om_j * X0[2])))
    oms = om_j + Gam * np.linspace(-half_width, half_width, n_pts)
    x = np.zeros(2 * N0)
    rows, I_prev = [], None
    for om in oms:
        u, x = solve_forced(om, alpha, tau, x)
        if u is None:
            continue
        Kc, Mc = KM(om, tau)
        a_th = a_theory(Kc, Mc, alpha * np.sum(W * PHI0 * np.exp(1j * om * X0[2])), I_prev)
        I_prev = abs(a_th) ** 2
        rows.append(((om - om_j) / Gam, np.sum(np.abs(u) ** 2 * W), abs(a_th) ** 2))
    r = np.array(rows)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(r[:, 0], r[:, 1], "o", ms=4, label=r"$\|u_h\|_W^2$ numerical")
    ax.plot(r[:, 0], r[:, 2], "-", lw=1.3, color="k",
            label=r"$|a_{\rm th}|^2$ amplitude eq.")
    ax.set(xlabel=r"$(\hat\omega-\hat\omega_j)/\Gamma$", ylabel="intensity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_line_sweep.png", dpi=150, bbox_inches="tight")
    print("wrote fig1_line_sweep.png")


def remainder_scaling(taus=(200.0, 1000.0, 5000.0, 25000.0, 125000.0),
                      a_small=0.01, tau_pair=(200.0, 5000.0), n_amp=10,
                      a_max=1.2, xi_gen=3.0):
    """Experiment 2: check ||W_h||_W / |a_h|  (a) fixed |a|, sweep tau at xi = 0 and
    xi = xi_gen;  (b) fixed tau, xi = 0, sweep amplitude."""
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4))

    for xi, mark, p, ls, glab in ((0.0, "o", -1.0, "--", r"$\propto\tau^{-1}$"),
                                  (xi_gen, "s", -0.5, ":", r"$\propto\tau^{-1/2}$")):
        rows = []
        for tau in taus:
            om_j = 1.0 / np.sqrt(tau * LAM0)
            Gam = MJ ** 2 / (8.0 * np.pi * LAM0 ** 2 * tau)
            om = om_j + xi * Gam
            Kc0, _ = KM(om, tau)
            fc = abs(np.sum(W * PHI0 * np.exp(1j * om * X0[2])))
            u, _ = solve_forced(om, a_small * abs(Kc0) / fc, tau,
                                np.zeros(2 * N0))
            if u is None:
                continue
            a = np.sum(W * PHI0 * u)
            E = np.sqrt(np.sum(np.abs(u - a * PHI0) ** 2 * W)) / abs(a)
            rows.append((tau, E))
        A = np.array(rows)
        ax[0].loglog(A[:, 0], A[:, 1], mark, ms=5,
                     label=rf"$(\hat\omega-\hat\omega_j)/\Gamma={xi:g}$")
        ax[0].loglog(A[:, 0], A[0, 1] * (A[:, 0] / A[0, 0]) ** p, ls,
                     color="k", lw=1.0, label=glab)
    ax[0].set(xlabel=r"$\tau$", ylabel=r"$\|W_h\|_W\,/\,|a_h|$")

    curves = []
    for tau in tau_pair:
        om_j = 1.0 / np.sqrt(tau * LAM0)
        Kc0, Mc0 = KM(om_j, tau)
        fc = abs(np.sum(W * PHI0 * np.exp(1j * om_j * X0[2])))
        x = np.zeros(2 * N0)
        pts = []
        for at in np.geomspace(a_small, a_max, n_amp):
            u, x = solve_forced(om_j, at * abs(Kc0 - ETA * Mc0 * at ** 2) / fc,
                                tau, x)
            if u is None:
                continue
            a = np.sum(W * PHI0 * u)
            wn = np.sqrt(np.sum(np.abs(u - a * PHI0) ** 2 * W))
            pts.append((abs(a), wn / abs(a)))
        curves.append((tau, np.array(pts)))

    # joint leastsquares fit C1 eps^2 + C2 eta |a|^2 over both xi = 0 curves
    D, y = [], []
    for tau, pts in curves:
        D.append(np.column_stack([np.full(len(pts), 1.0 / tau),
                                  ETA * pts[:, 0] ** 2]))
        y.append(pts[:, 1])
    (C1, C2), *_ = nla.lstsq(np.vstack(D), np.concatenate(y), rcond=None)

    for tau, pts in curves:
        al = np.geomspace(pts[0, 0], pts[-1, 0], 100)
        ln, = ax[1].loglog(pts[:, 0], pts[:, 1], "o", ms=5,
                           label=rf"$\tau={tau:.0f}$")
        ax[1].loglog(al, C1 / tau + C2 * ETA * al ** 2, "--", lw=1.0,
                     color=ln.get_color())
    ax[1].set(xlabel=r"$|a_h|$", ylabel=r"$\|W_h\|_W\,/\,|a_h|$")
    for a_ in ax:
        a_.grid(alpha=0.3, which="both")
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_remainder_scaling.png", dpi=150, bbox_inches="tight")
    print("wrote fig2_remainder_scaling.png")


def scattered_comparison(taus=(200.0, 1000.0, 5000.0, 25000.0, 125000.0),
                         xi_fix=3.0, a_t=0.01, r1=1.5, r2=2.5, n_r=3, m_ang=200):
    """experimnt 3: tau -> inf at fixed (om-om_j)/Gamma.  Plot errors
    inside D and (exterior) on the annulus r1 <= |x| <= r2, and the amplitude-equation error."""
    # annulus quadrature
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    rs = 0.5 * (r2 + r1) + 0.5 * (r2 - r1) * xr
    ws = 0.5 * (r2 - r1) * wr
    k = np.arange(m_ang) + 0.5
    th = np.arccos(1.0 - 2.0 * k / m_ang)
    ph = np.pi * (1.0 + 5.0 ** 0.5) * k
    dirs = np.vstack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    XS = np.hstack([rr * dirs for rr in rs])
    WS = np.concatenate([np.full(m_ang, w * rr ** 2 * 4.0 * np.pi / m_ang)
                         for rr, w in zip(rs, ws)])

    rows = []
    for tau in taus:
        om_j = 1.0 / np.sqrt(tau * LAM0)
        Gam = MJ ** 2 / (8.0 * np.pi * LAM0 ** 2 * tau)
        om = om_j + xi_fix * Gam
        Kc, Mc = KM(om, tau)
        fc = np.sum(W * PHI0 * np.exp(1j * om * X0[2]))
        # a_t fixed to be 0.01 to reproduce experiments, but chooes: a_t = tau ** -0.25 in case of very very large contrasts
        alpha = a_t * abs(Kc - ETA * Mc * a_t ** 2) / abs(fc)
        u, _ = solve_forced(om, alpha, tau, np.zeros(2 * N0))
        if u is None:
            print(f"  [tau] {tau}  Newton failed at a_t={a_t:g} "
                  f"(eta|M|a_t^2/|K| = {ETA*abs(Mc)*a_t**2/abs(Kc):.2f}); skipped")
            continue
        a_h = np.sum(W * PHI0 * u)
        f = alpha * fc
        a_th = a_theory(Kc, Mc, f)
        usc = u - alpha * np.exp(1j * om * X0[2])
        nrm = np.sqrt(np.sum(W * np.abs(usc) ** 2))
        e0 = np.sqrt(np.sum(W * np.abs(u - a_th * PHI0) ** 2)) / nrm
        d = abs((Kc - ETA * Mc * abs(a_h) ** 2) * a_h - f) / abs(a_h)
        # exterior scattered field on the annulus
        us_ext = evaluate_volume_potential(complex(om), PRE,
                                           u + ETA * np.abs(u) ** 2 * u, XS)
        phi_ext = evaluate_volume_potential(complex(om), PRE,
                                            PHI0.astype(complex), XS)
        # proposition compares against a * Phi_j
        e_ext = float(np.sqrt(np.sum(WS * np.abs(us_ext - a_h * phi_ext) ** 2)/ np.sum(WS * np.abs(us_ext) ** 2)))
        rows.append((tau, e0, d, e_ext))
    if len(rows) < 2:
        print("  fig 3 skipped: fewer than two converged points")
        return
    r = np.array(rows)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.loglog(r[:, 0], r[:, 1], "o", ms=5,
              label=r"$\|u^{sc}_h-(a_{\rm th}\varphi_j-u^{\rm inc})\|_W\,/\,\|u^{sc}_h\|_W$")
    ax.loglog(r[:, 0], r[:, 3], "^", ms=5,
              label=r"$\|u^{sc}_h-a_h\Phi_j\|_{L^2(\Omega)}\,/\,\|u^{sc}_h\|_{L^2(\Omega)}$")
    ax.loglog(r[:, 0], r[0, 1] * (r[:, 0] / r[0, 0]) ** -0.5, ":", color="k",
              lw=1.2, label=r"$\propto\tau^{-1/2}$")
    ax.loglog(r[:, 0], r[:, 2], "s", ms=5,
              label=r"$|(K-\eta M|a_h|^2)a_h-f|/|a_h|$")
    ax.loglog(r[:, 0], r[0, 2] * (r[:, 0] / r[0, 0]) ** -1.0, "--", color="k",
              lw=1.0, label=r"$\propto\tau^{-1}$")
    ax.set(xlabel=r"$\tau$", ylabel="error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_scattered_comparison.png", dpi=150, bbox_inches="tight")
    print("wrote fig3_scattered_comparison.png")


def fold_check(tau=200.0, xi_demo=-4):
    """Multivalued response, plot at xi_demo = (om-om_j)/Gamma: PALC in the plane wave parameter alpha, traces the predicted S-shaped curve through both folds"""
    om_j = 1.0 / np.sqrt(tau * LAM0)
    Gam = MJ ** 2 / (8.0 * np.pi * LAM0 ** 2 * tau)
    # continuation in alpha from below the lower fold point to above the upper one
    om = om_j + xi_demo * Gam
    Kc, Mc = KM(om, tau)
    em = ETA * Mc
    g = (Kc * np.conj(em)).real
    disc = 4.0 * g ** 2 - 3.0 * abs(em) ** 2 * abs(Kc) ** 2
    Ip = (2.0 * g + np.sqrt(disc)) / (3.0 * abs(em) ** 2)
    Im_ = (2.0 * g - np.sqrt(disc)) / (3.0 * abs(em) ** 2)
    Plo, Phi_ = Ip * abs(Kc - em * Ip) ** 2, Im_ * abs(Kc - em * Im_) ** 2 # P(I+), P(I-)

    F, J, einc = forced_system(om, tau)
    fc = np.sum(W * PHI0 * einc)
    a_lo = 0.7 * np.sqrt(Plo) / abs(fc)
    a_hi = 1.25 * np.sqrt(Phi_) / abs(fc)
    Fp = np.concatenate([-einc.real, -einc.imag])
    _, x = solve_forced(om, a_lo, tau, np.zeros(2 * N0))
    br = palc_continue(F, J, x, a_lo, ds=0.03, ds_max=0.1, max_steps=400,
                       newton_tol=1e-10, F_p=lambda xx, pp: Fp, orient=+1.0,
                       p_min=0.5 * a_lo, p_max=a_hi, feasible=lambda x: True)
    pts = np.array([[pt.p * abs(fc),
                     abs(np.sum(W * PHI0 * (pt.x[:N0] + 1j * pt.x[N0:]))) ** 2]
                    for pt in br.sol])
    turn = np.where(np.diff(np.sign(np.diff(pts[:, 0]))) != 0)[0] + 1

    # actually show three distinct solutions at same abs(f), using Newton restarted at seeds: before <sqrt(P(I-)), between sqrt(P(I-)),sqrt(P(I+)), and after >sqrt(P(I+))
    f_mid = (Plo * Phi_) ** 0.25
    a_mid = f_mid / abs(fc)
    trip = []
    for seg in np.split(np.arange(len(pts)), turn)[:3]:
        k = seg[np.argmin(np.abs(pts[seg, 0] - f_mid))]
        u3, _ = solve_forced(om, a_mid, tau, br.sol[k].x.copy())
        if u3 is not None:
            trip.append(abs(np.sum(W * PHI0 * u3)) ** 2)
    r = np.roots([abs(em) ** 2, -2.0 * g, abs(Kc) ** 2, -f_mid ** 2])
    r = r.real[np.abs(r.imag) <= 1e-8 * (1.0 + np.abs(r.real))]

    # predicted theory curve
    Ith = np.linspace(1e-4, 1.1 * pts[:, 1].max(), 400)
    Pth = Ith * np.abs(Kc - em * Ith) ** 2

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot(Pth, Ith, "-", color="k", lw=1.2, label="amplitude equation")
    ax.plot(pts[:, 0] ** 2, pts[:, 1], "o", ms=3.5, mfc="none",
            label="numerical solution")
    for fv, lab in ((Plo, r"$|f|^2=P(I_\pm)$"), (Phi_, None)):
        ax.axvline(fv, ls=":", color="k", lw=1.0, label=lab)
    if trip:
        ax.plot([f_mid ** 2] * len(trip), trip, ".", ms=6, color="C3", zorder=5,
                label=rf"{len(trip)} solutions at $|f|^2={f_mid ** 2:.4f}$")
    ax.set(xlabel=r"$|f|^2$", ylabel=r"$|a_h|^2$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_fold_scurve.png", dpi=150, bbox_inches="tight")
    print("wrote fig4_fold_scurve.png")


def main():
    line_sweep()
    remainder_scaling()
    scattered_comparison()
    fold_check()


if __name__ == "__main__":
    main()
