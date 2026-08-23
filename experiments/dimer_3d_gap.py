"""Branch tracking for 3D dimer: symmetric, antisymmetric and the
two asymmetric branches above N_crit with bifurcation diagrams."""
import os
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import bisect

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ndr import (
    volume_quadrature_sphere, precompute_kernel, theory_dimer,
    continue_branch_reduced,
    residual_full_imperfect, jacobian_full_imperfect,
    palc_full_imperfect_in_N, continue_full_imperfect_in_delta,
    newton_solve, asymmetry_measure,
    pack_state, unpack_state, normalize_power, enforce_component_gauge,
)
from ndr.kernel import build_full_K

R = 1.0
TAU = 500.0
ETA = 1.0 + 0j
BETA = 1.0 + 0j
RADIAL_BETA = 2.0

TRACK_QUAD = (8, 8, 12)
N_MIN_TRACK = 1e-4
PALC_THETA = 1.0
NTOL_TRACK = 1e-11
NMAXIT_TRACK = 60
IMPERF_DELTA0 = 0.05
MIRROR_D2 = True# reflect D_2 through the mid-plane

OUTDIR = os.path.join(ROOT, "output_close_dimer")


def trace_asymmetric_branch(pre, L, N_lo, N_crit, N_max, delta0):
    """One asymmetric branch, sign(A) = sign(delta0), by Keller's Method V
    (branch switching through the (1+-delta) imperfection)
    Returns dict(N, A, wr, wi) or None if anything fails."""
    n0 = len(pre.W); n = 2 * n0
    sgn = np.sign(delta0)

    # seed: principal mode of the perturbed linear dimer
    Wf = np.concatenate([pre.W, pre.W])
    eta_v = np.concatenate([np.full(n0, ETA.real * (1.0 + delta0)),
                            np.full(n0, ETA.real * (1.0 - delta0))])
    G = build_full_K(0j, L, pre).real / Wf[None, :]
    d = np.sqrt(Wf * eta_v)
    Ksym = (d[:, None] * G) * d[None, :]
    Ksym = 0.5 * (Ksym + Ksym.T)
    vals, vecs = eigh(Ksym)
    lam = float(vals[-1])
    u0 = (vecs[:, -1] / d).astype(complex)
    if u0[int(np.argmax(np.abs(u0)))].real < 0:
        u0 = -u0
    normalize_power(u0, Wf, N_lo)
    om0 = 1.0 / np.sqrt(TAU * lam)# eta already partof lam
    om0 = complex(om0, -0.02 * om0)
    gidx = int(np.argmax(np.abs(u0)))
    enforce_component_gauge(u0, gidx)

    p0 = np.array([N_lo, L, delta0])
    r0 = newton_solve(
        lambda x: residual_full_imperfect(x, p0, pre, TAU, ETA, BETA, gidx),
        lambda x: jacobian_full_imperfect(x, p0, pre, TAU, ETA, BETA, gidx),
        pack_state(u0, om0), tol=NTOL_TRACK, max_iter=NMAXIT_TRACK)
    if not r0.converged or asymmetry_measure(unpack_state(r0.x, n)[0], pre) * sgn <= 0:
        return None

    # PALC with p=N up to N_max
    br_up = palc_full_imperfect_in_N(
        pre, TAU, ETA, BETA, L, delta0, N_lo, N_max, r0.x, gidx,
        ds=0.03, ds_max=0.15, ds_min=1e-7, max_steps=2000, theta=PALC_THETA,
        newton_tol=NTOL_TRACK, newton_max_iter=NMAXIT_TRACK, verbosity=0)
    if not br_up.sol:
        return None
    top = min(br_up.sol, key=lambda pt: abs(pt.p - N_max))
    N_s = float(top.p)
    if asymmetry_measure(unpack_state(top.x, n)[0], pre) * sgn <= 0:
        return None

    # delta ->0 at fixed N_s lands on the exact asymmetric branch
    br_d = continue_full_imperfect_in_delta(
        pre, TAU, ETA, BETA, N_s, L, delta0, 0.0, top.x, gidx,
        d_delta_max=4e-3, newton_tol=NTOL_TRACK, newton_max_iter=NMAXIT_TRACK,
        verbosity=0)
    if abs(br_d.param[-1]) > 1e-5:
        return None
    x0 = br_d.sol[-1].x
    A0 = asymmetry_measure(unpack_state(x0, n)[0], pre)
    if abs(A0) < 1e-2 or np.sign(A0) != sgn:
        return None
    sign = int(sgn)

    # delta=0: up to N_max, then down to N_crit, one PALC step moves N by <= ds_max
    pts = {}
    ms_dn = max(150, int(3.0 * (N_s - N_crit) / 0.15))
    for n_to, ms in ((N_max, 1500), (N_crit, ms_dn)):
        br = palc_full_imperfect_in_N(
            pre, TAU, ETA, BETA, L, 0.0, N_s, n_to, x0, gidx,
            ds=0.02, ds_max=0.15, ds_min=1e-6, max_steps=ms, theta=PALC_THETA,
            newton_tol=NTOL_TRACK, newton_max_iter=NMAXIT_TRACK, verbosity=0)
        for pt in br.sol:
            if not (N_crit - 1e-9 <= pt.p <= N_max + 1e-9):
                continue
            u, w = unpack_state(pt.x, n)
            A = asymmetry_measure(u, pre)
            if abs(A) < 1e-3 or np.sign(A) != sign:
                continue
            pts[round(pt.p, 10)] = (pt.p, A, w.real, w.imag)


    rows = []
    for r in sorted(pts.values(), key=lambda r: r[0]):#  keep the longest non-decreasing-|A| run
        if not rows or abs(r[0] - rows[-1][0]) > 1e-4 or abs(r[1] - rows[-1][1]) > 1e-4:
            rows.append(r)
    if len(rows) >= 3:
        a = [abs(r[1]) for r in rows]
        tails, ends, parent = [], [], [-1] * len(rows)
        for i, ai in enumerate(a):
            j = bisect.bisect_right(ends, ai)
            parent[i] = tails[j - 1] if j > 0 else -1
            if j == len(ends):
                ends.append(ai); tails.append(i)
            else:
                ends[j] = ai; tails[j] = i
        seq, k = [], tails[-1]
        while k != -1:
            seq.append(rows[k]); k = parent[k]
        rows = seq[::-1]
    if not rows:
        return None
    return dict(N=np.array([r[0] for r in rows]), A=np.array([r[1] for r in rows]),
                wr=np.array([r[2] for r in rows]), wi=np.array([r[3] for r in rows]))


def run_branch_tracking(g, n_max=None, quad=None):
    os.makedirs(OUTDIR, exist_ok=True)
    L = 2.0 * R + g
    quad = tuple(quad) if quad else TRACK_QUAD
    X0, W0, a0 = volume_quadrature_sphere(
        R, n_radial=quad[0], n_mu=quad[1], n_phi=quad[2],
        polar_axis="x", radial_map="tanh", radial_beta=RADIAL_BETA)
    pre = precompute_kernel(X0, W0, a0, mirror_d2=MIRROR_D2)
    n0 = len(pre.W)

    # clip window for the tracking range
    th = theory_dimer(pre, L, tau=TAU, eta=ETA)
    n_pred = th.Ncrit_beta1 / max(BETA.real, 1e-16)
    if n_max is None:
        n_kerr = 1.0 / th.App # (App = 2 int phi_e^4 which is approx R^-3)
        n_max = float(np.clip(8.0 * n_pred, 0.7 * n_kerr, 6.5 * n_kerr))
    dN = max((n_max - N_MIN_TRACK) / 150.0, 1e-4)
    print(f"\n[track] g/R={g/R:.3f}  L={L:.4f}  quad={quad}  n0={n0}  "
          f"N_crit~{n_pred:.3g}  N<={n_max:.3g}")

    # symm/antisymm branches
    reduced = {}
    for sigma in (+1, -1):
        br, _, _ = continue_branch_reduced(
            pre, sigma=sigma, tau=TAU, eta=ETA, beta=BETA, L=L,
            N_min=N_MIN_TRACK, N_max=n_max, dN_max=dN, gauge="component",
            newton_tol=NTOL_TRACK, newton_max_iter=NMAXIT_TRACK, verbosity=0)
        om = np.array([unpack_state(pt.x, n0)[1] for pt in br.sol])
        reduced[sigma] = dict(N=np.array([pt.p for pt in br.sol]), wr=om.real, wi=om.imag)
    even, odd = reduced[+1], reduced[-1]

    empty = dict(N=np.array([]), A=np.array([]), wr=np.array([]), wi=np.array([]))
    N_lo = max(0.4 * n_pred, float(even["N"][0]))
    asym_p = trace_asymmetric_branch(pre, L, N_lo, n_pred, n_max, +IMPERF_DELTA0) or empty
    asym_m = trace_asymmetric_branch(pre, L, N_lo, n_pred, n_max, -IMPERF_DELTA0) or empty

    #onset of symmbreaking: fit A^2 = k(N -N_c), None if unusable(then leading-order N_crit is the better estimate)
    n_onset = None
    N = np.concatenate([asym_p["N"], asym_m["N"]])
    A = np.abs(np.concatenate([asym_p["A"], asym_m["A"]]))
    m = A <= 0.35
    if m.sum() >= 4:
        k, b0 = np.polyfit(N[m], A[m] ** 2, 1)
        Nc = -b0 / k
        if k > 0 and 0.7 * n_pred <= Nc <= N.min() + 1e-9:
            n_onset = float(Nc)

    dev = None
    if len(asym_p["N"]) and len(asym_m["N"]):
        lo = max(asym_p["N"].min(), asym_m["N"].min())
        hi = min(asym_p["N"].max(), asym_m["N"].max())
        if hi > lo:
            Ng = np.linspace(lo, hi, 50)
            dev = float(np.max(np.abs(np.interp(Ng, asym_p["N"], asym_p["A"]) +
                                      np.interp(Ng, asym_m["N"], asym_m["A"]))))
    for ortho in ("omega", "A"):
        plot_bifurcation(g, even, odd, asym_p, asym_m, n_onset, n_pred, ortho)


def plot_bifurcation(g, even, odd, asym_p, asym_m, n_onset, n_pred, ortho):
    """Bifurcation diagram, ortho="A" (asymmetry fct vs N) or "omega" (Re(omega) vs N)"""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    n_c = n_onset if n_onset is not None else n_pred

    if ortho == "A":
        ax.plot(even["N"], np.zeros_like(even["N"]), "-", color="C0", lw=2.0,
                label="symmetric")
        ax.plot(odd["N"], np.zeros_like(odd["N"]), "--", color="C1", lw=1.5,
                label="antisymmetric")
        for br, lab, c in [(asym_p, "asymmetric +", "C2"),
                           (asym_m, "asymmetric -", "C3")]:
            if not len(br["N"]):
                continue
            N = np.concatenate([[n_c], br["N"]])
            A = np.concatenate([[0.0], br["A"]])
            ax.plot(N, A, "-", color=c, lw=1.8, label=lab)
        amax = max((np.abs(br["A"]).max() for br in (asym_p, asym_m)
                    if len(br["N"])), default=0.3)
        lim = min(1.05, 1.2 * amax)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(r"asymmetry $A=(P_1-P_2)/(P_1+P_2)$")
    else:
        wr_all = list(even["wr"]) + list(odd["wr"])
        ax.plot(even["N"], even["wr"], "-", color="C0", lw=2.0, label="symmetric")
        ax.plot(odd["N"], odd["wr"], "-", color="C1", lw=1.5, label="antisymmetric")
        w0 = float(np.interp(n_c, even["N"], even["wr"]))
        labeled = False
        for br in (asym_p, asym_m):
            if not len(br["N"]):
                continue
            N = np.concatenate([[n_c], br["N"]])
            wr = np.concatenate([[w0], br["wr"]])
            ax.plot(N, wr, "-", color="C2", lw=1.8,
                    label=None if labeled else "asymmetric")
            labeled = True
            wr_all += list(br["wr"])
        lo, hi = min(wr_all), max(wr_all); pad = 0.06 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(r"$\mathrm{Re}\,\omega$")

    ax.axvline(n_pred, ls=":", color="0.35", lw=1.0,
               label=rf"$N_{{\rm crit}}^{{\rm theory}}={n_pred:.3g}$ (leading order)")
    ax.axvline(n_c, ls=":", color="red", lw=1.5, alpha=0.7,
               label=rf"$N_{{\rm crit}}={n_c:.3g}$ (symmetry breaking)")
    ax.set_xlabel(r"normalization $N$")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, f"bifdiag_g{g/R:.3f}_{ortho}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"        wrote {path}")


def plot_splitting_and_ncrit(gap_list, quad=None):
    """Plot lambda_+ - lambda_- and leadingorder N_crit vs gap"""
    os.makedirs(OUTDIR, exist_ok=True)
    quad = tuple(quad) if quad else TRACK_QUAD
    X0, W0, a0 = volume_quadrature_sphere(
        R, n_radial=quad[0], n_mu=quad[1], n_phi=quad[2],
        polar_axis="x", radial_map="tanh", radial_beta=RADIAL_BETA)
    pre = precompute_kernel(X0, W0, a0, mirror_d2=MIRROR_D2)
    gR = np.array(sorted(gap_list), dtype=float)
    split, ncrit = [], []
    for x in gR:
        th = theory_dimer(pre, 2.0 * R + x * R, tau=TAU, eta=ETA)
        split.append(float((th.lambda_even - th.lambda_odd).real))
        ncrit.append(float(th.Ncrit_beta1))

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax[0].loglog(gR, split, "o-", color="C0")
    ax[0].set(xlabel=r"gap $g/R$", ylabel=r"$\lambda_+ - \lambda_-$")
    ax[1].loglog(gR, ncrit, "s-", color="C3")
    ax[1].set(xlabel=r"gap $g/R$", ylabel=r"$N_{\rm crit}$ (leading order)")
    for a in ax:
        a.invert_xaxis()
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(OUTDIR, "splitting_and_ncrit_vs_gap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    GAP_OVER_R_LIST = [0.05, 0.1, 1.0, 10.0, 19.0]
    g_list_test = [1/30, 1/20, 1/12, 1/8, 1/5, 1/3, 1/2,
                   1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    # GAP_OVER_R_LIST = [0.1,1.0]
    NMAX = None
    QUAD = None
    plot_splitting_and_ncrit(g_list_test, quad=QUAD)
    for gR in GAP_OVER_R_LIST:
        run_branch_tracking(gR * R, n_max=NMAX, quad=QUAD)


if __name__ == "__main__":
    main()
