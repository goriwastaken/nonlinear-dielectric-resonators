#extract branch data and plot
import matplotlib.pyplot as plt
import numpy as np

from .state import unpack_state
from .branch_switching import asymmetry_measure
from .continuation import NaturalBranch


def extract_reduced_series(branch, pre):
    n0 = len(pre.W)
    out = {"N": [], "omega_r": [], "omega_i": [], "P_sphere": []}
    for pt in branch.sol:
        u, w = unpack_state(pt.x, n0)
        out["N"].append(pt.p)
        out["omega_r"].append(w.real)
        out["omega_i"].append(w.imag)
        out["P_sphere"].append(float(np.sum(np.abs(u)**2 * pre.W)))
    return {k: np.array(v) for k, v in out.items()}


def extract_full_series(branch, pre, N_fixed):
    n = 2 * len(pre.W)
    out = {"L": [], "omega_r": [], "omega_i": [], "A": []}
    for pt in branch.sol:
        u, w = unpack_state(pt.x, n)
        out["L"].append(pt.p)
        out["omega_r"].append(w.real)
        out["omega_i"].append(w.imag)
        out["A"].append(asymmetry_measure(u, pre))
    d = {k: np.array(v) for k, v in out.items()}
    d["N"] = N_fixed
    return d


def extract_full_series_in_N(branch, pre, L_fixed):
    n = 2 * len(pre.W)
    out = {"N": [], "omega_r": [], "omega_i": [], "A": []}
    for pt in branch.sol:
        u, w = unpack_state(pt.x, n)
        out["N"].append(pt.p)
        out["omega_r"].append(w.real)
        out["omega_i"].append(w.imag)
        out["A"].append(asymmetry_measure(u, pre))
    d = {k: np.array(v) for k, v in out.items()}
    d["L"] = L_fixed
    return d


def plot_branch_omega(series, label="", outpath=None, show=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(series["N"], series["omega_r"], lw=2, label=label)
    ax1.set(xlabel="N", ylabel="Re(omega)")
    ax1.legend()
    ax2.plot(series["N"], series["omega_i"], lw=2, label=label)
    ax2.set(xlabel="N", ylabel="Im(omega)")
    ax2.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_bifurcation_diagram(ser_even, ser_odd=None, ser_asym_plus=None,
                             ser_asym_minus=None, Ncrit_th=None,
                             outpath=None, show=False):
    fig, (ax_a, ax_wr, ax_wi) = plt.subplots(1, 3, figsize=(15, 4))

    ax_a.plot(ser_even["N"], np.zeros(len(ser_even["N"])), lw=2.5, label="symmetric")
    if ser_odd is not None:
        ax_a.plot(ser_odd["N"], np.zeros(len(ser_odd["N"])), lw=2, ls="--", label="antisymmetric")
    if ser_asym_plus is not None:
        ax_a.plot(ser_asym_plus["N"], ser_asym_plus["A"], lw=2, label="asym +")
    if ser_asym_minus is not None:
        ax_a.plot(ser_asym_minus["N"], ser_asym_minus["A"], lw=2, label="asym -")
    if Ncrit_th is not None:
        ax_a.axvline(Ncrit_th, ls=":", lw=1.5, label=f"Ncrit={Ncrit_th:.3g}")
    ax_a.set(xlabel="N", ylabel="A")
    ax_a.legend(fontsize=8)

    for ax, key, lab in [(ax_wr, "omega_r", "Re(omega)"), (ax_wi, "omega_i", "Im(omega)")]:
        ax.plot(ser_even["N"], ser_even[key], lw=2, label="even")
        if ser_odd is not None:
            ax.plot(ser_odd["N"], ser_odd[key], lw=2, ls="--", label="odd")
        if ser_asym_plus is not None:
            ax.plot(ser_asym_plus["N"], ser_asym_plus[key], lw=2, label="asym +")
        if Ncrit_th is not None:
            ax.axvline(Ncrit_th, ls=":", lw=1.5)
        ax.set(xlabel="N", ylabel=lab)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_odd_smin(N_vals, smins, smin_thresh, Ncrit_th=None, outpath=None, show=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(N_vals, smins, lw=2)
    ax.axhline(smin_thresh, ls="--", lw=1.5, label=f"thresh={smin_thresh:.1e}")
    if Ncrit_th is not None:
        ax.axvline(Ncrit_th, ls=":", lw=1.5, label=f"Ncrit={Ncrit_th:.3g}")
    ax.set(xlabel="N", ylabel="smin(J_odd)")
    ax.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_imperfect_pitchfork(pb_plus, pb_minus, L, delta0,
                             outpath_A=None, outpath_omega=None, show=False):
    fig_a, ax_a = plt.subplots(figsize=(6, 4))
    if len(pb_plus["N"]) > 0:
        ax_a.plot(pb_plus["N"], np.zeros(len(pb_plus["N"])), lw=3, label="symmetric")
        ax_a.plot(pb_plus["N"], pb_plus["A"], lw=2, marker="o", ms=3, label="asym +")
    if len(pb_minus["N"]) > 0:
        ax_a.plot(pb_minus["N"], pb_minus["A"], lw=2, marker="o", ms=3, label="asym -")
    ax_a.set(xlabel="N", ylabel="A", ylim=(-1.05, 1.05),
             title=f"L={L:.3f}  delta0={delta0:.1e}")
    ax_a.legend(fontsize=8)
    plt.tight_layout()
    if outpath_A:
        plt.savefig(outpath_A, dpi=150)
    if show:
        plt.show()
    plt.close(fig_a)

    fig_w, ax_w = plt.subplots(figsize=(6, 4))
    if len(pb_plus["N"]) > 0:
        ax_w.plot(pb_plus["N"], pb_plus["omega_r"], lw=2, label="Re(w) +")
        ax_w.plot(pb_plus["N"], pb_plus["omega_i"], lw=2, label="Im(w) +")
    if len(pb_minus["N"]) > 0:
        ax_w.plot(pb_minus["N"], pb_minus["omega_r"], lw=2, ls="--", label="Re(w) -")
        ax_w.plot(pb_minus["N"], pb_minus["omega_i"], lw=2, ls="--", label="Im(w) -")
    ax_w.set(xlabel="N", ylabel="omega", title=f"omega(N)  L={L:.3f}")
    ax_w.legend(fontsize=8)
    plt.tight_layout()
    if outpath_omega:
        plt.savefig(outpath_omega, dpi=150)
    if show:
        plt.show()
    plt.close(fig_w)
