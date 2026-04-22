import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ndr import (
    volume_quadrature_sphere, precompute_kernel, theory_dimer,
    continue_branch_reduced, continue_full_imperfect_in_N, continue_full_imperfect_in_delta,
    palc_full_imperfect_in_N, palc_full_imperfect_in_delta,
    odd_block_smin, asymmetry_measure,
    newton_solve, deflated_newton_solve,
    residual_full_imperfect, jacobian_full_imperfect, residual_and_jacobian_full_imperfect,
    pack_state, unpack_state, normalize_power, enforce_component_gauge,
    plot_imperfect_pitchfork,
    NaturalBranch, BranchPoint,
)
from ndr.kernel import _HAS_NUMBA

# physical parameters
R = 0.22
TAU = 500.0
ETA = 1.0 + 0j
BETA = 1.0 + 0j

# quadrature resolution (n0 = N_RADIAL * N_MU * N_PHI per sphere)
N_RADIAL = 5
N_MU = 5
N_PHI = 8

# L values for each scan
LS_LINEAR = np.linspace(2.2, 6.0, 14)
LS_PITCH = np.linspace(2.2, 5.2, 7)
LS_IMPERFECT = [2.7, 4.7]

NMIN = 1e-5
NMAX = 5.0
SMIN_THRESH = 1e-2

# imperfect pitchfork settings
DELTA0 = 5e-3
NMAX_IMP = 0.1
DNMAX_IMP = 0.001
DDELTA_MAX = 5e-4

# continuation method: "natural" or "palc"
METHOD = "palc"
PALC_THETA = 1.0       # arclength scaling weight for the parameter

# newton/continuation tolerances
NTOL = 1e-11
NMAXIT_HARD = 200
DN_MAX = 0.01
DN_MIN = 1e-9

OUTDIR = os.path.join(ROOT, "output_ndr_dimer_report")


def mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


def local_mins(y):
    return [i for i in range(1, len(y)-1) if y[i] < y[i-1] and y[i] < y[i+1]]


def run_theory_scan(pre, outdir):
    mkdir(outdir)
    print("\ntheory calibration")

    rows = []
    for L in LS_LINEAR:
        th = theory_dimer(pre, L, tau=TAU, eta=ETA)
        rows.append((L, th.lambda_even, th.lambda_odd,
                     th.omega_even_star, th.omega_odd_star, th.Ncrit_beta1))
        print(f"  L={L:.2f}  lam_e={th.lambda_even:.5f}  lam_o={th.lambda_odd:.5f}"
              f"  Ncrit={th.Ncrit_beta1:.4g}")
    arr = np.array(rows)

    rows_p = [(L, theory_dimer(pre, L, tau=TAU, eta=ETA).Ncrit_beta1) for L in LS_PITCH]
    arr_p = np.array(rows_p)
    ok = np.isfinite(arr_p[:, 1]) & (arr_p[:, 1] > 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(arr[:, 0], arr[:, 1], lw=2, label="lam_even")
    axes[0].plot(arr[:, 0], arr[:, 2], lw=2, label="lam_odd")
    axes[0].set(xlabel="L", ylabel="lambda")
    axes[0].legend()

    axes[1].plot(arr[:, 0], arr[:, 3], lw=2, label="w*_even")
    axes[1].plot(arr[:, 0], arr[:, 4], lw=2, label="w*_odd")
    axes[1].set(xlabel="L", ylabel="omega*")
    axes[1].legend()

    axes[2].plot(arr_p[ok, 0], arr_p[ok, 1], lw=2, marker="o")
    axes[2].set(xlabel="L", ylabel="Ncrit (beta=1)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "theory_curves.png"), dpi=150)
    plt.close(fig)


def run_pitchfork_scan(pre, outdir):
    mkdir(outdir)
    print("\npitchfork indicator scan")
    n0 = len(pre.W)

    nrows = int(np.ceil(len(LS_PITCH) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3*nrows))
    axf = axes.flatten() if nrows > 1 else list(axes)

    for j, L in enumerate(LS_PITCH):
        print(f"  L={L:.2f}", end=" ", flush=True)
        br, _, _ = continue_branch_reduced(
            pre, sigma=+1, tau=TAU, eta=ETA, beta=BETA, L=L,
            N_min=NMIN, N_max=NMAX, dN_max=DN_MAX, dN_min=DN_MIN,
            gauge="component", newton_tol=1e-8, newton_max_iter=50)

        N_arr = np.array(br.param)
        smins = np.array([
            odd_block_smin(*unpack_state(pt.x, n0), L, pre, TAU, ETA, BETA, return_vec=False)[0]
            for pt in br.sol
        ])

        imin = int(np.argmin(smins))
        cands = [i for i in local_mins(smins.tolist()) if smins[i] < SMIN_THRESH]
        if cands:
            ii = cands[int(np.argmin(N_arr[cands]))]
            print(f"candidate smin={smins[ii]:.3e} at N={N_arr[ii]:.4f}")
        else:
            print(f"global min smin={smins[imin]:.3e} at N={N_arr[imin]:.4f}")

        axf[j].semilogy(N_arr, smins, lw=2)
        axf[j].axhline(SMIN_THRESH, ls="--", lw=1.5)
        axf[j].set(title=f"L={L:.1f}", xlabel="N", ylabel="smin(J_odd)", xlim=(0, NMAX))

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "odd_smin_grid.png"), dpi=150)
    plt.close(fig)


def branch_series(br, n):
    Nv, Av, wr, wi = [], [], [], []
    for pt in br.sol:
        u, w = unpack_state(pt.x, n)
        Nv.append(pt.p)
        Av.append(asymmetry_measure(u, pre_global))
        wr.append(w.real)
        wi.append(w.imag)
    return {"N": np.array(Nv), "A": np.array(Av),
            "omega_r": np.array(wr), "omega_i": np.array(wi)}


def state_at_N(br, Nt):
    pars = np.array(br.param)
    i = int(np.argmin(np.abs(pars - Nt)))
    return br.param[i], br.sol[i].x


def odd_biased_guess(x_small, eps, Npow, n0, Wf, gidx, direction=1.0):
    n = 2 * n0
    u, w = unpack_state(x_small, n)
    v = 0.5 * (u[:n0] + u[n0:])
    du = direction * np.concatenate([v, -v])
    ug = u + eps * du
    normalize_power(ug, Wf, Npow)
    enforce_component_gauge(ug, gidx)
    return pack_state(ug, w)


def find_large_root(sign_d, br_small, gi, n0, Wf, L, delta0, N0_eff,
                    eps_list=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)):
    Ns, xs = state_at_N(br_small, N0_eff)
    ps = np.array([Ns, L, sign_d * delta0])
    F = lambda x: residual_full_imperfect(x, ps, pre_global, TAU, ETA, BETA, gi)
    J = lambda x: jacobian_full_imperfect(x, ps, pre_global, TAU, ETA, BETA, gi)

    sol_sym = newton_solve(F, J, xs, tol=NTOL, max_iter=NMAXIT_HARD, linesearch=True)
    if not sol_sym.converged:
        print(f"    small root reconverge FAILED at N={Ns:.4g}")
        return None
    xsc = sol_sym.x

    for eps in eps_list:
        xg = odd_biased_guess(xsc, eps, Ns, n0, Wf, gi, direction=sign_d)
        sol = deflated_newton_solve(
            F, J, xg, [xsc], deflate_p=2, deflate_alpha=0.1,
            tol=NTOL, max_iter=NMAXIT_HARD, linesearch=True)
        if sol.converged:
            u_l, _ = unpack_state(sol.x, 2*n0)
            A = asymmetry_measure(u_l, pre_global)
            if abs(A) > 1e-6:
                print(f"    large root delta={sign_d*delta0:+.2e} N={Ns:.4g} A={A:.3e}")
                return (Ns, sol.x)

    warnings.warn(f"no large root at delta={sign_d*delta0:.2e} L={L:.3f}")
    return None


def march_branch(Ns, Ne, xs, sign_d, gi, dN0, dNmax, L, delta0, method="natural",
                 palc_theta=PALC_THETA):
    if method == "palc":
        orient = +1.0 if Ne > Ns else -1.0
        return palc_full_imperfect_in_N(
            pre_global, TAU, ETA, BETA, L, sign_d*delta0,
            Ns, Ne, xs, gi,
            ds=dN0, ds_max=dNmax, theta=palc_theta,
            newton_tol=NTOL, newton_max_iter=NMAXIT_HARD, verbosity=0)

    # natural continuation (original logic)
    d = 1.0 if Ne > Ns else -1.0
    dN = d * dN0
    br = NaturalBranch()
    br.sol.append(BranchPoint(x=xs.copy(), p=Ns, step=0))
    br.param.append(Ns)
    x, Nc, step = xs.copy(), Ns, 0
    x_prev, Nc_prev = None, None
    while (d > 0 and Nc < Ne - 1e-14) or (d < 0 and Nc > Ne + 1e-14):
        Nt = min(Nc+dN, Ne) if d > 0 else max(Nc+dN, Ne)
        # secant predictor
        if x_prev is not None and abs(Nc - Nc_prev) > 1e-14:
            x_guess = x + ((Nt - Nc) / (Nc - Nc_prev)) * (x - x_prev)
        else:
            x_guess = x.copy()
        pv = np.array([Nt, L, sign_d*delta0])
        F = lambda x: residual_full_imperfect(x, pv, pre_global, TAU, ETA, BETA, gi)
        J = lambda x: jacobian_full_imperfect(x, pv, pre_global, TAU, ETA, BETA, gi)
        FJ = lambda x: residual_and_jacobian_full_imperfect(x, pv, pre_global, TAU, ETA, BETA, gi)
        sol = newton_solve(F, J, x_guess, tol=NTOL, max_iter=NMAXIT_HARD, linesearch=True, FJ=FJ)
        if sol.converged:
            step += 1
            x_prev = x.copy()
            Nc_prev = Nc
            x = sol.x
            Nc = Nt
            br.sol.append(BranchPoint(x=x.copy(), p=Nc, step=step))
            br.param.append(Nc)
            dN = np.sign(dN) * min(abs(dN)*1.2, dNmax)
        else:
            dN *= 0.5
            if abs(dN) < 1e-8:
                break
    return br


def merge_branches(b1, b2):
    sols = b1.sol + b2.sol
    pars = b1.param + b2.param
    idx = np.argsort(pars)
    ps = [pars[i] for i in idx]
    ss = [sols[i] for i in idx]
    keep = [0] + [i for i in range(1, len(ps)) if abs(ps[i]-ps[i-1]) > 1e-12]
    br = NaturalBranch()
    br.sol = [ss[i] for i in keep]
    br.param = [ps[i] for i in keep]
    return br


def pullback_to_delta0(br_large, sign_d, gi, n0, L, delta0, d_delta_max, subsample=5,
                       method="natural", palc_theta=PALC_THETA):
    n = 2 * n0
    Npb, Apb, wrpb, wipb = [], [], [], []
    pts = br_large.sol[::subsample]
    for pt in pts:
        if method == "palc":
            br_d = palc_full_imperfect_in_delta(
                pre_global, TAU, ETA, BETA, pt.p, L,
                sign_d*delta0, 0.0, pt.x, gi,
                ds_max=d_delta_max, theta=palc_theta,
                newton_tol=NTOL, newton_max_iter=NMAXIT_HARD, verbosity=0)
        else:
            br_d = continue_full_imperfect_in_delta(
                pre_global, TAU, ETA, BETA, pt.p, L, sign_d*delta0, 0.0, pt.x, gi,
                d_delta_max=d_delta_max, newton_tol=NTOL, newton_max_iter=NMAXIT_HARD)
        if not br_d.sol:
            continue
        u, w = unpack_state(br_d.sol[-1].x, n)
        Npb.append(pt.p)
        Apb.append(asymmetry_measure(u, pre_global))
        wrpb.append(w.real)
        wipb.append(w.imag)
    return {"N": np.array(Npb), "A": np.array(Apb),
            "omega_r": np.array(wrpb), "omega_i": np.array(wipb)}


def run_imperfect_pitchfork(pre, L, outdir, delta0=DELTA0, N_min=1e-4, N_max=NMAX_IMP,
                            dN_max=DNMAX_IMP, d_delta_max=DDELTA_MAX,
                            method=METHOD, palc_theta=PALC_THETA):
    global pre_global
    pre_global = pre

    mkdir(outdir)
    n0 = len(pre.W)
    n = 2 * n0
    Wf = np.concatenate([pre.W, pre.W])

    Ncrit = theory_dimer(pre, L, tau=TAU, eta=ETA).Ncrit_beta1 / max(BETA.real, 1e-16)

    # seed from even branch at N_min
    br_seed, _, _ = continue_branch_reduced(
        pre, sigma=+1, tau=TAU, eta=ETA, beta=BETA, L=L,
        N_min=N_min, N_max=max(N_min*1.2, N_min+1e-4),
        dN_max=dN_max, dN_min=1e-8, gauge="component",
        newton_tol=1e-10, newton_max_iter=30)
    u0, w0 = unpack_state(br_seed.sol[0].x, n0)
    uf0 = np.concatenate([u0, u0]).astype(complex)
    normalize_power(uf0, Wf, N_min)
    gidx = int(np.argmax(np.abs(uf0)))
    gidx2 = gidx + n0
    enforce_component_gauge(uf0, gidx)
    x0 = pack_state(uf0, w0)

    # continue at fixed delta to get the biased small-A branches
    print(f"  small-A branches at delta=+-{delta0:.1e}  (method={method})")
    if method == "palc":
        br_p = palc_full_imperfect_in_N(
            pre, TAU, ETA, BETA, L, +delta0, N_min, N_max, x0, gidx,
            ds_max=dN_max, theta=palc_theta,
            newton_tol=NTOL, newton_max_iter=NMAXIT_HARD)
        br_m = palc_full_imperfect_in_N(
            pre, TAU, ETA, BETA, L, -delta0, N_min, N_max, x0, gidx,
            ds_max=dN_max, theta=palc_theta,
            newton_tol=NTOL, newton_max_iter=NMAXIT_HARD)
    else:
        br_p = continue_full_imperfect_in_N(
            pre, TAU, ETA, BETA, L, +delta0, N_min, N_max, x0, gidx,
            dN_max=dN_max, newton_tol=NTOL, newton_max_iter=NMAXIT_HARD)
        br_m = continue_full_imperfect_in_N(
            pre, TAU, ETA, BETA, L, -delta0, N_min, N_max, x0, gidx,
            dN_max=dN_max, newton_tol=NTOL, newton_max_iter=NMAXIT_HARD)

    Nmax_p = br_p.param[-1] if br_p.param else N_min
    Nmax_m = br_m.param[-1] if br_m.param else N_min
    print(f"    delta>0 reached N={Nmax_p:.4g},  delta<0 reached N={Nmax_m:.4g}  (Ncrit={Ncrit:.4g})")

    # search for the large-|A| roots at N0_eff
    N0_eff = np.clip(0.01, 1.5*N_min, 0.9*N_max)
    print(f"  searching large roots at N0_eff={N0_eff:.4g}")
    large_p = find_large_root(+1, br_p, gidx,  n0, Wf, L, delta0, N0_eff)
    large_m = find_large_root(-1, br_m, gidx2, n0, Wf, L, delta0, N0_eff)

    if large_p is None or large_m is None:
        warnings.warn(f"large roots not found at L={L:.3f}, saving skeleton")
        plot_skeleton(br_p, br_m, L, delta0, outdir, n)
        return

    # continue the large branches up and down in N
    print(f"  continuing large branches  (method={method})")
    def cont_large(sign_d, info, gi):
        Ns, xs = info
        dN0 = max(0.1*Ns, 5e-4)
        bd = march_branch(Ns, N_min, xs, sign_d, gi, dN0, dN_max, L, delta0,
                          method=method, palc_theta=palc_theta)
        bu = march_branch(Ns, N_max, xs, sign_d, gi, dN0, dN_max, L, delta0,
                          method=method, palc_theta=palc_theta)
        return merge_branches(bd, bu)

    br_lp = cont_large(+1, large_p, gidx)
    br_lm = cont_large(-1, large_m, gidx2)

    # pull back delta -> 0 to get exact asymmetric solutions
    print(f"  pulling back delta -> 0  (method={method})")
    pb_p = pullback_to_delta0(br_lp, +1, gidx,  n0, L, delta0, d_delta_max,
                              method=method, palc_theta=palc_theta)
    pb_m = pullback_to_delta0(br_lm, -1, gidx2, n0, L, delta0, d_delta_max,
                              method=method, palc_theta=palc_theta)

    plot_imperfect_pitchfork(
        pb_p, pb_m, L, delta0,
        outpath_A=os.path.join(outdir, f"A_vs_N_L{L:.3f}.png"),
        outpath_omega=os.path.join(outdir, f"omega_vs_N_L{L:.3f}.png"))
    print(f"  done")


def plot_skeleton(br_p, br_m, L, delta0, outdir, n):
    # fallback plot when large roots are not found
    fig, (ax_a, ax_w) = plt.subplots(1, 2, figsize=(12, 4))

    def add_series(br, ls, tag):
        Nv, Av, wr = [], [], []
        for pt in br.sol:
            u, w = unpack_state(pt.x, n)
            Nv.append(pt.p)
            Av.append(asymmetry_measure(u, pre_global))
            wr.append(w.real)
        if Nv:
            ax_a.plot(Nv, Av, lw=2, ls=ls, label=tag)
            ax_w.plot(Nv, wr, lw=2, ls=ls, label=tag)

    add_series(br_p, "-",  f"delta=+{delta0:.1e}")
    add_series(br_m, "--", f"delta=-{delta0:.1e}")
    ax_a.set(xlabel="N", ylabel="A", ylim=(-1.05, 1.05), title=f"L={L:.3f} (skeleton)")
    ax_a.legend(fontsize=8)
    ax_w.set(xlabel="N", ylabel="Re(omega)")
    ax_w.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"A_vs_N_L{L:.3f}.png"), dpi=150)
    plt.close(fig)


def main():
    mkdir(OUTDIR)
    print(f"output dir: {OUTDIR}")

    X0, W0, a0 = volume_quadrature_sphere(R, n_radial=N_RADIAL, n_mu=N_MU, n_phi=N_PHI)
    pre = precompute_kernel(X0, W0, a0)
    print(f"n0={len(W0)} per sphere,  numba={'yes' if _HAS_NUMBA else 'no'}")

    run_theory_scan(pre, OUTDIR)

    print("\nimperfect pitchfork diagrams")
    for L in LS_IMPERFECT:
        print(f"\nL={L:.3f}")
        run_imperfect_pitchfork(pre, L, OUTDIR)

    print(f"\ndone")


if __name__ == "__main__":
    main()
