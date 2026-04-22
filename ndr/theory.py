# Ammari-Li: Ncrit = (lam_e - lam_o) / (3 lam_o Apm - lam_e App)
# At omega=0 the kernel is real symmetric its eigenpairs give
# the linear resonance frequencies and the Ammari-Li Ncrit formula.

import numpy as np
from dataclasses import dataclass
from scipy.linalg import eigh
from .quadrature import KernelPrecomp
from .kernel import K_self, K_cross


@dataclass
class TheoryResult:
    lambda_even: float
    lambda_odd: float
    phi_even: np.ndarray
    phi_odd: np.ndarray
    App: float # overlap int phi_e^4
    Aplusminus: float # mixed overlap int phi_e^2 phi_o^2
    omega_even_star: float
    omega_odd_star: float
    Ncrit_beta1: float # pitchfork threshold at beta=1
    denom: float


def theory_dimer(pre, L, tau, eta=1.0+0j):
    """Compute terms at separation L"""
    W = pre.W
    sqW = np.sqrt(W)
    isqW = 1.0 / sqW

    Ks = K_self(0j, pre).real
    Kc = K_cross(0j, L, pre).real
    Keven = Ks + Kc
    Kodd = Ks - Kc

    def leading_eig(Keff):
        # symmetrise W^{1/2} K W^{-1/2} and take largest eigenvalue
        A = (sqW[:, None] * Keff) * isqW[None, :]
        A = 0.5 * (A + A.T)
        vals, vecs = eigh(A)
        idx = np.argmax(vals)
        phi = isqW * vecs[:, idx]
        return vals[idx], phi

    lam_e, phi_e = leading_eig(Keven)
    lam_o, phi_o = leading_eig(Kodd)

    # normalise: full dimer even mode is [phi_e; phi_e], so 2 int phi^2 W = 1
    phi_e /= np.sqrt(2) * np.sqrt(np.sum(W * phi_e**2))
    phi_o /= np.sqrt(2) * np.sqrt(np.sum(W * phi_o**2))

    # quartic overlaps (factor 2 from two spheres)
    App = 2 * np.sum(W * phi_e**4)
    Apm = 2 * np.sum(W * phi_e**2 * phi_o**2)

    eta_r = eta.real
    w_e = 1.0 / np.sqrt(tau * eta_r * lam_e)
    w_o = 1.0 / np.sqrt(tau * eta_r * lam_o)

    # Ammari-Li: Ncrit = (lam_e - lam_o) / (3 lam_o Apm - lam_e App)
    den = 3 * lam_o * Apm - lam_e * App
    Ncrit = (lam_e - lam_o) / den if den != 0 else np.inf

    return TheoryResult(
        lambda_even=lam_e, lambda_odd=lam_o,
        phi_even=phi_e, phi_odd=phi_o,
        App=App, Aplusminus=Apm,
        omega_even_star=w_e, omega_odd_star=w_o,
        Ncrit_beta1=Ncrit, denom=den,
    )


def choose_beta_from_target(pre, Lstar, tau, eta, Ncrit_target):
    """Pick beta so that Ncrit(Lstar) = Ncrit_target
    Heuristic: Ncrit ~ Ncrit(beta=1)/beta"""
    th = theory_dimer(pre, Lstar, tau=tau, eta=eta)
    N1 = th.Ncrit_beta1
    if not np.isfinite(N1) or N1 <= 0:
        import warnings
        warnings.warn(f"Ncrit_beta1 at L={Lstar} is {N1}, returning beta=1")
        return 1.0, th
    return N1 / Ncrit_target, th
