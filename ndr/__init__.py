# ndr - nonlinear dielectric resonator package

from .quadrature import volume_quadrature_sphere, volume_quadrature_peanut, volume_quadrature_ellipsoid, precompute_kernel, KernelPrecomp
from .kernel import K_self, K_cross, dK_self, dK_cross, evaluate_volume_potential
from .state import pack_state, unpack_state, normalize_power, enforce_gauge, enforce_component_gauge
from .residuals import (
    residual_reduced, jacobian_reduced,
    residual_reduced_component_gauge, jacobian_reduced_component_gauge,
    residual_full, jacobian_full,
    residual_full_component_gauge, jacobian_full_component_gauge,
    residual_full_imperfect, jacobian_full_imperfect,
    residual_and_jacobian_full_imperfect,
    param_deriv_full_imperfect_N, param_deriv_full_imperfect_delta,
)
from .theory import theory_dimer, choose_beta_from_target, TheoryResult
from .newton import newton_solve, deflated_newton_solve, NewtonResult
from .continuation import (
    NaturalBranch, BranchPoint, natural_continue,
    continue_branch_reduced,
    continue_full_imperfect_in_N,
    continue_full_imperfect_in_delta,
    palc_full_imperfect_in_N,
    palc_full_imperfect_in_delta,
)
from .palc import PALCBranch, PALCPoint, palc_continue
from .branch_switching import (
    linear_scan_mode, linear_scan_mode_2d,
    odd_block_smin, seed_asymmetric_from_even, asymmetry_measure,
)
from .postprocess import (
    extract_reduced_series, extract_full_series, extract_full_series_in_N,
    plot_branch_omega, plot_bifurcation_diagram, plot_odd_smin, plot_imperfect_pitchfork,
)
