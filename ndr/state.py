# State vector packing/unpacking, power normalisation, phase gauges
#
# The complex field u and frequency omega are stored as a single real vector:
#   x = [Re(u), Im(u), Re(omega), Im(omega)]

import numpy as np


def pack_state(u, omega):
    """Pack complex (u, omega) into real vector x"""
    n = len(u)
    x = np.empty(2 * n + 2)
    x[:n] = u.real
    x[n:2*n] = u.imag
    x[2*n] = omega.real
    x[2*n+1] = omega.imag
    return x


def unpack_state(x, n):
    """Unpack real vector x into complex (u, omega)"""
    u = x[:n] + 1j * x[n:2*n]
    omega = complex(x[2*n], x[2*n+1])
    return u, omega


def inner_W(u, v, W):
    """Weighted inner product <u, v>_W = sum(conj(v) * u * W)"""
    return np.dot(W * u, v.conj())


def normalize_power(u, W, target):
    """Scale u in-place so that ||u||^2_W = target"""
    p = np.sum(np.abs(u)**2 * W)
    u *= np.sqrt(target / p)
    return u


def enforce_gauge(u, uref, W):
    """Innerprod gauge: rotate u so Im(<u, uref>_W) = 0"""
    s = inner_W(u, uref, W)
    u *= np.exp(-1j * np.angle(s))
    return u


def enforce_component_gauge(u, idx):
    """Component gauge: rotate u so Im(u[idx]) = 0 and Re(u[idx]) > 0"""
    u *= np.exp(-1j * np.angle(u[idx]))
    if u[idx].real < 0:
        u *= -1
    return u
