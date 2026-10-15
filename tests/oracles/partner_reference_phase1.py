import math
import numpy as np


def lr_max(sy, su):
    return float((sy + su) / (2.0 * sy))


def option1_curve(lr, sy, su, E, discontinuous=False, luders_strain=None):
    if lr < 0.0:
        raise ValueError('Lr must be nonnegative')
    cutoff = lr_max(sy, su)
    if lr >= cutoff:
        return 0.0
    N = 0.3 * (1.0 - sy / su)
    if discontinuous:
        if lr < 1.0:
            return float((1.0 + lr * lr / 2.0) ** -0.5)
        if luders_strain is None:
            luders_strain = 0.0375 * (1.0 - 0.001 * (sy / 1.0e6))
        lam = 1.0 + E * luders_strain / sy
        f1 = (lam + 1.0 / (2.0 * lam)) ** -0.5
    else:
        mu = min(0.001 * E / sy, 0.6)
        if lr <= 1.0:
            return float((1.0 + lr * lr / 2.0) ** -0.5 * (0.3 + 0.7 * math.exp(-mu * lr ** 6)))
        f1 = (1.5 ** -0.5) * (0.3 + 0.7 * math.exp(-mu))
    if lr == 1.0:
        return float(f1)
    return float(f1 * lr ** ((N - 1.0) / (2.0 * N)))


def rho(chi, lr):
    if chi < 0.0 or lr < 0.0:
        raise ValueError('chi and Lr must be nonnegative')
    if chi < 5.2:
        r1 = 0.1 * chi ** 0.714 - 0.007 * chi ** 2 + 0.00003 * chi ** 5
    elif chi > 5.2:
        r1 = 0.25
    else:
        raise ValueError('The supplied rho1 equations omit chi == 5.2')
    if lr <= 0.8:
        return float(r1)
    if lr < 1.05:
        return float(4.0 * r1 * (1.05 - lr))
    return 0.0


def v_factor(chi, lr):
    if chi < 0.0 or lr < 0.0:
        raise ValueError('chi and Lr must be nonnegative')
    if lr < 1.05:
        return float(min(1.0 + 0.2 * lr + 0.02 * chi * (1.0 + 2.0 * lr), 3.1 - 2.0 * lr))
    if lr > 1.05:
        return 1.0
    raise ValueError('The supplied V equations omit Lr == 1.05')


def kj_secondary(ks, a, sy, beta):
    a_eff = a + (ks / sy) ** 2 / (2.0 * math.pi * beta)
    return float(math.sqrt(a_eff / a) * ks)


def kr(kp, ks, lr, kmat, a, sy, method='rho', beta=1.0):
    if method not in ('rho', 'V'):
        raise ValueError('method must be rho or V')
    if ks <= 0.0:
        return float(kp / kmat)
    if kp == 0.0:
        return float(kj_secondary(ks, a, sy, beta) / kmat)
    chi = ks * lr / kp
    if method == 'rho':
        return float((kp + ks) / kmat + rho(chi, lr))
    return float((kp + v_factor(chi, lr) * ks) / kmat)


def surface_mm_mb(a, c, B, theta):
    r = a / c
    x = a / B
    if not 0.0 < r <= 2.0:
        raise ValueError('Required aspect-ratio range is 0 < a/(2c) <= 1')
    st = math.sin(theta)
    ct = math.cos(theta)
    if r <= 1.0:
        M1 = 1.13 - 0.09 * r
        M2 = 0.89 / (0.2 + r) - 0.54
        M3 = 0.5 - 1.0 / (0.65 + r ** 2) + 14.0 * (1.0 - r) ** 24
        g = 1.0 + (0.1 + 0.35 * x ** 2) * (1.0 - st) ** 2
        ft = (r ** 2 * ct ** 2 + st ** 2) ** 0.25
        Phi = math.sqrt(1.0 + 1.464 * r ** 1.65)
        q = 0.2 + r + 0.6 * x
        H1 = 1.0 - 0.34 * x - 0.11 * r * x
        G1 = -1.22 - 0.12 * r
        G2 = 0.55 - 1.05 * r ** 0.75 + 0.47 * r ** 1.5
    else:
        z = c / a
        M1 = math.sqrt(z) * (1.0 + 0.04 * z)
        M2 = 0.2 * z ** 4
        M3 = -0.11 * z ** 4
        g = 1.0 + (0.1 + 0.35 * z * x ** 2) * (1.0 - st) ** 2
        ft = (z ** 2 * st ** 2 + ct ** 2) ** 0.25
        Phi = math.sqrt(1.0 + 1.464 * z ** 1.65)
        q = 0.2 + z + 0.6 * x
        H1 = 1.0 - (0.04 + 0.41 * z) * x + (0.55 - 1.93 * z ** 0.75 + 1.38 * z ** 1.5) * x ** 2
        G1 = -2.11 + 0.77 * z
        G2 = 0.55 - 0.72 * z ** 0.75 + 0.14 * z ** 1.5
    Mm = (M1 + M2 * x ** 2 + M3 * x ** 4) * g * ft / Phi
    H2 = 1.0 + G1 * x + G2 * x ** 2
    H = H1 + (H2 - H1) * st ** q
    return float(Mm), float(H * Mm)


def surface_fw(a, c, B, W):
    return float(math.sqrt(1.0 / math.cos((math.pi * c / W) * math.sqrt(a / B))))


def surface_sref(pm, pb, a, c, B, W, pin_jointed=False):
    if W >= 2.0 * (c + B):
        al = (a / B) / (1.0 + B / c)
    else:
        al = (a / B) * (2.0 * c / W)
    u = pb + 3.0 * pm * al if pin_jointed else pb
    return float((u + math.sqrt(u ** 2 + 9.0 * pm ** 2 * (1.0 - al) ** 2)) / (3.0 * (1.0 - al) ** 2))


def through_fw(a, W):
    return float(math.sqrt(1.0 / math.cos(math.pi * a / W)))


def through_sref(pm, pb, a, W):
    return float((pb + math.sqrt(pb ** 2 + 9.0 * pm ** 2)) / (3.0 * (1.0 - 2.0 * a / W)))


def k_primary_secondary(a, mm, mb, fw, pm, pb, qm, qb):
    scale = math.sqrt(math.pi * a)
    return float(fw * (mm * pm + mb * pb) * scale), float((mm * qm + mb * qb) * scale)


def radial_margin(lr, kr_value, sy, su, E, discontinuous=False):
    if lr < 0.0 or kr_value < 0.0:
        raise ValueError('The ray calculation requires nonnegative coordinates')
    if lr == 0.0:
        return math.inf if kr_value == 0.0 else float(1.0 / kr_value)
    cutoff = lr_max(sy, su)
    if cutoff <= 0.0:
        raise ValueError('Positive cutoff required')
    tc = cutoff / lr
    mu = min(0.001 * E / sy, 0.6)
    def low_curve(x):
        base = (1.0 + x * x / 2.0) ** -0.5
        return base if discontinuous else base * (0.3 + 0.7 * math.exp(-mu * x ** 6))
    def first_root(lo, hi, fn):
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if mid * kr_value >= fn(mid * lr):
                hi = mid
            else:
                lo = mid
        return float(hi)
    t1 = min(1.0 / lr, tc)
    if t1 * kr_value >= low_curve(t1 * lr):
        return first_root(0.0, t1, low_curve)
    if t1 == tc:
        return float(tc)
    f1 = option1_curve(1.0, sy, su, E, discontinuous)
    if t1 * kr_value >= f1:
        return float(t1)
    N = 0.3 * (1.0 - sy / su)
    exponent = (N - 1.0) / (2.0 * N)
    def high_curve(x):
        return f1 * x ** exponent
    if tc * kr_value >= high_curve(cutoff):
        return first_root(t1, tc, high_curve)
    return float(tc)


def assess_surface(B, W, a, two_c, sy, su, E, kmat, pm, pb, qm, qb, method='rho', pin_jointed=False):
    c = two_c / 2.0
    lr = surface_sref(pm, pb, a, c, B, W, pin_jointed) / sy
    fw = surface_fw(a, c, B, W)
    ratios = []
    margins = []
    for theta in (math.pi / 2.0, 0.0):
        mm, mb = surface_mm_mb(a, c, B, theta)
        kp, ks = k_primary_secondary(a, mm, mb, fw, pm, pb, qm, qb)
        ratio = kr(kp, ks, lr, kmat, a, sy, method)
        ratios.append(ratio)
        margins.append(radial_margin(lr, ratio, sy, su, E))
    return {'lr': float(lr), 'kr_deepest': ratios[0], 'kr_surface': ratios[1], 'F': float(min(margins))}

