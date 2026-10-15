'''Fixed-step Heun fatigue integrators, using the supplied equations.

Lengths: m; stresses: Pa; dK: Pa sqrt(m); time: years.
Only third-party dependency: numpy. Public integrators return Python floats.

Run tabulate() to generate numerical expectations, formatted to 15
significant figures. Domain errors are reported rather than extrapolating
sqrt(sec(x)) outside its real, positive branch.
'''
import numpy as np

PI = float(np.pi)


def _positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + ' must be finite and positive')
    return x


def surface_flaw_dk(a, c, B, W, dsm, dsb, phi):
    a = _positive(a, 'a')
    c = _positive(c, 'c')
    B = _positive(B, 'B')
    W = _positive(W, 'W')
    dsm, dsb, phi = float(dsm), float(dsb), float(phi)
    if not all(np.isfinite(x) for x in (dsm, dsb, phi)):
        raise ValueError('stresses and phi must be finite')
    if not (a <= c and a < B and 0.0 <= phi <= PI / 2.0):
        raise ValueError('requires a <= c, a < B, and 0 <= phi <= pi/2')
    r, u = a / c, a / B
    width_angle = PI * c / W * float(np.sqrt(u))
    if width_angle >= PI / 2.0:
        raise ValueError('finite-width correction is outside its real branch')
    Q = 1.0 + 1.464 * r**1.65
    M1 = 1.13 - 0.09 * r
    M2 = -0.54 + 0.89 / (0.2 + r)
    M3 = 0.5 - 1.0 / (0.65 + r) + 14.0 * (1.0 - r)**24
    s, co = float(np.sin(phi)), float(np.cos(phi))
    g = 1.0 + (0.1 + 0.35 * u * u) * (1.0 - s)**2
    f_phi = (r * r * co * co + s * s)**0.25
    f_w = float(np.sqrt(1.0 / float(np.cos(width_angle))))
    F = (M1 + M2 * u * u + M3 * u**4) * g * f_phi * f_w
    H1 = 1.0 - 0.34 * u - 0.11 * r * u
    G1 = -1.22 - 0.12 * r
    G2 = 0.55 - 1.05 * r**0.75 + 0.47 * r**1.5
    H2 = 1.0 + G1 * u + G2 * u * u
    p = 0.2 + r + 0.6 * u
    H = H1 + (H2 - H1) * s**p
    return float((dsm + H * dsb) * float(np.sqrt(PI * a / Q)) * F)


def _step_parameters(A_si, m, cycles_per_year, h_years, n_steps):
    A_si = _positive(A_si, 'A_si')
    m = _positive(m, 'm')
    cycles_per_year = _positive(cycles_per_year, 'cycles_per_year')
    h_years = _positive(h_years, 'h_years')
    n = int(n_steps)
    if n != n_steps or n < 0:
        raise ValueError('n_steps must be a nonnegative integer')
    return A_si, m, cycles_per_year, h_years, n


def integrate_heun(a0, c0, B, W, dsm, dsb, A_si, m,
                   cycles_per_year, h_years, n_steps):
    A_si, m, cycles_per_year, h, n = _step_parameters(
        A_si, m, cycles_per_year, h_years, n_steps)
    a, c = _positive(a0, 'a0'), _positive(c0, 'c0')

    def velocity(a, c):
        ka = surface_flaw_dk(a, c, B, W, dsm, dsb, PI / 2.0)
        kc = surface_flaw_dk(a, c, B, W, dsm, dsb, 0.0)
        if ka < 0.0 or kc < 0.0:
            raise ValueError('negative stress-intensity range')
        return (float(cycles_per_year * (A_si * ka**m)),
                float(cycles_per_year * (A_si * kc**m)))

    for _ in range(n):
        va0, vc0 = velocity(a, c)
        ap, cp = a + h * va0, c + h * vc0
        va1, vc1 = velocity(ap, cp)
        a = a + h * (va0 + va1) / 2.0
        c = c + h * (vc0 + vc1) / 2.0
    return float(a), float(c)


def through_flaw_dk(c, W, dsigma):
    c, W = _positive(c, 'c'), _positive(W, 'W')
    dsigma = _positive(dsigma, 'dsigma')
    if c >= W / 2.0:
        raise ValueError('through-flaw secant is outside its real positive branch')
    secant = 1.0 / float(np.cos(PI * c / W))
    return float(float(np.sqrt(secant)) * dsigma * float(np.sqrt(PI * c)))


def integrate_through_heun(c0, W, dsigma, A_si, m,
                           cycles_per_year, h_years, n_steps):
    A_si, m, cycles_per_year, h, n = _step_parameters(
        A_si, m, cycles_per_year, h_years, n_steps)
    c = _positive(c0, 'c0')

    def velocity(c):
        dk = through_flaw_dk(c, W, dsigma)
        return float(cycles_per_year * (A_si * dk**m))

    for _ in range(n):
        v0 = velocity(c)
        cp = c + h * v0
        v1 = velocity(cp)
        c = c + h * (v0 + v1) / 2.0
    return float(c)


CASE_SPECS = [
    ('case_1', 'integrate_heun', [2e-3, 5e-3, 25e-3, 0.5, 100e6, 20e6, 1.4097e-28, 2.88, 2e5, 1e-3, 500]),
    ('case_2', 'integrate_heun', [1e-3, 6e-3, 12e-3, 0.3, 120e6, 0.0, 1.4097e-28, 2.88, 1e5, 5e-3, 400]),
    ('case_3', 'integrate_heun', [3e-3, 3e-3, 25e-3, 0.5, 80e6, 40e6, 1.4097e-28, 2.88, 5e4, 2e-3, 1000]),
    ('case_4', 'integrate_through_heun', [5e-3, 0.5, 100e6, 1.64754666094773e-29, 3.0, 1e6, 1e-3, 100]),
    ('case_5', 'integrate_through_heun', [2e-2, 0.2, 150e6, 1.4097e-28, 2.88, 1e5, 1e-2, 200])
]


def tabulate():
    '''Generate numerical values and end-state validity checks.

    A value_15_significant_figures entry contains decimal strings so that
    trailing significant zeros survive serialization.
    '''
    out = []
    for name, fn, args in CASE_SPECS:
        entry = {'case': name}
        try:
            result = globals()[fn](*args)
            values = list(result) if isinstance(result, tuple) else [result]
            entry['value'] = values
            entry['value_15_significant_figures'] = [format(x, '.15g') for x in values]
            if fn == 'integrate_heun':
                a, c = values
                B, W = args[2], args[3]
                checks = {'a_less_than_B': a < B,
                          'a_over_c_at_most_one': a / c <= 1.0,
                          'two_c_over_W_less_than_0_8': 2.0 * c / W < 0.8,
                          'two_c_less_than_W_over_two': 2.0 * c < W / 2.0}
            else:
                c, W = values[0], args[1]
                checks = {'two_c_over_W_less_than_0_8': 2.0 * c / W < 0.8,
                          'two_c_less_than_W_over_two': 2.0 * c < W / 2.0,
                          'secant_on_real_positive_branch': c < W / 2.0}
            entry['end_state_checks'] = checks
            entry['all_applicable_end_state_limits_satisfied'] = all(checks.values())
        except ValueError as exc:
            entry['error_type'] = 'ValueError'
            entry['message'] = str(exc)
            entry['end_state_checks'] = None
        out.append(entry)
    return out
