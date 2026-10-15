"""Independent, numpy-only fixed-step Heun reference.

Units: lengths m, stresses Pa, time years. No threshold or event handling.
Run this module to print numerical oracle records with 15 significant figures.
The equations and cylinder table supplied in the request are the only sources.
"""
import numpy as np

PI = float(np.pi)


def _finite(x, name):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(name + ' must be finite')
    return x


def _positive(x, name):
    x = _finite(x, name)
    if x <= 0:
        raise ValueError(name + ' must be positive')
    return x


def finite_width(a, c, p, W):
    a, c, W = _positive(a, 'a'), _positive(c, 'c'), _positive(W, 'W')
    p = _finite(p, 'p')
    if p < 0 or 2*c/W > 0.8:
        raise ValueError('invalid finite-width geometry')
    effective_B = 2*a + 2*p
    angle = PI*c/W * float(np.sqrt(2*a/effective_B))
    cosine = float(np.cos(angle))
    if cosine <= 0:
        raise ValueError('nonpositive secant denominator')
    return float(np.sqrt(1/cosine))


def membrane(a, c, p, B, W, theta):
    a, c = _positive(a, 'a'), _positive(c, 'c')
    B, W = _positive(B, 'B'), _positive(W, 'W')
    p, theta = _finite(p, 'p'), _finite(theta, 'theta')
    if p < 0 or 2*a+p > B or not -PI <= theta <= PI:
        raise ValueError('invalid embedded geometry or angle')
    h = a/(2*c)
    if not 0 < h <= 1 or not 2*c/W < 0.5:
        raise ValueError('membrane applicability exceeded')
    effective_B = 2*a + 2*p
    r = a/c
    if h <= 0.1 and not a/effective_B < 0.625*(r+0.6):
        raise ValueError('low-aspect-ratio applicability exceeded')
    t = 2*a/effective_B
    radicand = 2.6 - 4*a/effective_B
    if radicand < 0:
        raise ValueError('negative membrane radicand')
    M1 = 1.0 if h <= 0.5 else float(np.sqrt(c/a))
    M2 = 0.05/(0.11+r**1.5)
    M3 = 0.29/(0.23+r**1.5)
    g = 1-t**4*float(np.sqrt(radicand))*abs(float(np.cos(theta)))/(1+4*r)
    s, co = float(np.sin(theta)), float(np.cos(theta))
    if h <= 0.5:
        f = (r*r*co*co+s*s)**0.25
    else:
        f = ((c/a)**2*s*s+co*co)**0.25
    q = min(r, 1/r)
    phi = float(np.sqrt(1+1.464*q**1.65))
    return float((M1+M2*t*t+M3*t**4)*g*f/phi)


# Axes are deliberately ascending. Each array is indexed by
# (a/c, a/B, B/r_i), with axes ([.2,.4], [.2,.4], [.1,.25]).
_DEEP = np.array([[[1.062, 1.045], [1.260, 1.240]],
                  [[0.932, 0.919], [1.016, 0.998]]], dtype=float)
_SURFACE = np.array([[[0.578, 0.577], [0.695, 0.698]],
                     [[0.676, 0.669], [0.768, 0.759]]], dtype=float)


def cylinder_factors(a_over_c, a_over_B, B_over_ri):
    x = _finite(a_over_c, 'a/c')
    y = _finite(a_over_B, 'a/B')
    z = _finite(B_over_ri, 'B/r_i')
    if not (0.2 <= x <= 0.4 and 0.2 <= y <= 0.4 and 0.1 <= z <= 0.25):
        raise ValueError('query outside supplied cylinder cell')
    tx, ty, tz = (x-0.2)/0.2, (y-0.2)/0.2, (z-0.1)/0.15
    wx, wy, wz = (1-tx, tx), (1-ty, ty), (1-tz, tz)
    values = []
    for table in (_DEEP, _SURFACE):
        v = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    v += wx[i]*wy[j]*wz[k]*float(table[i,j,k])
        values.append(float(v))
    return tuple(values)


def _parameters(A_si, m, cycles_per_year, h_years, n_steps):
    A = _positive(A_si, 'A_si')
    exponent = _positive(m, 'm')
    cycles = _positive(cycles_per_year, 'cycles_per_year')
    h = _positive(h_years, 'h_years')
    ns = _finite(n_steps, 'n_steps')
    n = int(ns)
    if ns != n or n < 0:
        raise ValueError('n_steps must be a nonnegative integer')
    return A, exponent, cycles, h, n


def _heun(a, c, velocity, h, n):
    for _ in range(n):
        va, vc = velocity(a, c)
        ap, cp = a+h*va, c+h*vc
        vap, vcp = velocity(ap, cp)
        a, c = a+h*(va+vap)/2, c+h*(vc+vcp)/2
    return float(a), float(c)


def integrate_embedded_heun(a0, c0, p0, B, W, dsigma, A_si, m,
                            cycles_per_year, h_years, n_steps):
    a, c = _positive(a0, 'a0'), _positive(c0, 'c0')
    p = _finite(p0, 'p0')
    B, W = _positive(B, 'B'), _positive(W, 'W')
    stress = _finite(dsigma, 'dsigma')
    if p < 0 or stress < 0:
        raise ValueError('negative ligament or stress range')
    A, exponent, cycles, h, n = _parameters(A_si, m, cycles_per_year, h_years, n_steps)
    effective_B = 2*a+2*p

    def velocity(aa, cc):
        pp = (effective_B-2*aa)/2
        width = finite_width(aa, cc, pp, W)
        scale = width*stress*float(np.sqrt(PI*aa))
        ka = membrane(aa, cc, pp, B, W, PI/2)*scale
        kc = membrane(aa, cc, pp, B, W, 0.0)*scale
        if ka < 0 or kc < 0:
            raise ValueError('negative stress-intensity range')
        return float(cycles*(A*ka**exponent)), float(cycles*(A*kc**exponent))

    a, c = _heun(a, c, velocity, h, n)
    return a, c, float((effective_B-2*a)/2)


def integrate_cylinder_surface_heun(a0, c0, B, r_i, dsigma, A_si, m,
                                    cycles_per_year, h_years, n_steps):
    a, c = _positive(a0, 'a0'), _positive(c0, 'c0')
    B, r_i = _positive(B, 'B'), _positive(r_i, 'r_i')
    stress = _finite(dsigma, 'dsigma')
    if stress < 0:
        raise ValueError('negative stress range')
    A, exponent, cycles, h, n = _parameters(A_si, m, cycles_per_year, h_years, n_steps)

    def velocity(aa, cc):
        aa, cc = _positive(aa, 'a'), _positive(cc, 'c')
        md, ms = cylinder_factors(aa/cc, aa/B, B/r_i)
        scale = stress*float(np.sqrt(PI*aa))
        ka, kc = md*scale, ms*scale
        return float(cycles*(A*ka**exponent)), float(cycles*(A*kc**exponent))

    return _heun(a, c, velocity, h, n)


CASE_SPECS = [
    ('embedded_1', 'integrate_embedded_heun', [2e-3,8e-3,5e-3,25e-3,0.5,100e6,1.4097e-28,2.88,1e5,1e-3,400]),
    ('embedded_2', 'integrate_embedded_heun', [1e-3,10e-3,2e-3,25e-3,0.5,80e6,1.4097e-28,2.88,2e5,2e-3,300]),
    ('embedded_3', 'integrate_embedded_heun', [3e-3,4e-3,6e-3,30e-3,0.6,120e6,1.64754666094773e-29,3.0,5e4,5e-3,200]),
    ('cylinder_4', 'integrate_cylinder_surface_heun', [5e-3,20e-3,20e-3,0.1,30e6,1.4097e-28,2.88,1e5,1e-2,300]),
    ('cylinder_5', 'integrate_cylinder_surface_heun', [4.5e-3,20e-3,20e-3,0.15,30e6,1.4097e-28,2.88,1e5,2e-2,200])
]


def tabulate():
    records = []
    for name, function, arguments in CASE_SPECS:
        value = globals()[function](*arguments)
        records.append({
            'name': name,
            'inputs': {'function': function, 'args': arguments},
            'expected': {'value': list(value),
                         'value_15_significant_figures': [format(v, '.15g') for v in value]},
            'how_obtained': 'Direct evaluation of the supplied factors and explicit predictor-corrector Heun recurrence; no event handling.'
        })
    return records


if __name__ == '__main__':
    # No imports beyond numpy. This prints valid JSON numeric oracle records.
    print('{"cases": [')
    for index, record in enumerate(tabulate()):
        value = record['expected']['value_15_significant_figures']
        suffix = ',' if index+1 < len(CASE_SPECS) else ''
        print('  {"case": %d, "value": [%s]}%s' % (index+1, ', '.join(value), suffix))
    print(']}')
