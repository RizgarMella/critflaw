import numpy as np

K1 = 0.95
SQRT2 = float(np.sqrt(2.0))


def _check_n(n):
    n = float(n)
    if not np.isfinite(n) or n <= 1.0:
        raise ValueError('This circular-transition construction requires n > 1.')
    return n


def arc(n):
    """Return (K2, xc, yc, r), using a one-dimensional search in r.

    Put a = n*t**(n-1), where t is the contact stress. The outward
    radius vector at contact is r*(-1, a)/sqrt(1+a*a). Consequently
    tangency requires a*(t**n-xc)+(t-yc)=0. For each trial radius,
    solve this normality equation, then bisect distance(center,contact)-r.
    """
    n = _check_n(n)

    def contact(r):
        xc = K1 + r / SQRT2
        yc = K1 - r / SQRT2
        lo = 1.0
        hi = float(np.exp(np.log(xc) / n))
        for unused in range(100):
            t = (lo + hi) / 2.0
            a = n * t ** (n - 1.0)
            f = a * (t ** n - xc) + t - yc
            if f <= 0.0:
                lo = t
            else:
                hi = t
        t = (lo + hi) / 2.0
        residual = float(np.hypot(t ** n - xc, t - yc)) - r
        return t, residual

    # At this radius the normality equation has its contact at t=1.
    lo = (1.0 - K1) * SQRT2 * (n + 1.0) / (n - 1.0)
    hi = 2.0 * lo
    for unused in range(200):
        if contact(hi)[1] <= 0.0:
            break
        hi *= 2.0
    else:
        raise ArithmeticError('Could not bracket the transition radius.')

    for unused in range(100):
        mid = (lo + hi) / 2.0
        if contact(mid)[1] > 0.0:
            lo = mid
        else:
            hi = mid
    r = (lo + hi) / 2.0
    t = contact(r)[0]
    return float(t), float(K1 + r / SQRT2), float(K1 - r / SQRT2), float(r)


def strain(y, n):
    """Normalized total strain; scalars in, Python float out."""
    n = _check_n(n)
    y = float(y)
    if not np.isfinite(y):
        return float('nan')
    if y <= K1:
        return float(y)
    K2, xc, yc, r = arc(n)
    if y >= K2:
        return float(y ** n)
    # The selected arc is the LEFT branch when x is expressed in terms
    # of y; equivalently it is the UPPER branch when y is a function of x.
    z = r * r - (y - yc) ** 2
    if z < 0.0:
        if z >= -64.0 * np.finfo(float).eps * r * r:
            z = 0.0
        else:
            raise ArithmeticError('Negative circle radicand.')
    return float(xc - np.sqrt(z))


def plastic_compliance(s, n):
    n = _check_n(n)
    s = float(s)
    if s == 0.0 or not np.isfinite(s):
        return float('nan')
    return float((strain(s, n) - s) / s)


def _check_ratios(*ratios):
    for r in ratios:
        if not np.isfinite(r) or r <= 1.0:
            raise ValueError('Mesh refinement ratios must be finite and greater than one.')


def _difference_ratio(p, r21, r32):
    a = p * float(np.log(r21))
    b = p * float(np.log(r32))
    return float(np.expm1(a) / (-np.expm1(-b)))


def observed_order(d12, d23, r21, r32):
    d12, d23 = float(d12), float(d23)
    r21, r32 = float(r21), float(r32)
    _check_ratios(r21, r32)
    if not np.isfinite(d12) or not np.isfinite(d23) or d23 == 0.0:
        return float('nan')
    ratio = d12 / d23
    if not np.isfinite(ratio):
        return float('nan')
    lo, hi = 0.05, 6.0
    lower = _difference_ratio(lo, r21, r32)
    upper = _difference_ratio(hi, r21, r32)
    if ratio < lower or ratio > upper:
        return float('nan')
    if ratio == lower:
        return float(lo)
    if ratio == upper:
        return float(hi)
    for unused in range(100):
        mid = (lo + hi) / 2.0
        if _difference_ratio(mid, r21, r32) < ratio:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def predicted_ratio(p, r32, r43):
    p, r32, r43 = float(p), float(r32), float(r43)
    _check_ratios(r32, r43)
    if not np.isfinite(p) or p <= 0.0:
        return float('nan')
    b = p * float(np.log(r32))
    c = p * float(np.log(r43))
    return float(np.exp(-b) * (-np.expm1(-c)) / (-np.expm1(-b)))


def coarse_error(J1, J2, p, r21, Fs=1.25):
    J1, J2, p, r21, Fs = map(float, (J1, J2, p, r21, Fs))
    _check_ratios(r21)
    if not all(np.isfinite(v) for v in (J1, J2, p, Fs)) or p <= 0.0:
        return float('nan')
    denominator = -float(np.expm1(-p * np.log(r21)))
    return float(Fs * abs(J1 - J2) / denominator)


def synthetic_sequence(p, Jstar=100.0, C=4.0, h1=1.0,
                       r21=1.25, r32=1.2, r43=4.0/3.0, Fs=1.25):
    p = float(p)
    h2 = h1 / r21
    h3 = h2 / r32
    h4 = h3 / r43
    J1, J2, J3, J4 = [float(Jstar + C * h ** p)
                       for h in (h1, h2, h3, h4)]
    d12, d23, d34 = J1-J2, J2-J3, J3-J4
    return {
        'J': [J1, J2, J3, J4],
        'observed_order': observed_order(d12, d23, r21, r32),
        'predicted_ratio': predicted_ratio(p, r32, r43),
        'realized_ratio': float(d34 / d23),
        'coarse_error_without_safety': coarse_error(J1, J2, p, r21, 1.0),
        'coarse_error': coarse_error(J1, J2, p, r21, Fs)
    }


def _print_value(label, value):
    if isinstance(value, dict):
        for key, item in value.items():
            _print_value(label + '.' + key, item)
    elif isinstance(value, (list, tuple)):
        print(label + ' = (' + ', '.join(format(float(v), '.15g') for v in value) + ')')
    else:
        print(label + ' = ' + format(float(value), '.15g'))


def run_requested_cases():
    results = []
    for n in (3, 5, 10, 20):
        results.append(('arc(' + str(n) + ')', arc(n)))
    for y, n in ((0.9, 10), (0.97, 10), (1.0, 10), (1.05, 10), (1.2, 5)):
        results.append(('strain(' + str(y) + ',' + str(n) + ')', strain(y, n)))
    for s, n in ((0.5, 10), (0.99, 10), (1.3, 10)):
        results.append(('plastic_compliance(' + str(s) + ',' + str(n) + ')', plastic_compliance(s, n)))
    for p in (2.0, 0.8, 3.0):
        results.append(('synthetic_sequence(' + str(p) + ')', synthetic_sequence(p)))
    results.append(('observed_order_no_solution', observed_order(1.0, 1.0, 1.25, 1.2)))
    for label, value in results:
        _print_value(label, value)
    return results


if __name__ == '__main__':
    run_requested_cases()
