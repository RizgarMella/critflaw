"""Independent extended-flaw fixed-step Heun reference.

Dependencies: numpy only. Units: metres, Pa, cycles, years.
No extrapolation, clamping, event handling, width factor, or bulging factor.
Running this module computes and prints the four requested oracle records.
Steps in error messages are numbered starting at 1.
"""
import numpy as np

PI = float(np.pi)

_MM = np.array([[1.930, 1.784],
                [2.960, 2.566],
                [4.820, 3.461]], dtype=float)
_MB = np.array([[1.143, 1.033],
                [1.484, 1.094],
                [1.990, 0.949]], dtype=float)


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


def extended_factors(a_over_B, B_over_ri):
    """Return (M_m, M_b), bilinearly interpolated in the containing cell."""
    x = _finite(a_over_B, 'a/B')
    z = _finite(B_over_ri, 'B/r_i')
    if not (0.4 <= x <= 0.8 and 0.1 <= z <= 0.25):
        raise ValueError('query outside supplied extended-flaw grid: '
                         'a/B=%.15g, B/r_i=%.15g' % (x, z))
    if x <= 0.6:
        row, lo, hi = 0, 0.4, 0.6
    else:
        row, lo, hi = 1, 0.6, 0.8
    tx = (x-lo)/(hi-lo)
    tz = (z-0.1)/(0.25-0.1)
    factors = []
    for table in (_MM, _MB):
        lower = (1-tz)*float(table[row, 0]) + tz*float(table[row, 1])
        upper = (1-tz)*float(table[row+1, 0]) + tz*float(table[row+1, 1])
        factors.append(float((1-tx)*lower + tx*upper))
    return tuple(factors)


def integrate_extended_heun(a0, B, r_i, dsigma_m, dsigma_b,
                            A_si, m, cycles_per_year, h_years, n_steps):
    a = _positive(a0, 'a0')
    B = _positive(B, 'B')
    r_i = _positive(r_i, 'r_i')
    sm = _finite(dsigma_m, 'dsigma_m')
    sb = _finite(dsigma_b, 'dsigma_b')
    if sm < 0 or sb < 0:
        raise ValueError('negative stress range')
    A, exponent, cycles, h, n = _parameters(
        A_si, m, cycles_per_year, h_years, n_steps)
    curvature = B/r_i

    def velocity(aa):
        mm, mb = extended_factors(aa/B, curvature)
        dk = (mm*sm + mb*sb)*float(np.sqrt(PI*aa))
        return float(cycles*(A*dk**exponent))

    if n == 0:
        extended_factors(a/B, curvature)
        return float(a)
    for step in range(1, n+1):
        try:
            v0 = velocity(a)
        except ValueError as exc:
            raise ValueError('step %d, current: %s' % (step, exc)) from None
        a_pred = a+h*v0
        try:
            v1 = velocity(a_pred)
        except ValueError as exc:
            raise ValueError('step %d, predictor: %s' % (step, exc)) from None
        a = a+h*(v0+v1)/2
    return float(a)


CASE_SPECS = [
    ('extended_1', [8.2e-3, 20e-3, 0.1, 30e6, 0.0,
                    1.4097e-28, 2.88, 1e5, 1e-2, 200]),
    ('extended_2', [8.2e-3, 20e-3, 0.1, 20e6, 20e6,
                    1.4097e-28, 2.88, 1e5, 1e-2, 200]),
    ('extended_3', [9e-3, 20e-3, 0.15, 25e6, 10e6,
                    1.4097e-28, 2.88, 2e5, 5e-3, 300]),
    ('extended_4', [12.5e-3, 25e-3, 0.125, 0.0, 40e6,
                    1.64754666094773e-29, 3.0, 5e4, 2e-2, 100])
]


def tabulate():
    records = []
    for name, args in CASE_SPECS:
        initial = extended_factors(args[0]/args[1], args[1]/args[2])
        expected = {
            'initial_factors': list(initial),
            'value': None,
            'value_15_significant_figures': None,
            'error': None,
            'error_step': None
        }
        try:
            value = integrate_extended_heun(*args)
            expected['value'] = value
            expected['value_15_significant_figures'] = format(value, '.15g')
        except ValueError as exc:
            error = str(exc)
            expected['error'] = error
            if error.startswith('step '):
                expected['error_step'] = int(error.split(',', 1)[0].split()[1])
        records.append({
            'name': name,
            'inputs': {'function': 'integrate_extended_heun', 'args': args},
            'expected': expected,
            'how_obtained': 'Bilinear interpolation of the supplied printed '
                            'grid followed by the supplied fixed-step Heun '
                            'recurrence; both current and predictor queries '
                            'are checked without clamping or extrapolation.'
        })
    return records


if __name__ == '__main__':
    # No JSON-library dependency; these error messages contain no quotes.
    records = tabulate()
    print('{"cases": [')
    for index, record in enumerate(records):
        result = record['expected']
        value = result['value_15_significant_figures']
        value_text = 'null' if value is None else value
        error_text = 'null' if result['error'] is None else '"'+result['error']+'"'
        suffix = ',' if index+1 < len(records) else ''
        print('  {"case": %d, "value": %s, "error": %s}%s' %
              (index+1, value_text, error_text, suffix))
    print(']}')
