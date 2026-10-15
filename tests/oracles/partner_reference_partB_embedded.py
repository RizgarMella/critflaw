"""Independent reference for the supplied Part B equations.

Only numpy is imported. No external implementation or primary-reference text
was available. Public equation functions return Python floats and raise
ValueError for invalid inputs. evaluate() executes the batched oracle cases.

Lengths must use consistent units. Stress and moment signs are retained.
Zero flaw size is accepted as a limiting value in the shell equations where
the displayed equation is defined, but not in the embedded-flaw equations.

Geometry checks deliberately do NOT enforce p <= (B-2a)/2: that interpretation
of 'nearer' would invalidate some explicitly requested bending samples.
They do enforce 2a+p <= B. See the accompanying doubts.
"""
import numpy as np

PI = float(np.pi)


def _finite(*values):
    try:
        values = tuple(float(x) for x in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Expected finite real scalars') from exc
    if not all(np.isfinite(x) for x in values):
        raise ValueError('Expected finite real scalars')
    return values


def _out(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Nonfinite result')
    return value


def _geometry(a, c, p, B):
    a, c, p, B = _finite(a, c, p, B)
    if a <= 0 or c <= 0 or B <= 0 or p < 0:
        raise ValueError('Invalid embedded-flaw dimensions')
    if 2*a + p > B:
        raise ValueError('Flaw and stated ligament do not fit in plate')
    return a, c, p, B


def _phi(a, c):
    r = a/c
    if r > 1:
        r = 1/r
    return float(np.sqrt(1 + 1.464*r**1.65))


def finite_width(a, c, p, W):
    a, c, p, W = _finite(a, c, p, W)
    if a <= 0 or c <= 0 or p < 0 or W <= 0:
        raise ValueError('Invalid finite-width dimensions')
    if 2*c/W > 0.8:
        raise ValueError('Finite-width applicability exceeded')
    effective_B = 2*a + 2*p
    angle = PI*c/W * np.sqrt(2*a/effective_B)
    cosine = float(np.cos(angle))
    if cosine <= 0:
        raise ValueError('Nonpositive secant denominator')
    return _out(np.sqrt(1/cosine))


def membrane(a, c, p, B, W, theta, form='general'):
    a, c, p, B = _geometry(a, c, p, B)
    W, theta = _finite(W, theta)
    if W <= 0 or not (-PI <= theta <= PI):
        raise ValueError('Invalid width or angle')
    h = a/(2*c)
    if not (0 < h <= 1):
        raise ValueError('Membrane aspect-ratio applicability exceeded')
    if not (2*c/W < 0.5):
        raise ValueError('Membrane width applicability exceeded')
    effective_B = 2*a + 2*p
    if h <= 0.1 and not (a/effective_B < 0.625*(a/c + 0.6)):
        raise ValueError('Low-aspect-ratio applicability exceeded')
    r = a/c
    t = 2*a/effective_B
    M1 = 1.0 if h <= 0.5 else np.sqrt(c/a)
    M2 = 0.05/(0.11 + r**1.5)
    M3 = 0.29/(0.23 + r**1.5)
    loss = t**4*np.sqrt(2.6 - 4*a/effective_B)/(1 + 4*r)
    g = 1 - loss*abs(np.cos(theta))
    if h <= 0.5:
        ftheta = (r*r*np.cos(theta)**2 + np.sin(theta)**2)**0.25
    else:
        ftheta = ((c/a)**2*np.sin(theta)**2 + np.cos(theta)**2)**0.25
    general = (M1 + M2*t*t + M3*t**4)*g*ftheta/_phi(a, c)
    if form == 'general':
        return _out(general)
    if form == 'minor':
        if theta != PI/2:
            raise ValueError('Minor simplified form requires theta = pi/2')
        gs = 1.0
        fs = 1.0 if h <= 0.5 else np.sqrt(c/a)
    elif form == 'major':
        if theta != 0:
            raise ValueError('Major simplified form requires theta = 0')
        gs = 1 - loss
        fs = np.sqrt(a/c) if h <= 0.5 else 1.0
    else:
        raise ValueError('Unknown membrane form')
    simplified = (M1 + M2*t*t + M3*t**4)*gs*fs/_phi(a, c)
    assert np.isclose(general, simplified, rtol=2e-14, atol=2e-15)
    return _out(simplified)


def bending(a, c, p, B):
    a, c, p, B = _geometry(a, c, p, B)
    if not (0 < a/(2*c) <= 0.5):
        raise ValueError('Bending aspect-ratio applicability exceeded')
    q, s = a/B, p/B
    if s <= 0.1841:
        L1, L2, L3, L4 = 1.044, -2.44, 0.0, -3.166
    elif q <= 0.125:
        L1, L2, L3, L4 = 0.94, -1.875, -0.1146, -1.844
    else:
        L1, L2, L3, L4 = 1.06, -2.20, -0.6666, -0.6666
    return _out((L1 + L2*s + L3*q + L4*s*q)/_phi(a, c))


def embedded_reference(a, c, p, B, W, P_m, P_b):
    a, c, p, B = _geometry(a, c, p, B)
    W, P_m, P_b = _finite(W, P_m, P_b)
    if W <= 0:
        raise ValueError('Nonpositive width')
    if W >= 2*(c+B):
        alpha = (2*a/B)/(1+B/c)
    else:
        alpha = (2*a/B)*(2*c/W)
    D = (1-alpha)**2 + 4*p*alpha/B
    if D <= 0:
        raise ValueError('Nonpositive reference-stress denominator')
    u = P_b + 3*P_m*alpha
    return _out((u + np.sqrt(u*u + 9*P_m*P_m*D))/(3*D))


def extended_reference(a, p, B, P_m, P_b):
    a, p, B, P_m, P_b = _finite(a, p, B, P_m, P_b)
    if a <= 0 or p < 0 or B <= 0 or 2*a+p > B:
        raise ValueError('Invalid extended-flaw dimensions')
    if P_m == 0:
        raise ValueError('Displayed lambda is undefined for P_m = 0')
    alpha = a/B
    k = (B/2-p-a)/B
    if alpha > k/2:
        raise ValueError('No supplied solution for alpha > alpha2')
    lam = P_b/(6*P_m)
    c3 = 1 - 8*alpha*k - 4*alpha*alpha
    if c3 <= 0:
        raise ValueError('Nonpositive c3')
    u = 2*(lam+alpha)
    return _out((u + np.sqrt(u*u+c3))/c3*P_m)


def bulging(half_length, r_m, B):
    """Both M_through_axial and M_T; their displayed formulas are identical."""
    half_length, r_m, B = _finite(half_length, r_m, B)
    if half_length < 0 or r_m <= 0 or B <= 0:
        raise ValueError('Invalid bulging dimensions')
    return _out(np.sqrt(1 + 1.6*half_length**2/(r_m*B)))


def surface_axial(a, c, r_m, B):
    a, c, r_m, B = _finite(a, c, r_m, B)
    if B <= 0 or r_m <= 0 or c < 0 or not (0 <= a < B):
        raise ValueError('Invalid surface-flaw dimensions')
    MT = bulging(c, r_m, B)
    return _out((1-a/(B*MT))/(1-a/B))


def shell_lambda(a, r_m, B, nu):
    a, r_m, B, nu = _finite(a, r_m, B, nu)
    if a < 0 or r_m <= 0 or B <= 0 or abs(nu) >= 1:
        raise ValueError('Invalid lambda dimensions or Poisson ratio')
    return _out((12*(1-nu*nu))**0.25*a/np.sqrt(r_m*B))


def shell_beta(a, r_m):
    a, r_m = _finite(a, r_m)
    if a <= 0 or r_m <= 0:
        raise ValueError('Displayed beta requires positive lengths')
    x = a/(2*r_m)
    # Literal real-domain reading, not an unstated first-tangent-lobe limit.
    # Reject floating-point representations of tangent poles explicitly.
    if abs(np.cos(x)) <= 1e-14:
        raise ValueError('Tangent pole')
    radicand = np.tan(x)/x
    if radicand < 0:
        raise ValueError('Negative beta radicand')
    return _out(np.sqrt(radicand))


def moment_membrane(M_global, r_i, a, B):
    M_global, r_i, a, B = _finite(M_global, r_i, a, B)
    if r_i <= 0 or B <= 0 or not (0 <= a <= B):
        raise ValueError('Invalid internal-surface-flaw dimensions')
    # Algebraically identical to (r_i+B)**4-r_i**4, without subtracting
    # nearly equal fourth powers.
    D = B*(4*r_i**3 + 6*r_i*r_i*B + 4*r_i*B*B + B**3)
    return _out(4*M_global*(r_i+a)/(PI*D))


def thin_axial_reference(a, r_i, B, W, P_m, P_b):
    a, r_i, B, W, P_m, P_b = _finite(a, r_i, B, W, P_m, P_b)
    if a < 0 or r_i <= 0 or B <= 0 or W <= 0 or not (2*a < W):
        raise ValueError('Invalid thin-wall axial dimensions')
    MTprime = np.sqrt(1 + 1.6*a*a/(r_i*B))
    return _out(1.2*MTprime*P_m + 2*P_b/(3*(1-2*a/W)))


_FUNCTIONS = {
    'finite_width': finite_width,
    'membrane': membrane,
    'bending': bending,
    'embedded_reference': embedded_reference,
    'extended_reference': extended_reference,
    'bulging': bulging,
    'surface_axial': surface_axial,
    'shell_lambda': shell_lambda,
    'shell_beta': shell_beta,
    'moment_membrane': moment_membrane,
    'thin_axial_reference': thin_axial_reference,
}


def evaluate(function, samples):
    """Batch adapter: positional sample arrays, with results in the same order.

    Equation functions themselves raise ValueError. This adapter records those
    exceptions as {'error': 'ValueError'} for comparison with JSON fixtures.
    Numeric fixture values are rounded or truncated only after calculation;
    compare with rtol=5e-10, atol=5e-12, rather than exact float equality.
    """
    if function not in _FUNCTIONS:
        raise ValueError('Unknown function')
    results = []
    for arguments in samples:
        try:
            results.append(_FUNCTIONS[function](*arguments))
        except ValueError:
            results.append({'error': 'ValueError'})
    return {'results': results}
