"""Independent fatigue-growth oracle; standard-library math only.

All public numerical functions return Python floats, except life_fixed_step,
which raises because the supplied specification does not determine a cycle
step or a unique numerical result. No cycle increment is silently invented.

Units:
  rate_mm_per_cycle: dK in N/mm**(3/2), rate in mm/cycle.
  to_si: coefficient for dK in Pa*sqrt(m), rate in m/cycle.
  cycles_through_crack: lengths in m and stress range in Pa.

Newman-Raju surface-flaw equations used by surface_flaw_dk, for 0 < a/c <= 1:
  r = a/c; u = a/B
  Q = 1 + 1.464*r**1.65
  M1 = 1.13 - 0.09*r
  M2 = -0.54 + 0.89/(0.2 + r)
  M3 = 0.5 - 1/(0.65 + r) + 14*(1-r)**24
  g = 1 + (0.1 + 0.35*u**2)*(1-sin(phi))**2
  f_phi = (r**2*cos(phi)**2 + sin(phi)**2)**0.25
  f_w = sqrt(sec(pi*c/W*sqrt(u)))
  F = (M1 + M2*u**2 + M3*u**4)*g*f_phi*f_w
  H1 = 1 - 0.34*u - 0.11*r*u
  G1 = -1.22 - 0.12*r
  G2 = 0.55 - 1.05*r**0.75 + 0.47*r**1.5
  H2 = 1 + G1*u + G2*u**2
  p = 0.2 + r + 0.6*u
  H = H1 + (H2-H1)*sin(phi)**p
  dK = (dsm + H*dsb)*sqrt(pi*a/Q)*F
Here W is the full plate width. The deepest point has phi=pi/2 and
surface point phi=0. No extrapolation to a/c>1 is supplied.
Reference: Newman and Raju, Engineering Fracture Mechanics 15 (1981),
185-192. These equations include bending's H factor, not just the
membrane geometrical correction.

Case representation:
  inputs.function names the callable; inputs.arguments is its positional
  argument list. An expected object with transform='log10' specifies the
  base-10 logarithm of the returned scalar. This avoids discarding accuracy
  when tabulating extremely small coefficients and supplies a directly
  executable scalar oracle. Values without a transform are direct outputs.
  oracle_matches implements this convention, including expected exceptions.
"""

import math

LIFE_AMBIGUITY = "life_fixed_step is underdetermined: supply a cycle increment or an explicit rule for selecting it"

CATALOGUE = {
    "air": {
        "low_mean": ((1.21e-26, 8.16), (3.98e-13, 2.88)),
        "low_upper": ((4.37e-26, 8.16), (6.77e-13, 2.88)),
        "high_mean": ((4.80e-18, 5.10), (5.86e-13, 2.88)),
        "high_upper": ((2.10e-17, 5.10), (1.29e-12, 2.88)),
    },
    "freely_corroding": {
        "low_mean": ((3.0e-14, 3.42), (1.27e-7, 1.30)),
        "low_upper": ((8.55e-14, 3.42), (1.93e-7, 1.30)),
        "high_mean": ((5.37e-14, 3.42), (5.67e-7, 1.11)),
        "high_upper": ((1.72e-13, 3.42), (7.48e-7, 1.11)),
    },
    "cp_850": {
        "low_mean": ((1.21e-26, 8.16), (5.16e-12, 2.67)),
        "low_upper": ((4.37e-26, 8.16), (1.32e-11, 2.67)),
        "high_mean": ((4.80e-18, 5.10), (6.0e-12, 2.67)),
        "high_upper": ((2.10e-17, 5.10), (2.02e-11, 2.67)),
    },
    "cp_1100": {
        "low_mean": ((1.21e-26, 8.16), (5.51e-8, 1.40)),
        "low_upper": ((4.37e-26, 8.16), (9.24e-8, 1.40)),
        "high_mean": ((4.80e-18, 5.10), (5.25e-8, 1.40)),
        "high_upper": ((2.10e-17, 5.10), (1.02e-7, 1.40)),
    },
}

SD_LOG10 = {
    "air": (0.279, 0.320, 0.115, 0.171),
    "freely_corroding": (0.227, 0.253, 0.091, 0.060),
    "cp_850": (0.279, 0.320, 0.204, 0.264),
    "cp_1100": (0.279, 0.320, 0.112, 0.144),
}


def _finite(x, name):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(name + " must be finite")
    return x


def _positive(x, name):
    x = _finite(x, name)
    if x <= 0.0:
        raise ValueError(name + " must be positive")
    return x


def _stages(stages):
    if len(stages) not in (1, 2):
        raise ValueError("stages must contain one or two pairs")
    out = tuple((_positive(A, "A"), _positive(m, "m")) for A, m in stages)
    if len(out) == 2 and out[0][1] <= out[1][1]:
        raise ValueError("two-stage law requires m_A > m_B")
    return out


def rate_mm_per_cycle(dk_nmm, stages, dk0):
    dk = _finite(dk_nmm, "dk_nmm")
    threshold = _finite(dk0, "dk0")
    pairs = _stages(stages)
    if dk < 0.0 or threshold < 0.0:
        raise ValueError("dK and threshold must be nonnegative")
    if dk < threshold:
        return 0.0
    return float(min(A * dk**m for A, m in pairs))


def transition_point(A_a, m_a, A_b, m_b):
    A_a = _positive(A_a, "A_a")
    A_b = _positive(A_b, "A_b")
    m_a = _positive(m_a, "m_a")
    m_b = _positive(m_b, "m_b")
    if m_a <= m_b:
        raise ValueError("transition requires m_a > m_b")
    return float((A_b / A_a) ** (1.0 / (m_a - m_b)))


def upper_from_mean(A_mean, sd_log10):
    A_mean = _positive(A_mean, "A_mean")
    sd = _finite(sd_log10, "sd_log10")
    if sd < 0.0:
        raise ValueError("standard deviation must be nonnegative")
    return float(A_mean * 10.0 ** (2.0 * sd))


def to_si(A_nmm, m):
    A_nmm = _positive(A_nmm, "A_nmm")
    m = _positive(m, "m")
    # 1 N/mm**(3/2) = 10**4.5 Pa*sqrt(m); 1 mm = 10**-3 m.
    return float(A_nmm * 10.0 ** (-3.0 - 4.5 * m))


def threshold_unwelded_steel(R):
    R = _finite(R, "R")
    if R < 0.0:
        return 170.0
    if R < 0.5:
        return float(170.0 - 214.0 * R)
    return 63.0


def cycles_through_crack(a_i, a_f, dsigma, A, m):
    a_i = _positive(a_i, "a_i")
    a_f = _positive(a_f, "a_f")
    dsigma = _positive(dsigma, "dsigma")
    A = _positive(A, "A")
    m = _positive(m, "m")
    if a_f < a_i:
        raise ValueError("a_f must be at least a_i")
    if a_f == a_i:
        return 0.0
    prefactor = A * dsigma**m * math.pi ** (m / 2.0)
    if m == 2.0:
        return float(math.log(a_f / a_i) / prefactor)
    q = 1.0 - m / 2.0
    # Algebraically (a_f**q-a_i**q)/q; expm1 avoids near-m=2 cancellation.
    integral = a_i**q * math.expm1(q * math.log(a_f / a_i)) / q
    return float(integral / prefactor)


def surface_flaw_dk(a, c, B, W, dsm, dsb, phi):
    a = _positive(a, "a")
    c = _positive(c, "c")
    B = _positive(B, "B")
    W = _positive(W, "W")
    dsm = _finite(dsm, "dsm")
    dsb = _finite(dsb, "dsb")
    phi = _finite(phi, "phi")
    if not (a <= c and a < B and 0.0 <= phi <= math.pi / 2.0):
        raise ValueError("requires a <= c, a < B, and 0 <= phi <= pi/2")
    r, u = a / c, a / B
    width_angle = math.pi * c / W * math.sqrt(u)
    if width_angle >= math.pi / 2.0:
        raise ValueError("finite-width correction is outside its real branch")
    Q = 1.0 + 1.464 * r**1.65
    M1 = 1.13 - 0.09 * r
    M2 = -0.54 + 0.89 / (0.2 + r)
    M3 = 0.5 - 1.0 / (0.65 + r) + 14.0 * (1.0 - r) ** 24
    s, co = math.sin(phi), math.cos(phi)
    g = 1.0 + (0.1 + 0.35 * u * u) * (1.0 - s) ** 2
    f_phi = (r * r * co * co + s * s) ** 0.25
    f_w = math.sqrt(1.0 / math.cos(width_angle))
    F = (M1 + M2 * u * u + M3 * u**4) * g * f_phi * f_w
    H1 = 1.0 - 0.34 * u - 0.11 * r * u
    G1 = -1.22 - 0.12 * r
    G2 = 0.55 - 1.05 * r**0.75 + 0.47 * r**1.5
    H2 = 1.0 + G1 * u + G2 * u * u
    p = 0.2 + r + 0.6 * u
    H = H1 + (H2 - H1) * s**p
    return float((dsm + H * dsb) * math.sqrt(math.pi * a / Q) * F)


def surface_growth_rates_si(a, c, B, W, dsm, dsb, stages, dk0):
    pairs_si = tuple((to_si(A, m), m) for A, m in _stages(stages))
    threshold_si = _finite(dk0, "dk0") * 10.0**4.5
    if threshold_si < 0.0:
        raise ValueError("threshold must be nonnegative")

    def growth(phi):
        dk = surface_flaw_dk(a, c, B, W, dsm, dsb, phi)
        if dk < threshold_si:
            return 0.0
        return float(min(A * dk**m for A, m in pairs_si))

    return growth(math.pi / 2.0), growth(0.0)


def life_fixed_step(a0, c0, B, W, dsm, dsb, stages, dk0, n_steps, a_target):
    # Euler needs delta_N. Knowing a first-crossing step count imposes
    # inequalities, not a unique delta_N. Requiring exact final depth would
    # add an unstated shooting condition. Therefore neither is assumed.
    raise ValueError(LIFE_AMBIGUITY)


def oracle_matches(function_name, arguments, expected, rtol=2e-10, atol=1e-12):
    function = globals()[function_name]
    try:
        result = function(*arguments)
    except Exception as exc:
        return expected.get("error_type") == type(exc).__name__ and expected.get("message") == str(exc)
    if "error_type" in expected:
        return False
    if expected.get("transform") == "log10":
        if result <= 0.0:
            return False
        result = math.log10(result)
    return math.isclose(float(result), float(expected["value"]), rel_tol=rtol, abs_tol=atol)
