import numpy as np

# Scalar reference implementation of the supplied equations.
# All returned numerical results are Python floats. Only numpy is imported.
# The supplied running-maximum rule assumes n >= 1 and nonnegative
# offset and Luders strain; unsupported running-maximum inputs are rejected.

def option2_unmodified(lr, eps_p_over_s, kappa):
    s = float(lr)
    r = 1.0 + float(kappa) * float(eps_p_over_s)
    return float(np.power(r + s*s/(2.0*r), -0.5))

def power_law_q(s, offset, n, luders=None):
    s = float(s)
    q = float(offset) * float(np.power(s, float(n)-1.0))
    if luders is not None and s >= 1.0:
        q += float(luders)/s
    return float(q)

def power_law_q_hat(s, offset, n, luders=None):
    if float(s) < 0.0 or float(n) < 1.0 or float(offset) < 0.0:
        raise ValueError('q_hat requires s >= 0, n >= 1, offset >= 0')
    if luders is not None and float(luders) < 0.0:
        raise ValueError('q_hat requires nonnegative Luders strain')
    q = power_law_q(s, offset, n, luders)
    if luders is not None and float(s) >= 1.0:
        q = max(q, float(offset) + float(luders))
    return float(q)

def option2_power_law(lr, sy, su, E, offset, n, luders=None):
    cutoff = (float(sy) + float(su))/(2.0*float(sy))
    if float(lr) >= cutoff:
        return 0.0
    q = power_law_q_hat(lr, offset, n, luders)
    return option2_unmodified(lr, q, float(E)/float(sy))

def k_from_j(j, E, nu=0.3):
    return float(np.sqrt(float(E)*float(j)/(1.0-float(nu)**2)))

def ctod_m(sy, su):
    ratio = float(sy)/float(su)
    if 0.3 < ratio < 0.98:
        return float(1.517*np.power(ratio, -0.3188))
    return 1.5

def k_from_ctod(ctod, sy, su, E, nu=0.3):
    return float(np.sqrt(ctod_m(sy, su)*float(sy)*float(ctod)*float(E)/(1.0-float(nu)**2)))

def subsize(k_b_mpa, b, B):
    return float(20.0+(float(k_b_mpa)-20.0)*np.power(float(b)/float(B), 0.25))

def k_jc_limit(b0_m, sy_mpa, E_mpa, nu=0.3):
    return float(np.sqrt(float(E_mpa)*float(b0_m)*float(sy_mpa)/((1.0-float(nu)**2)*30.0)))

def master_curve(p, T, T0, l_mm, B_mm):
    l = min(float(l_mm), 2.0*float(B_mm))
    a = 11.0+77.0*np.exp(0.019*(float(T)-float(T0)))
    return float(20.0+a*np.power(25.0/l, 0.25)*np.power(-np.log1p(-float(p)), 0.25))

def charpy_master_curve(T, T0, B_mm, pf=0.05, TK=25.0):
    a = 11.0+77.0*np.exp(0.019*(float(T)-float(T0)-float(TK)))
    return float(20.0+a*np.power(25.0/float(B_mm), 0.25)*np.power(-np.log1p(-float(pf)), 0.25))

def charpy_lower_bound(cv, B_mm):
    return float((12.0*np.sqrt(float(cv))-20.0)*np.power(25.0/float(B_mm), 0.25)+20.0)

def charpy_upper_limit(cv):
    return float(0.54*float(cv)+55.0)

def charpy_upper_shelf(cv_us, E_mpa, nu=0.3):
    cv = float(cv_us)
    j = 0.53*np.power(cv, 1.28)*np.power(0.2, 0.133*np.power(cv, 0.256))
    return float(np.sqrt(float(E_mpa)*j/(1000.0*(1.0-float(nu)**2))))

def subsize_charpy_shift(b_ss_mm):
    return float(-51.4*np.log(2.0*np.power(float(b_ss_mm)/10.0, 0.25)-1.0))

def lognormal_params(mean, sd):
    variance_ln = np.log1p((float(sd)/float(mean))**2)
    return (float(np.log(float(mean))-variance_ln/2.0), float(np.sqrt(variance_ln)))

# Lanczos approximation to Gamma, with reflection. The supplied Weibull
# equations require positive shape; arguments to Gamma are then positive.
_LANCZOS = (0.99999999999980993, 676.5203681218851,
            -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6,
            1.5056327351493116e-7)

def _gamma(z):
    z = float(z)
    if z < 0.5:
        return float(np.pi/(np.sin(np.pi*z)*_gamma(1.0-z)))
    z -= 1.0
    a = _LANCZOS[0]
    for i in range(1, len(_LANCZOS)):
        a += _LANCZOS[i]/(z+i)
    t = z+7.5
    return float(np.sqrt(2.0*np.pi)*np.power(t, z+0.5)*np.exp(-t)*a)

def weibull_mean_sd(theta, k):
    theta = float(theta)
    k = float(k)
    if k <= 0.0:
        raise ValueError('Weibull shape must be positive')
    g1 = _gamma(1.0/k)
    g2 = _gamma(2.0/k)
    mean = theta*g1/k
    variance = theta*theta/k*(2.0*g2-g1*g1/k)
    return (float(mean), float(np.sqrt(variance)))

def clopper_pearson_zero_upper(n, conf=0.99):
    a = (1.0-float(conf))/2.0
    return float(1.0-np.power(a, 1.0/float(n)))

def clopper_pearson_all_lower(n, conf=0.99):
    a = (1.0-float(conf))/2.0
    return float(np.power(a, 1.0/float(n)))
