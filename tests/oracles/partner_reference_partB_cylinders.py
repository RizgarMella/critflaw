import numpy as np

# Independent transcription of the supplied equations; no other implementation
# or primary publication was available. All public functions return dictionaries.
# Case inputs use {"function": name, "kwargs": arguments}.

_PI = float(np.pi)

def _finite(name, x):
    try:
        x = float(x)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + ' must be a finite real scalar')
    if not np.isfinite(x):
        raise ValueError(name + ' must be finite')
    return x

def _positive(name, x):
    x = _finite(name, x)
    if x <= 0:
        raise ValueError(name + ' must be positive')
    return x

def _nonnegative(name, x):
    x = _finite(name, x)
    if x < 0:
        raise ValueError(name + ' must be nonnegative')
    return x

def _unit(name, x, closed=True):
    x = _nonnegative(name, x)
    if x > 1 or (not closed and x == 1):
        raise ValueError(name + ' outside permitted unit interval')
    return x

def _sqrt(x):
    if x < 0:
        raise ValueError('negative square-root argument')
    return float(np.sqrt(x))

def _bisect(f, lo, hi):
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError('physical root is not bracketed')
    for unused in range(180):
        mid = lo + (hi - lo) / 2
        if mid == lo or mid == hi:
            return mid
        fm = f(mid)
        if fm == 0:
            return mid
        if (flo < 0) == (fm < 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return lo + (hi - lo) / 2

def axial_thin(kind, a, B, radius, W, P_m=0.0, P_b=0.0, c=None):
    '''kind: through, surface, extended_internal, extended_external.
    For surface flaws radius is the selected inner or outer radius.'''
    a = _nonnegative('a', a)
    B = _positive('B', B)
    radius = _positive('radius', radius)
    W = _positive('W', W)
    P_m = _nonnegative('P_m', P_m)
    P_b = _nonnegative('P_b', P_b)
    if kind == 'through':
        if 2*a >= W:
            raise ValueError('through flaw requires 2*a < W')
        mt = _sqrt(1 + 1.6*a*a/(radius*B))
        value = 1.2*mt*P_m + 2*P_b/(3*(1-2*a/W))
    elif kind in ('surface', 'extended_internal', 'extended_external'):
        alpha = _unit('a/B', a/B, closed=False)
        if kind == 'surface':
            c = _positive('c', c)
            if 2*c > W:
                raise ValueError('surface flaw requires 2*c <= W')
            app = alpha/(1+B/c) if W >= 2*(c+B) else alpha*(2*c/W)
            mt = _sqrt(1 + 1.6*c*c/(radius*B))
            ms = (1-alpha/mt)/(1-alpha)
        elif kind == 'extended_internal':
            app = alpha
            mt = _sqrt(1 + 1.6*(W/2)**2/(radius*B))
            ms = (1-alpha/mt)/(1-alpha)
        else:
            app = alpha
            ms = 1/(1-alpha)
        if app >= 1:
            raise ValueError('undefined bending denominator')
        value = 1.2*ms*P_m + 2*P_b/(3*(1-app)**2)
    else:
        raise ValueError('unknown axial-thin kind')
    return {'sigma_ref': float(value)}

def axial_pressure(side, a, B, r_i, sigma_Y=1.0, c=None, extended=False):
    '''Thick-wall axial flaw; returns dimensional and normalised pressure.
    Extended internal limit: h1c=0 and p/sigma_Y =
    (1+alpha*eta)/(1+alpha*eta/2)*log((1+eta)/(1+alpha*eta)).
    Extended external limit: h2c=0 and p/sigma_Y=log(1+(1-alpha)*eta).
    At alpha=0 the continuous unflawed limit is log(1+eta).'''
    if side not in ('internal', 'external'):
        raise ValueError('unknown side')
    a = _nonnegative('a', a)
    B = _positive('B', B)
    r_i = _positive('r_i', r_i)
    sigma_Y = _positive('sigma_Y', sigma_Y)
    alpha = _unit('a/B', a/B)
    eta = B/r_i
    if not extended:
        c = _positive('c', c)
    if alpha == 0:
        value = float(np.log1p(eta))
    elif alpha == 1:
        value = 0.0
    elif side == 'external':
        residual = float(np.log1p((1-alpha)*eta))
        if extended:
            value = residual
        else:
            phi = a/c
            t = 1.4*alpha*eta/((1+eta)*phi*phi)
            m = _sqrt(1+t)
            # Rationalisation avoids cancellation in M_ex - 1.
            h = (1-alpha)*(m+1)/t
            value = (h*float(np.log1p(eta)) + residual)/(h+1)
    else:
        A = float(np.log1p(eta))
        C = float(np.log1p(alpha*eta))
        D = float(np.log1p((1-alpha)*eta/(1+alpha*eta)))
        if extended:
            fps = 1/(1+alpha*eta/2)
            value = fps*(1+alpha*eta)*D
        else:
            phi = a/c
            fps = 1/(1+alpha*eta/2) if phi <= 1 else phi/(phi+alpha*eta/2)
            q = 1/_sqrt(1+1.4*alpha*eta/(phi*phi*(1+alpha*eta)))
            # Original h1c divided in numerator and denominator by M_in.
            denominator = A-fps*(q*C+(1+alpha*eta*(1-q))*D)
            if denominator == 0:
                raise ValueError('undefined h1c denominator')
            h = (1-alpha)*fps*q*(C-alpha*eta*D)/denominator
            if h == -1:
                raise ValueError('undefined pressure denominator')
            value = (h*A+fps*(1+alpha*eta)*D)/(h+1)
    return {'p_over_sigma_Y': float(value), 'p_L': float(sigma_Y*value)}

def axial_external_limits(alpha, psi, lam, beta=0.0):
    '''lam=None denotes pure bending (lambda -> infinity).
    beta here is 2*c/W, not the neutral-axis angle used below.
    psi=beta=0 selects the explicitly supplied extended-flaw reduction.'''
    alpha = _unit('alpha', alpha)
    psi = _nonnegative('psi', psi)
    beta = _nonnegative('beta', beta)
    if beta > psi/(1+psi):
        raise ValueError('beta exceeds psi/(1+psi)')
    x = alpha*psi/(1+psi)
    d1 = (1-x)**2 + 2*alpha*alpha*psi/(1+psi)**2
    d2 = (1-x)*(1-psi*(1-alpha)) + 2*alpha*(1-alpha)*psi/(1+psi)
    if lam is None:
        alpha0 = (psi+1)/(psi+2)
        branch = 'i' if alpha <= alpha0 else 'ii'
        return {'L_N': 0.0, 'L_b': float(d1 if branch == 'i' else d2), 'branch': branch}
    lam = _nonnegative('lam', lam)
    if lam == 0:
        alpha0 = 1.0
    else:
        z = lam-0.5
        t = 2*(1+psi)*lam/(2+psi)
        root = _sqrt(z*z+t)
        alpha0 = t/(root+z) if z >= 0 else -z+root
    branch = 'i' if alpha <= alpha0 else 'ii'
    if branch == 'i':
        d = d1
        t = 2*lam+x
        denominator = t+_sqrt(t*t+d)
    else:
        d = d2
        t = 2*lam+psi*(1-alpha)
        denominator = t+_sqrt(t*t+(1+psi)*d)
    if denominator == 0:
        raise ValueError('undefined limit-stress denominator')
    ln = d/denominator
    return {'L_N': float(ln), 'L_b': float(4*lam*ln), 'branch': branch}

def circumferential_thin(kind, a, B, radius, P_m=0.0, P_b=0.0, c=None,
                         P_ma=0.0, P_mp=0.0, P_mb=0.0, P_bl=0.0):
    '''For through flaws radius is r_i. For surface/fully flaws it is
    r_i or r_o, selected by the caller. kind: through, surface, fully.'''
    a = _nonnegative('a', a)
    B = _positive('B', B)
    radius = _positive('radius', radius)
    if kind == 'through':
        if B/radius > 0.2:
            raise ValueError('through circumferential formula requires B/r_i <= 0.2')
        x = a/radius
        if 2*x/_PI > 0.5:
            raise ValueError('through circumferential flaw exceeds angular range')
        P_ma = _nonnegative('P_ma', P_ma)
        P_mp = _nonnegative('P_mp', P_mp)
        P_mb = _nonnegative('P_mb', P_mb)
        P_bl = _nonnegative('P_bl', P_bl)
        ro, rm = radius+B, radius+B/2
        dt = _PI-x-2*float(np.arcsin(0.5*np.sin(x)))
        db = _PI-x-2*float(np.sin(x))**2/(_PI-x)-float(np.sin(2*x))/2
        # (ro^4-ri^4)/B written without subtraction of fourth powers.
        section = (ro+radius)*(ro*ro+radius*radius)/(4*ro*rm*rm)
        value = _PI*(P_ma+P_mp)/dt + _PI*P_mb*section/db + 2*P_bl/(3*(1-2*x/_PI))
    elif kind in ('surface', 'fully'):
        alpha = _unit('a/B', a/B, closed=False)
        P_m = _nonnegative('P_m', P_m)
        P_b = _nonnegative('P_b', P_b)
        if kind == 'fully':
            x = _PI
            app = alpha
            sx = 0.0
        else:
            c = _positive('c', c)
            if c/(_PI*radius) > 0.8:
                raise ValueError('surface circumferential flaw exceeds angular range')
            x = c/radius
            app = alpha/(1+B/c) if _PI*radius >= c+B else alpha*c/(_PI*radius)
            sx = float(np.sin(x))
        if app >= 1:
            raise ValueError('undefined bending denominator')
        value = P_m*(_PI*(1-alpha)+2*alpha*sx)/((1-alpha)*(_PI-alpha*x)) + 2*P_b/(3*(1-app)**2)
    else:
        raise ValueError('unknown circumferential-thin kind')
    return {'sigma_ref': float(value)}

def circumferential_net(alpha, eta, theta, lam=0.0, side='internal',
                        r_m=None, sigma_Y=None):
    '''Dimensionless thick-wall net section. lam=None means pure bending.
    eta=0 is accepted as the requested thin-wall mathematical limit.
    If r_m and sigma_Y are supplied, B=eta*r_m and dimensional F_limit
    and M_limit are also returned. Only tensile/nonnegative load ratios
    and the physical neutral-axis root 0 <= beta <= pi/2 are selected.'''
    alpha = _unit('alpha', alpha)
    eta = _nonnegative('eta', eta)
    if eta >= 2:
        raise ValueError('positive inner radius requires eta < 2')
    theta = _nonnegative('theta', theta)
    if theta > _PI:
        raise ValueError('theta must not exceed pi')
    if side not in ('internal', 'external'):
        raise ValueError('unknown side')
    if lam is not None:
        lam = _nonnegative('lam', lam)
    if alpha == 1 and theta == _PI:
        raise ValueError('no remaining section; neutral-axis angle is undefined')
    fb = 1+eta*eta/12
    if side == 'internal':
        fa = 1-eta/2+alpha*eta/2
        fc = 1-eta+eta*eta/4+alpha*eta-alpha*eta*eta/2+alpha*alpha*eta*eta/3
        fd = (1-alpha)*(1+alpha*eta-alpha*eta*eta/6+alpha*alpha*eta*eta/3+eta*eta/12)/fb
        fe = 1-alpha+alpha*eta/2-alpha*alpha*eta/2
    else:
        fa = 1+eta/2-alpha*eta/2
        fc = 1+eta+eta*eta/4-alpha*eta-alpha*eta*eta/2+alpha*alpha*eta*eta/3
        fd = (1-alpha)*(1-alpha*eta-alpha*eta*eta/6+alpha*alpha*eta*eta/3+eta*eta/12)/fb
        fe = 1-alpha-alpha*eta/2+alpha*alpha*eta/2
    st = 0.0 if theta == _PI else float(np.sin(theta))
    def values(beta):
        if theta+beta <= _PI:
            ln = 1-fa*alpha*theta/_PI-2*beta/_PI
            lb = fb*float(np.sin(beta))-alpha*fc*st/2
            return ln, lb, 'i'
        if fe == 0:
            raise ValueError('branch ii undefined for zero residual section')
        ln = 2*fe*(1-beta/_PI)-1+(1-fe)*theta/_PI
        lb = fb*(fd*float(np.sin(beta))+(1-fd)*st/2)
        return ln, lb, 'ii'
    upper = min(_PI/2, _PI-theta) if fe == 0 else _PI/2
    if lam is None:
        beta = _bisect(lambda b: values(b)[0], 0.0, upper)
    else:
        beta = _bisect(lambda b: values(b)[1]-lam*(_PI/2)*values(b)[0], 0.0, upper)
    # Snap only to the exact branch boundary within rounding error.
    boundary = _PI-theta
    if abs(beta-boundary) <= 16*np.finfo(float).eps*_PI:
        beta = boundary
    ln, lb, branch = values(beta)
    if lam is None:
        ln = 0.0
    elif lam == 0:
        lb = 0.0
    if ln < -1e-12 or lb < -1e-12:
        raise ValueError('no nonnegative physical limit-load solution')
    result = {'L_N': float(max(0.0, ln)), 'L_b': float(max(0.0, lb)), 'beta': float(beta), 'branch': branch}
    if (r_m is None) != (sigma_Y is None):
        raise ValueError('supply both r_m and sigma_Y for dimensional loads')
    if r_m is not None:
        r_m = _positive('r_m', r_m)
        sigma_Y = _positive('sigma_Y', sigma_Y)
        B = eta*r_m
        result['F_limit'] = float(2*_PI*r_m*B*sigma_Y*result['L_N'])
        result['M_limit'] = float(4*r_m*r_m*B*sigma_Y*result['L_b'])
    return result
