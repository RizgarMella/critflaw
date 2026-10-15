PI = 3.141592653589793


def residual_qm(sy_app, su_app, sref, fraction=1.0, relaxation=False, sy_room=None):
    """Return residual membrane stress; use no external dependencies."""
    sy_app = float(sy_app)
    su_app = float(su_app)
    sref = float(sref)
    fraction = float(fraction)
    sy_u = sy_app if sy_room is None else float(sy_room)
    unrelaxed = fraction * sy_u
    if not relaxation:
        return float(unrelaxed)
    if fraction != 1.0:
        raise ValueError('relaxation requires fraction == 1.0')
    sf = (sy_app + su_app) / 2.0
    ratio = min(max(sref / sf, 0.4), 1.0)
    return float(min(unrelaxed, (1.4 - ratio) * sy_app))


def kr_rule(kp, ks, lr, kmat, a, sy, method='rho', beta=1.0):
    """Implement the supplied decision table in its stated order."""
    kp = float(kp)
    ks = float(ks)
    lr = float(lr)
    kmat = float(kmat)
    a = float(a)
    sy = float(sy)
    beta = float(beta)
    if kp < 0.0:
        kp = 0.0
    if ks < 0.0:
        ks = 0.0
    if ks == 0.0:
        return float(kp / kmat)
    if kp == 0.0 and lr == 0.0:
        a_eff = a + (ks / sy) ** 2 / (2.0 * PI * beta)
        kjs = (a_eff / a) ** 0.5 * ks
        return float(kjs / kmat)
    if kp == 0.0 and lr > 0.0:
        rho1 = 0.25
        v = 3.1 - 2.0 * lr if lr < 1.05 else 1.0
    else:
        chi = ks * lr / kp
        if chi < 5.2:
            rho1 = 0.1 * chi ** 0.714 - 0.007 * chi ** 2 + 0.00003 * chi ** 5
        else:
            rho1 = 0.25
        v = min(1.0 + 0.2 * lr + 0.02 * chi * (1.0 + 2.0 * lr), 3.1 - 2.0 * lr) if lr < 1.05 else 1.0
    if lr <= 0.8:
        rho = rho1
    elif lr < 1.05:
        rho = 4.0 * rho1 * (1.05 - lr)
    else:
        rho = 0.0
    if method == 'rho':
        return float((kp + ks) / kmat + rho)
    if method == 'V':
        return float((kp + v * ks) / kmat)
    raise ValueError("method must be 'rho' or 'V'")

