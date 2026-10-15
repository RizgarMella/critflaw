"""Assembly of the stress intensity factors and reference stress of a flaw in a cylinder for the assessment
engine (theory manual, sections 2.4 and 2.5.2). Returns per-position magnification factors, the bulging and finite-width
factors, validity, the suspect status and the corner envelope under the tabulated route, and the equations used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from critflaw.model import Cylinder, CylinderSurfaceFlaw, CylinderThroughFlaw, ExtendedFlaw, StressState
from critflaw.solutions import cylinder as C
from critflaw.solutions import cylinder_sref as S
from critflaw.solutions import cylinder_tables as T
from critflaw.solutions import plate
from critflaw.solutions.plate import BoolArray, FloatArray

CylFlaw = CylinderSurfaceFlaw | CylinderThroughFlaw | ExtendedFlaw


@dataclass
class CylinderFactors:
    positions: tuple[str, ...]
    mm: list[FloatArray]  # per position
    mb: list[FloatArray]
    bulging: FloatArray
    fw: FloatArray
    crack_size: FloatArray  # the a of K = Y sigma sqrt(pi a)
    front_length: FloatArray  # crack-front length for the Master Curve (capped at 2B by the caller)
    pm_total: FloatArray  # declared membrane stress plus the pressure-derived one
    membrane_multiplier: FloatArray  # beta of a circumferential through-wall flaw, else 1
    valid: BoolArray
    sref: FloatArray
    suspect: BoolArray
    envelope: tuple[FloatArray, FloatArray] | None  # corner envelope of the governing factors, tabulated route
    equations: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)


def _f(value: Any) -> FloatArray:
    return np.asarray(value, dtype=float)


def pressure_membrane(loads: StressState, orientation: str) -> FloatArray:
    """The membrane stress due to pressure for the flaw's orientation (0 without a PressureLoad)."""
    if loads.pressure is None:
        return np.asarray(0.0)
    p = _f(loads.pressure.pressure)
    factor = _f(loads.pressure.hoop_per_pressure if orientation == "axial" else loads.pressure.axial_per_pressure)
    return np.asarray(p * factor, dtype=float)


def cylinder_factors(
    component: Cylinder, flaw: CylFlaw, loads: StressState, pm: FloatArray, pb: FloatArray
) -> CylinderFactors:
    r_i, b, w = _f(component.r_i), _f(component.thickness), _f(component.length)
    r_m = r_i + 0.5 * b
    r_o = r_i + b
    route = component.k_route
    orientation = flaw.orientation
    pm_p = pressure_membrane(loads, orientation)
    pm_mom = _f(loads.pm_moment) if orientation == "circumferential" else np.asarray(0.0)
    pm_force = pm + pm_p  # the uniform membrane stress (axial loads and pressure)
    pm_tot = pm_force + pm_mom  # for K: the global-moment membrane stress (M.25) is added to P_m once
    equations: list[str] = []
    flags: list[str] = []
    notes: dict[str, str] = {
        "k_route": str(route),
        "sref_route": str(component.sref_route),
        "crack_face_pressure": "excluded by the K solutions",
    }
    shape = np.broadcast(pm_tot, b, pb).shape
    eligible = np.ones(shape, dtype=bool)  # per-realisation eligibility of the chosen routes (never a batch abort)
    if route == "pressure_only":
        mechanical = (pm != 0.0) | (pb != 0.0) | (_f(loads.pm_moment) != 0.0)
        eligible &= np.broadcast_to(~mechanical, shape)
        if bool(np.any(mechanical)):
            flags.append("pressure_only_route_refuses_mechanical_primary_stress")
    if component.sref_route == "thick_pressure":
        # the limit-pressure solutions carry pressure only: any mechanical primary component is ineligible
        mechanical = (pm != 0.0) | (pb != 0.0) | (_f(loads.pm_moment) != 0.0)
        eligible &= np.broadcast_to(~mechanical, shape)
        if bool(np.any(mechanical)):
            flags.append("thick_pressure_reference_stress_refuses_mechanical_primary_stress")
    if component.sref_route == "thick" and orientation == "circumferential":
        # the net-section limit load carries tension and global bending only: local through-wall bending is ineligible
        eligible &= np.broadcast_to(pb == 0.0, shape)
        if bool(np.any(pb != 0.0)):
            flags.append("thick_circumferential_limit_load_refuses_through_wall_bending")
    suspect = np.zeros(shape, dtype=bool)
    envelope: tuple[FloatArray, FloatArray] | None = None
    one = np.ones_like(pm_tot * b)
    beta = one
    rm_b = r_m / b
    b_ri = b / r_i

    # ---- surface flaws
    if isinstance(flaw, CylinderSurfaceFlaw):
        a, c = _f(flaw.a), 0.5 * _f(flaw.two_c)
        positions: tuple[str, ...] = ("deepest", "surface")
        crack = a
        front = 2.0 * c
        if route == "pressure_only":
            valid = plate.surface_flaw_valid(a, c, b, w) & (rm_b >= C.RM_OVER_B_MIN) & (rm_b <= C.RM_OVER_B_MAX)
            mm_d, mb_d = plate.surface_flaw_mm_mb(a, c, b, plate.THETA_DEEPEST)
            mm_s, mb_s = plate.surface_flaw_mm_mb(a, c, b, plate.THETA_SURFACE)
            bulging = C.bulging_surface_axial(a, c, r_m, b) if orientation == "axial" else one
            fw = plate.surface_flaw_fw(a, c, b, w)
            equations += (
                ["k.plate.surface", "k.cylinder.bulging_surface"] if orientation == "axial" else ["k.plate.surface"]
            )
        else:
            if orientation == "axial":
                table = T.TABLES["M.2"] if flaw.position == "internal" else T.TABLES["M.4"]
                width_ok = 2.0 * c / w <= 0.15
            else:
                if flaw.position == "external":
                    # the source directs the plate solution for circumferential external surface flaws
                    table = None
                    width_ok = np.ones_like(one, dtype=bool)
                else:
                    table = T.TABLES["M.7"]
                    width_ok = np.ones_like(one, dtype=bool)
            if table is None:
                # the plate-directed family: its through-wall destination is tabulated, so r_m/B must be in range
                valid = plate.surface_flaw_valid(a, c, b, w) & (rm_b >= C.RM_OVER_B_MIN) & (rm_b <= C.RM_OVER_B_MAX)
                mm_d, mb_d = plate.surface_flaw_mm_mb(a, c, b, plate.THETA_DEEPEST)
                mm_s, mb_s = plate.surface_flaw_mm_mb(a, c, b, plate.THETA_SURFACE)
                bulging, fw = one, plate.surface_flaw_fw(a, c, b, w)  # the plate solution carries its f_w (W = length)
                equations.append("k.plate.surface")
                notes["tabulated_route_uses_plate_solution"] = "circumferential external surface flaw"
            else:
                vals = [C.interpolate(table, q, a / c, a / b, b_ri) for q in range(4)]
                valid = vals[0].valid & width_ok & (a / c > 0.0)
                mm_d, mb_d, mm_s, mb_s = (v.value for v in vals)
                suspect = np.any([v.suspect for v in vals], axis=0)
                envelope = (
                    np.asarray(np.min([v.lower for v in vals], axis=0), dtype=float),
                    np.asarray(np.max([v.upper for v in vals], axis=0), dtype=float),
                )
                bulging, fw = one, one
                equations.append(
                    "k.cylinder.tables.axial_surface"
                    if orientation == "axial"
                    else "k.cylinder.tables.circumferential_surface"
                )
        mm = [np.asarray(mm_d, dtype=float), np.asarray(mm_s, dtype=float)]
        mb = [np.asarray(mb_d, dtype=float), np.asarray(mb_s, dtype=float)]
        # reference stress
        radius = r_i if flaw.position == "internal" else r_o
        if component.sref_route == "thin":
            if orientation == "axial":
                sref = S.axial_surface_thin(pm_force, pb, a, c, radius, b, w)
                equations.append("sref.cylinder.axial_thin")
            else:
                sref = S.circumferential_surface_thin(pm_tot, pb, a, c, radius, b)  # P_m total, the moment once
                valid = valid & S.circumferential_surface_thin_valid(c, radius)
                equations.append("sref.cylinder.circumferential_thin")
        elif component.sref_route == "thick_pressure":
            if orientation != "axial" or loads.pressure is None:
                raise ValueError("sref_route 'thick_pressure' applies to axial surface flaws with a PressureLoad")
            p_l = (S.axial_internal_limit_pressure if flaw.position == "internal" else S.axial_external_limit_pressure)(
                a, c, r_i, b, 1.0
            )
            # Lr = p / p_L; expressed as a reference stress through sref = sigma_Y p / p_L with p_L at sigma_Y = 1
            sref = _f(loads.pressure.pressure) / p_l
            notes["sref_from_limit_load"] = "thick-wall limit pressure: sref = sigma_Y p / p_L (Lr = p / p_L)"
            equations.append("sref.cylinder.axial_pressure")
        else:
            if orientation == "axial":
                if flaw.position != "external":
                    raise ValueError("the thick-wall membrane-and-bending limit is given for external axial flaws only")
                l_n, l_b, ok = S.axial_external_thick_limits(pm_force, pb, a, c, b, w)
                with np.errstate(divide="ignore", invalid="ignore"):
                    sref = np.where(pm_force > 0.0, pm_force / l_n, (2.0 / 3.0) * pb / np.where(l_b > 0.0, l_b, np.nan))
                valid = valid & ok
                notes["sref_from_limit_load"] = (
                    "thick-wall limit stress: sref = Pm / L_N (or (2/3) Pb / L_b), Lr = sref / sigma_Y"
                )
                equations.append("sref.cylinder.axial_thick_bending")
            else:
                # half-angle: the conservative reading of each pair (docs/unsourced_items.md): the flaw's own
                # surface c / r_i for internal flaws, mid-thickness c / r_m for external ones
                theta = c / (r_i if flaw.position == "internal" else r_m)
                sref = _net_section(a, theta, r_m, b, pm_force, pm_mom, flaw.position == "external")
                notes["sref_from_limit_load"] = "net-section limit: sref = sigma_Y F / F_limit"
                equations.append("sref.cylinder.circumferential_thick")
        return CylinderFactors(
            positions,
            mm,
            mb,
            np.asarray(bulging),
            np.asarray(fw),
            crack,
            front,
            pm_tot,
            beta,
            np.asarray(valid & eligible, dtype=bool),
            np.asarray(sref, dtype=float),
            np.asarray(suspect, dtype=bool),
            envelope,
            equations,
            flags,
            notes,
        )

    # ---- through-wall flaws
    if isinstance(flaw, CylinderThroughFlaw):
        a = 0.5 * _f(flaw.two_a)
        positions = ("outside", "inside")
        crack = a
        front = 2.0 * b
        if route == "pressure_only":
            valid = plate.through_flaw_valid(a, w) & (rm_b >= C.RM_OVER_B_MIN) & (rm_b <= C.RM_OVER_B_MAX)
            bulging = C.bulging_through_axial(a, r_m, b) if orientation == "axial" else one
            fw = plate.through_flaw_fw(a, w)
            mm = [one, one]
            mb = [one, one]
            equations += ["k.plate.through"] + (["k.cylinder.bulging_through"] if orientation == "axial" else [])
            lam_max = C.LAMBDA_MAX_AXIAL if orientation == "axial" else C.LAMBDA_MAX_CIRCUMFERENTIAL
            lam = C.shell_lambda(a, r_m, b)
            if bool(np.any(lam > lam_max)):
                notes["largest_lambda_beyond_check"] = (
                    f"{float(np.max(lam)):.3f} (the bulging factor's only independent check ends at {lam_max})"
                )
        else:
            lam = C.shell_lambda(a, r_m, b)
            names = ("M.1a", "M.1b", "M.1c", "M.1d") if orientation == "axial" else ("M.6a", "M.6b", "M.6c", "M.6d")
            vals = [C.interpolate_columns(T.TABLES[n], 0, lam, rm_b) for n in names]
            m1, m2, m3, m4 = (v.value for v in vals)
            mm = [m1 + m2, m1 - m2]  # outside, inside
            if loads.bending_tensile_surface == "outside":
                mb = [m3 + m4, m3 - m4]
            else:
                # bending tensile inside: the bending stress field is the negative of the source's convention, so by
                # linearity of the solution the bending term changes sign at both surfaces (derived, not a coefficient)
                mb = [-(m3 + m4), -(m3 - m4)]
                notes["bending_tensile_inside"] = "bending term negated at both surfaces by linear superposition"
            valid = np.all([v.valid for v in vals], axis=0)
            suspect = np.any([v.suspect for v in vals], axis=0)
            # sign-aware corner envelope of M_m = M_1 +/- M_2 at both surfaces (the bending factor likewise)
            lo_out = vals[0].lower + vals[1].lower
            hi_out = vals[0].upper + vals[1].upper
            lo_in = vals[0].lower - vals[1].upper
            hi_in = vals[0].upper - vals[1].lower
            lo_b_out = vals[2].lower + vals[3].lower
            hi_b_out = vals[2].upper + vals[3].upper
            lo_b_in = vals[2].lower - vals[3].upper
            hi_b_in = vals[2].upper - vals[3].lower
            envelope = (
                np.asarray(np.minimum(np.minimum(lo_out, lo_in), np.minimum(lo_b_out, lo_b_in)), dtype=float),
                np.asarray(np.maximum(np.maximum(hi_out, hi_in), np.maximum(hi_b_out, hi_b_in)), dtype=float),
            )
            notes["envelope"] = "sign-aware corner envelopes of M_1 +/- M_2 and M_3 +/- M_4 over both surfaces"
            bulging, fw = one, one
            if orientation == "circumferential":
                beta = C.shell_beta(a, r_m)
            equations.append("k.cylinder.tables.through")
        if orientation == "axial":
            sref = S.axial_through_thin(pm_force, pb, a, r_i, b, w)
            equations.append("sref.cylinder.axial_thin")
        else:
            if component.sref_route == "thick":
                sref = _net_section(b, a / r_m, r_m, b, pm_force, pm_mom, False)
                notes["sref_from_limit_load"] = "net-section limit: sref = sigma_Y F / F_limit"
                equations.append("sref.cylinder.circumferential_thick")
            else:
                sref = S.circumferential_through_thin(pm, pm_p, _f(loads.pm_moment), pb, a, r_i, b)
                valid = valid & S.circumferential_through_thin_valid(a, r_i, b)
                equations.append("sref.cylinder.circumferential_thin")
        return CylinderFactors(
            positions,
            mm,
            mb,
            np.asarray(bulging),
            np.asarray(fw),
            crack,
            front,
            pm_tot,
            beta,
            np.asarray(valid & eligible, dtype=bool),
            np.asarray(sref, dtype=float),
            np.asarray(suspect, dtype=bool),
            envelope,
            equations,
            flags,
            notes,
        )

    # ---- extended flaws
    a = _f(flaw.a)
    positions = ("deepest",)
    crack = a
    front = 2.0 * b
    external = flaw.position == "external"
    if route == "pressure_only" or (orientation == "circumferential" and external):
        if orientation == "circumferential" and external:
            mm_v = C.fully_circumferential_external_mm(a, r_i, r_o)
            mm, mb = [mm_v], [mm_v]
            valid = (a > 0.0) & (a < b)
            equations.append("k.cylinder.extended_external")
            notes["bending_added_to_membrane"] = "the source's note: Pb, Qb are added to Pm, Qm for this solution"
        else:
            mm_v, mb_v = C.extended_plate_mm_mb(a, b)
            mm, mb = [mm_v], [mb_v]
            valid = C.extended_plate_valid(a, b)
            equations.append("k.plate.extended")
        bulging, fw = one, one
        if route == "pressure_only":
            valid = valid & (rm_b >= C.RM_OVER_B_MIN) & (rm_b <= C.RM_OVER_B_MAX)
            notes["bulging_extended"] = "M = 1 for extended flaws under the pressure-only route (choice, listed)"
    else:
        name = {("axial", False): "M.3", ("axial", True): "M.5", ("circumferential", False): "M.8"}[
            (orientation, external)
        ]
        table = T.TABLES[name]
        if name == "M.5":
            vals = [C.interpolate(table, q, b_ri, a / b) for q in range(2)]
        else:
            vals = [C.interpolate(table, q, a / b, b_ri) for q in range(2)]
        mm, mb = [vals[0].value], [vals[1].value]
        valid = vals[0].valid
        suspect = vals[0].suspect | vals[1].suspect
        envelope = (
            np.asarray(np.minimum(vals[0].lower, vals[1].lower), dtype=float),
            np.asarray(np.maximum(vals[0].upper, vals[1].upper), dtype=float),
        )
        bulging, fw = one, one
        equations.append(
            {
                "M.3": "k.cylinder.tables.axial_surface",
                "M.5": "k.cylinder.tables.axial_extended_external",
                "M.8": "k.cylinder.tables.circumferential_extended",
            }[name]
        )
    radius = r_o if external else r_i
    if component.sref_route == "thick_pressure":
        if orientation != "axial" or loads.pressure is None:
            raise ValueError("sref_route 'thick_pressure' applies to axial flaws with a PressureLoad")
        p_l = (S.axial_external_limit_pressure if external else S.axial_internal_limit_pressure)(
            a, a, r_i, b, 1.0, extended=True
        )
        sref = _f(loads.pressure.pressure) / p_l
        notes["sref_from_limit_load"] = "thick-wall limit pressure: sref = sigma_Y p / p_L (Lr = p / p_L)"
        equations.append("sref.cylinder.axial_pressure")
    elif orientation == "axial":
        sref = S.axial_extended_thin(pm_force, pb, a, radius, b, w, external)
        equations.append("sref.cylinder.axial_thin")
    elif component.sref_route == "thick":
        sref = _net_section(a, np.pi, r_m, b, pm_force, pm_mom, external)
        notes["sref_from_limit_load"] = "net-section limit: sref = sigma_Y F / F_limit"
        equations.append("sref.cylinder.circumferential_thick")
    else:
        sref = S.circumferential_fully_thin(pm_tot, pb, a, radius, b)  # P_m total, the moment once
        equations.append("sref.cylinder.circumferential_thin")
    return CylinderFactors(
        positions,
        mm,
        mb,
        np.asarray(bulging),
        np.asarray(fw),
        crack,
        front,
        pm_tot,
        beta,
        np.asarray(valid & eligible, dtype=bool),
        np.asarray(sref, dtype=float),
        np.asarray(suspect, dtype=bool),
        envelope,
        equations,
        flags,
        notes,
    )


def _net_section(
    a: FloatArray, theta: Any, r_m: FloatArray, b: FloatArray, pm: FloatArray, pm_moment: FloatArray, external: bool
) -> FloatArray:
    """sref = sigma_Y x (applied / limit) for the thick-wall circumferential net-section solution: the applied
    force F = 2 pi r_m B Pm and moment M = pi r_m^2 B Pm,b (the stress Pm,b at the extreme fibre of a thin ring),
    load ratio lam = M / (r_m F); Lr = F / F_limit = M / M_limit at that ratio."""
    f_app = 2.0 * np.pi * r_m * b * pm
    m_app = np.pi * r_m**2 * b * pm_moment
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(f_app > 0.0, m_app / (r_m * f_app), np.inf)
    l_n, l_b, _, _ = S.circumferential_thick_limit(a, theta, r_m, b, lam, external=external)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(f_app > 0.0, f_app / (2.0 * np.pi * r_m * b * l_n), m_app / (4.0 * r_m**2 * b * l_b))
    return np.asarray(lr, dtype=float)  # a stress: Pm / L_N (sigma_Y cancels)
