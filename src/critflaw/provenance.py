"""Equation register and bibliography (theory manual, section 8.1).

Every implemented equation or coefficient set has one record here. Records cite
the primary reference that the source standard itself gives; the clause each
item was transcribed from is kept in the private, git-ignored clause map.

``checked_against_primary`` is "no" until the implemented coefficients have been
compared with the primary publication itself. Until then the citation means
"as cited by the source standard".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

REFERENCES: dict[str, str] = {
    "RookeCartwright1976": (
        "Rooke, D.P. and Cartwright, D.J. Compendium of stress intensity factors. London: The Stationery Office, 1976."
    ),
    "Miller1988": (
        "Miller, A.G. Review of limit loads of structures containing defects. International Journal of Pressure "
        "Vessels and Piping 32 (1988) 197-327."
    ),
    "KiefnerEtAl1973": (
        "Kiefner, J.F., Maxey, W.A., Eiber, R.J. and Duffy, A.R. Failure stress levels of flaws in pressurized "
        "cylinders. Progress in Flaw Growth and Fracture Toughness Testing, ASTM STP 536 (1973) 461-481."
    ),
    "FranceGreenSharples1996": (
        "France, C.C., Green, D. and Sharples, J.K. New stress intensity factor and crack opening area solutions "
        "for through-wall cracks in pipes and cylinders. AEA Technology Report AEAT-0643, 1996; addendum 1999."
    ),
    "RajuNewman1981": (
        "Raju, I.S. and Newman, J.C. Stress intensity factor equations for cracks in three dimensional finite "
        "bodies. NASA TM-83200, 1981."
    ),
    "NewmanRaju1981EFM": (
        "Newman, J.C. and Raju, I.S. An empirical stress intensity factor equation for the surface crack. "
        "Engineering Fracture Mechanics 15 (1981) 185-192."
    ),
    "RajuNewmanVessels": (
        "Raju, I.S. and Newman, J.C. Stress intensity factor influence coefficients for internal and external "
        "surface cracks in cylindrical vessels. In: Aspects of Fracture Mechanics in Pressure Vessels and Piping, "
        "ASME PVP (1982)."
    ),
    "AndersonEtAlWRC471": (
        "Anderson, T.L., Thorwald, G., Revelle, R.J., Osage, D.A., Janelle, J.L. and Fuhry, M.E. Development of "
        "stress intensity factor solutions for surface and embedded cracks in API 579. WRC Bulletin 471, 2002."
    ),
    "DedhiaHarris1983": (
        "Dedhia, D.D. and Harris, D.O. Improved influence functions for part-circumferential cracks in pipes. "
        "In: Circumferential Cracks in Pressure Vessels and Piping, ASME PVP (1983)."
    ),
    "BergmanBrickstad1990": (
        "Bergman, M. and Brickstad, B. Stress-intensity factors for circumferential cracks in pipes by "
        "line-spring calculations. SA/FoU Report 90/05, Swedish Plant Inspectorate, 1990."
    ),
    "JosephErdogan1989": (
        "Joseph, P.F. and Erdogan, F. Surface crack problem in plates. International Journal of Fracture 41 "
        "(1989) 105-131."
    ),
    "AndrasicParker1984": (
        "Andrasic, C.P. and Parker, A.P. Dimensionless stress intensity factors for cracked thick cylinders under "
        "polynomial crack face loadings. Engineering Fracture Mechanics 19 (1984) 187-193."
    ),
    "LeiBudden2004": (
        "Lei, Y. and Budden, P. Limit load solutions for plates with embedded cracks under combined tension and "
        "bending. International Journal of Pressure Vessels and Piping 81 (2004) 589-597."
    ),
    "Folias1984": (
        "Folias, E.S. An axial crack in a pressurized cylindrical shell. International Journal of Fracture 26 "
        "(1984) 251-260."
    ),
    "KastnerEtAl1981": (
        "Kastner, W., Rohrich, E., Schmitt, W. and Steinbuch, R. Critical crack sizes in ductile piping. "
        "International Journal of Pressure Vessels and Piping 9 (1981) 197-219."
    ),
    "Lei2007": (
        "Lei, Y. Limit load solutions for cylinders with axial cracks. British Energy report "
        "E/REP/BBGB/0019/GEN/07, 2007."
    ),
    "Lei2008": (
        "Lei, Y. A review of limit load solutions for cylinders with axial cracks and development of new "
        "solutions. International Journal of Pressure Vessels and Piping 85 (2008) 825-850."
    ),
    "Lei2004": (
        "Lei, Y. J-integral and limit load analysis of semi-elliptical surface cracks in plates under combined "
        "tension and bending. International Journal of Pressure Vessels and Piping 81 (2004) 43-56."
    ),
    "Lei2008EAN": (
        "Lei, Y. Local limit load solutions for plates with surface cracks under combined tension and bending. "
        "British Energy engineering advice note E/EAN/BBGB/0014/GEN/08, 2008."
    ),
    "JonesEshelby1990": (
        "Jones, M.R. and Eshelby, J.M. Limit solutions for circumferentially cracked cylinders under internal "
        "pressure and combined tension and bending. Nuclear Electric report TD/SID/REP/0032, 1990."
    ),
    "ParisErdogan1963": (
        "Paris, P.C. and Erdogan, F. A critical analysis of crack propagation laws. Journal of Basic "
        "Engineering (Trans. ASME) 85 D(4) (1963) 528-534."
    ),
    "King1998": (
        "King, R. A review of fatigue crack growth rates in air and seawater. HSE Report OTH 511. "
        "London: Health & Safety Executive Books, 1998."
    ),
    "R6": (
        "R6 Panel. R6: Assessment of the integrity of structures containing defects. "
        "Revision 4, as amended. Gloucester: EDF Energy, 2001."
    ),
    "NewmanRaju1983": (
        "Newman, J.C. and Raju, I.S. Stress intensity factor equation for cracks in "
        "three-dimensional finite bodies. In: Fracture Mechanics: Proc. of 14th Symposium. "
        "Vol. I: Theory and Analysis. STP 791. Philadelphia: ASTM, 1983: 238-265."
    ),
    "NewmanRajuTM85793": (
        "Newman, J.C. and Raju, I.S. Stress intensity factor equations for cracks in "
        "three-dimensional finite bodies subjected to tension and bending loads. "
        "NASA Technical Memorandum 85793."
    ),
    "WilloughbyDavey1989": (
        "Willoughby, A.A. and Davey, T.G. Plastic collapse at part wall flaws in plates. "
        "In: R.P. Wei, ed. Fracture mechanics: perspectives and directions. Proc. 20th "
        "national symposium. STP 1020. Philadelphia: ASTM, 1989: 390-409."
    ),
    "Wallin1989": (
        "Wallin, K. Simple theoretical Charpy V-K_IC correlation for irradiation embrittlement. In: "
        "Innovative Approaches to Irradiation Damage and Fracture Analysis. New York: ASME, 1989. PVP-170."
    ),
    "WallinRintamaa1997": (
        "Wallin, K. and Rintamaa, R. Statistical reevaluation of the ASME K_IC and K_IR fracture toughness "
        "reference curves. 23rd MPA Seminar, Stuttgart, October 1997."
    ),
    "INSTA1991": (
        "INSTA Technical Report. Assessment of Structures Containing Discontinuities. Stockholm: "
        "Materials Standards Institution, 1991."
    ),
    "LuconEtAl2005": (
        "Lucon, E., Wallin, K., Langenberg, P. and Pisarski, H. The use of Charpy/fracture toughness "
        "correlations in the FITNET procedure. OMAE 2005-67569, Proc. OMAE 2005, Halkidiki, Greece."
    ),
    "AEA2002": (
        "AEA Technology Plc. Partial safety factors for SINTAP procedure. HSE Offshore Technology Report "
        "2000/020. London: The Stationery Office, 2002."
    ),
    "Ainsworth1986": (
        "Ainsworth, R.A. The treatment of thermal and residual stresses in fracture "
        "assessments. Engng Fract Mech, 1986, 24: 65-76."
    ),
    "AinsworthSmithWiesner1999": (
        "Ainsworth, R.A., Smith, S.D. and Wiesner, C.S. Treatment of thermal and residual "
        "stresses in the SINTAP defect assessment procedure. British Energy Generation "
        "Report EDP/GEN/REP/0423/99 and SINTAP/NE/018, 1999."
    ),
}

Checked = Literal["yes", "no", "unobtainable"]


@dataclass(frozen=True)
class EquationRecord:
    id: str
    description: str
    implemented_in: str
    primary_reference: str | None  # key into REFERENCES, or None if the standard cites none
    checked_against_primary: Checked = "no"
    pdf_checked: bool = True  # every record below was diffed against the printed page
    oracles: tuple[str, ...] = ()  # oracle kinds "a".."e" (theory manual index, section 4), declared explicitly
    applicability: str = ""

    @property
    def self_consistent_only(self) -> bool:
        """True if no oracle independent of the source text exists (theory manual, section 8.3): neither published
        values of the primary reference (kind a — the source's own worked values do not count) nor a
        definition-pinning or limiting-case check (kind b)."""
        return not ({"a", "b"} & set(self.oracles))


_RECORDS = [
    EquationRecord(
        id="fad.cutoff",
        description="Plastic collapse cut-off Lr,max from yield and tensile strength",
        implemented_in="critflaw.fad.curves",
        primary_reference=None,
        applicability="mean tensile properties",
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="fad.option1.continuous",
        description="Option 1 assessment line, continuous yielding",
        implemented_in="critflaw.fad.curves.Option1Curve",
        primary_reference=None,
        applicability="materials without a yield discontinuity",
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="fad.option1.discontinuous",
        description="Option 1 assessment line, discontinuous yielding (step at Lr = 1)",
        implemented_in="critflaw.fad.curves.Option1Curve",
        primary_reference=None,
        applicability="materials with a Lueders plateau",
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="material.luders_strain",
        description="Estimate of Lueders strain from yield strength",
        implemented_in="critflaw.fad.curves.luders_strain_estimate",
        primary_reference=None,
        applicability="yield strength below 946 MPa",
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="fad.kr",
        description="Fracture ratio Kr with additive rho or multiplying V",
        implemented_in="critflaw.fad.kr.fracture_ratio",
        primary_reference=None,
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="fad.rho.simplified",
        description="Simplified plasticity interaction term rho",
        implemented_in="critflaw.fad.kr.rho_simplified",
        primary_reference="Ainsworth1986",
        applicability="chi = Ks*Lr/Kp <= 4 (over-conservative above)",
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="fad.v.simplified",
        description="Simplified plasticity interaction factor V",
        implemented_in="critflaw.fad.kr.v_simplified",
        primary_reference="AinsworthSmithWiesner1999",
        applicability="chi = Ks*Lr/Kp <= 4 (over-conservative above)",
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="fad.kjs.plastic_zone",
        description="Plastic-zone-corrected secondary stress intensity factor",
        implemented_in="critflaw.fad.kr.kj_secondary",
        primary_reference="R6",
        applicability="secondary stresses acting alone",
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="residual.as_welded",
        description="As-welded membrane residual stress and its relaxation with primary load",
        implemented_in="critflaw.model.Residual, critflaw.assessment.Assessment.run",
        primary_reference=None,
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="residual.pwht",
        description="Residual stress level after post-weld heat treatment",
        implemented_in="critflaw.model.Residual.pwht",
        primary_reference=None,
        applicability="C-Mn and 2.25Cr1Mo steels, enclosed furnace 550-620 C",
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="k.general",
        description="General form K = (Y sigma) sqrt(pi a) with primary and secondary parts",
        implemented_in="critflaw.solutions.plate",
        primary_reference=None,
        oracles=("c", "d"),  # explicit (was the dataclass default); counted only where kb.evidence links tests
    ),
    EquationRecord(
        id="k.plate.surface",
        oracles=("b", "c", "d"),
        description="Semi-elliptical surface flaw in a plate, membrane and bending",
        implemented_in="critflaw.solutions.plate.surface_flaw_mm_mb",
        primary_reference="NewmanRajuTM85793",
        applicability="0 < a/2c <= 1; a/B < 1.25(a/c + 0.6) for a/2c <= 0.1, else a/B < 1; 2c/W <= 0.8",
    ),
    EquationRecord(
        id="k.plate.through",
        oracles=("b", "c", "d"),
        description="Through-thickness flaw in a plate with finite width correction",
        implemented_in="critflaw.solutions.plate.through_flaw_fw",
        primary_reference=None,
        applicability="2a < W",
    ),
    EquationRecord(
        id="sref.plate.surface",
        oracles=("b", "c", "d"),
        description="Reference stress, surface flaw in a plate, tension and bending",
        implemented_in="critflaw.solutions.plate.surface_flaw_reference_stress",
        primary_reference="WilloughbyDavey1989",
    ),
    EquationRecord(
        id="sref.plate.through",
        oracles=("b", "c", "d"),
        description="Reference stress, through-thickness flaw in a plate, tension and bending",
        implemented_in="critflaw.solutions.plate.through_flaw_reference_stress",
        primary_reference="WilloughbyDavey1989",
    ),
    EquationRecord(
        id="fad.option2",
        description="Option 2 assessment line from the true stress-strain curve",
        implemented_in="critflaw.fad.option2.Option2Curve",
        primary_reference=None,
        oracles=("b", "c"),
        applicability="mean uniaxial true stress-strain curve at the assessment temperature",
    ),
    EquationRecord(
        id="fad.option2.monotone",
        description="Monotone Option 2 line by a non-decreasing secant plastic compliance (this library's rule)",
        implemented_in="critflaw.fad.option2",
        primary_reference=None,
        oracles=("b",),
        applicability="e(s) >= s^2/sqrt(2)",
    ),
    EquationRecord(
        id="material.power_law",
        description="Stress-strain estimate from yield strength, tensile strength and a hardening exponent",
        implemented_in="critflaw.fad.option2.PowerLaw",
        primary_reference=None,
        oracles=("b", "c"),
        applicability="ferritic steels; lower-bound exponent for 0.65 < sy/su < 0.95 and yield 300-1000 MPa",
    ),
    EquationRecord(
        id="toughness.k_from_j",
        description="Fracture toughness from the J-integral",
        implemented_in="critflaw.toughness.k_from_j",
        primary_reference=None,
        oracles=("b", "c"),
    ),
    EquationRecord(
        id="toughness.k_from_ctod",
        description="Fracture toughness from CTOD with the constraint factor m",
        implemented_in="critflaw.toughness.k_from_ctod",
        primary_reference=None,
        oracles=("c",),
        applicability="steels; deeply notched high-constraint specimens; 0.3 < sy/su < 0.98 for the m correlation",
    ),
    EquationRecord(
        id="toughness.subsize",
        description="Thickness correction of cleavage toughness from sub-size specimens",
        implemented_in="critflaw.toughness.subsize_correction",
        primary_reference=None,
        oracles=("b", "c"),
    ),
    EquationRecord(
        id="toughness.master_curve",
        description="Master Curve toughness distribution with crack-front-length adjustment",
        implemented_in="critflaw.toughness.master_curve",
        primary_reference="WallinRintamaa1997",
        oracles=("b", "c"),  # kind a dropped: the source's worked values are no acceptance basis
        applicability="ferritic steels, lower shelf and transition; crack front length capped at twice the thickness",
    ),
    EquationRecord(
        id="toughness.charpy_t0",
        description="Reference temperature T0 from the 27 J or 40 J Charpy transition temperature",
        implemented_in="critflaw.toughness.t0_from_charpy",
        primary_reference="Wallin1989",
        oracles=("c",),  # kind a dropped: the source's worked values are no acceptance basis
    ),
    EquationRecord(
        id="toughness.charpy_master_curve",
        description="Charpy-based Master Curve toughness estimate",
        implemented_in="critflaw.toughness.charpy_master_curve",
        primary_reference="Wallin1989",
        oracles=("c",),  # kind a dropped: the source's worked values are no acceptance basis
        applicability="ferritic steels; not for ductile behaviour",
    ),
    EquationRecord(
        id="toughness.charpy_lower_bound",
        description="Lower-bound toughness from Charpy energy at the service temperature",
        implemented_in="critflaw.toughness.charpy_lower_bound",
        primary_reference="INSTA1991",
        oracles=("c",),
    ),
    EquationRecord(
        id="toughness.charpy_upper_limit",
        description="Upper limit on Charpy-based toughness estimates",
        implemented_in="critflaw.toughness.charpy_upper_limit",
        primary_reference=None,
        oracles=("c",),  # kind a dropped: the source's worked values are no acceptance basis
        applicability="ferritic steels with yield strength below 480 MPa",
    ),
    EquationRecord(
        id="toughness.charpy_upper_shelf",
        description="Initiation toughness from upper-shelf Charpy energy",
        implemented_in="critflaw.toughness.charpy_upper_shelf",
        primary_reference="LuconEtAl2005",
        oracles=("c", "e"),
    ),
    EquationRecord(
        id="toughness.charpy_subsize",
        description="Transition temperature shift for sub-size Charpy specimens",
        implemented_in="critflaw.toughness.subsize_charpy_temperature_shift",
        primary_reference=None,
        oracles=("c",),  # kind a dropped: the source's worked values are no acceptance basis
    ),
    EquationRecord(
        id="toughness.mote",
        description="Minimum-of-three-equivalent toughness",
        implemented_in="critflaw.toughness.minimum_of_three_equivalent",
        primary_reference=None,
        oracles=("c",),
    ),
    EquationRecord(
        id="toughness.lower_bound",
        description="Lower-bound toughness by a one-sided tolerance limit (computed exactly)",
        implemented_in="critflaw.toughness.lower_bound_toughness",
        primary_reference=None,
        oracles=("b",),  # kind a dropped: the source's worked values are no acceptance basis
    ),
    EquationRecord(
        id="partial_factors.table",
        description="Published partial safety factors for fracture assessment (discrete table)",
        implemented_in="critflaw.prob.partial_factors",
        primary_reference="AEA2002",
        oracles=("c",),
        applicability="continuous-yield Option 1; independent variables; stated distribution families",
    ),
    EquationRecord(
        id="k.plate.embedded",
        oracles=("b", "c", "d"),
        description="Embedded elliptical flaw in a plate: membrane magnification at the minor- and major-axis "
        "ends, bending magnification at the minor-axis ends (three coefficient sets), finite-width factor",
        implemented_in="critflaw.solutions.embedded",
        primary_reference="NewmanRajuTM85793",
        applicability="a/B' < 0.625(a/c + 0.6) for a/2c <= 0.1; 0 < a/2c <= 1; 2c/W < 0.5; bending a/2c <= 0.5",
    ),
    EquationRecord(
        id="sref.plate.embedded",
        oracles=("b", "c", "d"),
        description="Reference stress, embedded flaw in a plate, tension and bending (ligament term)",
        implemented_in="critflaw.solutions.embedded.embedded_flaw_reference_stress",
        primary_reference="WilloughbyDavey1989",
    ),
    EquationRecord(
        id="sref.plate.embedded_extended",
        oracles=("c",),
        description="Reference stress, extended embedded flaw in a plate (no K solution exists for it)",
        implemented_in="critflaw.solutions.embedded.extended_embedded_reference_stress",
        primary_reference="LeiBudden2004",
        applicability="a/B <= (B/2 - p - a)/(2B)",
    ),
    EquationRecord(
        id="k.plate.extended",
        oracles=("b", "c"),
        description="Extended (full-width) surface flaw in a plate, membrane and bending polynomials",
        implemented_in="critflaw.solutions.cylinder.extended_plate_mm_mb",
        primary_reference="RookeCartwright1976",
        applicability="a/B <= 0.6",
    ),
    EquationRecord(
        id="k.cylinder.bulging_through",
        oracles=("b", "c"),
        description="Bulging factor of an axial through-wall flaw in a curved shell under internal pressure",
        implemented_in="critflaw.solutions.cylinder.bulging_through_axial",
        primary_reference=None,
        applicability="the limits of the plate solutions; the only independent check is Table M.1a, 5 <= r_m/B <= 100",
    ),
    EquationRecord(
        id="k.cylinder.bulging_surface",
        oracles=("b", "c"),
        description="Bulging factor of an axial surface flaw in a curved shell under internal pressure (with M_T)",
        implemented_in="critflaw.solutions.cylinder.bulging_surface_axial",
        primary_reference="KiefnerEtAl1973",
        applicability="the limits of the plate solutions; singular at a = B (the engine ends the sample at a/B = 0.9)",
    ),
    EquationRecord(
        id="k.cylinder.tables.through",
        oracles=("b", "c", "d"),
        description="Tabulated factors M_1..M_4 for axial and circumferential through-wall flaws in lambda",
        implemented_in="critflaw.solutions.cylinder_tables (M.1, M.6), critflaw.solutions.cylinder.interpolate_columns",
        primary_reference="FranceGreenSharples1996",
        applicability="5 <= r_m/B <= 100; lambda up to each column's last printed value (12.211 / 15.143 at most)",
    ),
    EquationRecord(
        id="k.cylinder.tables.axial_surface",
        oracles=("b", "c", "d"),
        description="Tabulated factors for axial internal and external surface flaws and the extended internal flaw",
        implemented_in="critflaw.solutions.cylinder_tables (M.2, M.3, M.4), critflaw.solutions.cylinder.interpolate",
        primary_reference="RajuNewmanVessels",
        applicability="0 <= a/B <= 0.8; 0.05 <= a/c <= 1; 0.1 <= B/r_i <= 0.25; 2c/W <= 0.15",
    ),
    EquationRecord(
        id="k.cylinder.tables.axial_extended_external",
        oracles=("b", "c", "d"),
        description="Tabulated factors for the extended axial external surface flaw",
        implemented_in="critflaw.solutions.cylinder_tables (M.5)",
        primary_reference="AndersonEtAlWRC471",
        applicability="0 <= a/B <= 0.8; 0.001 <= B/r_i <= 0.333",
    ),
    EquationRecord(
        id="k.cylinder.tables.circumferential_surface",
        oracles=("b", "c", "d"),
        description="Tabulated factors for circumferential internal surface flaws",
        implemented_in="critflaw.solutions.cylinder_tables (M.7)",
        primary_reference="DedhiaHarris1983",
        applicability="0 <= a/B <= 0.8; 0.1 <= a/c <= 1; 0.1 <= B/r_i <= 0.2",
    ),
    EquationRecord(
        id="k.cylinder.tables.circumferential_extended",
        oracles=("b", "c", "d"),
        description="Tabulated factors for fully circumferential internal flaws",
        implemented_in="critflaw.solutions.cylinder_tables (M.8)",
        primary_reference="JosephErdogan1989",
        applicability="0 <= a/B <= 0.8; 0.1 <= B/r_i <= 0.2",
    ),
    EquationRecord(
        id="k.cylinder.extended_external",
        oracles=("b", "c"),
        description="Fully circumferential external surface flaw in a cylinder, membrane loading",
        implemented_in="critflaw.solutions.cylinder.fully_circumferential_external_mm",
        primary_reference="RookeCartwright1976",
    ),
    EquationRecord(
        id="sref.cylinder.axial_thin",
        oracles=("b", "c", "d"),
        description="Thin-wall reference stresses for axial flaws in cylinders (through-wall, surface, extended) "
        "with the empirical 1.2 multiplier",
        implemented_in="critflaw.solutions.cylinder_sref",
        primary_reference="Folias1984",
    ),
    EquationRecord(
        id="sref.cylinder.axial_pressure",
        oracles=("c", "d"),
        description="Thick-wall limit pressures for internal and external axial surface flaws (defect-face "
        "pressure included) and their extended forms",
        implemented_in="critflaw.solutions.cylinder_sref",
        primary_reference="Lei2008",
    ),
    EquationRecord(
        id="sref.cylinder.axial_thick_bending",
        oracles=("c", "d"),
        description="Thick-wall normalised limit force and moment for an external axial surface flaw under "
        "membrane and through-wall bending",
        implemented_in="critflaw.solutions.cylinder_sref.axial_external_thick_limits",
        primary_reference="Lei2004",
        applicability="2c/W <= psi/(1 + psi)",
    ),
    EquationRecord(
        id="sref.cylinder.circumferential_thin",
        oracles=("b", "c", "d"),
        description="Thin-wall reference stresses for circumferential flaws (through-wall under tension, "
        "bending and pressure; surface; fully circumferential)",
        implemented_in="critflaw.solutions.cylinder_sref",
        primary_reference="KastnerEtAl1981",
        applicability="through-wall: 2a/(pi r_i) <= 0.5, B/r_i <= 0.2; surface: c/(pi r_i) <= 0.8",
    ),
    EquationRecord(
        id="sref.cylinder.circumferential_thick",
        oracles=("c", "d"),
        description="Thick-wall net-section limit force and moment for circumferential flaws under tension "
        "and global bending",
        implemented_in="critflaw.solutions.cylinder_sref.circumferential_thick_limit",
        primary_reference="JonesEshelby1990",
    ),
    EquationRecord(
        id="fatigue.paris_law",
        description="Crack growth rate da/dN = A (dK)^m above a threshold dK0, zero below; two-stage form as the "
        "minimum of two such lines; dK = Y (dsigma) sqrt(pi a)",
        implemented_in="critflaw.fatigue.laws.GrowthLaw",
        primary_reference="ParisErdogan1963",
        oracles=("c", "d"),  # kind a dropped: the source's worked values are no acceptance basis
        applicability="central (Paris) part of the sigmoidal curve; dK0 <= dK; Kmax well below Kc",
    ),
    EquationRecord(
        id="fatigue.recommended_laws",
        description="Recommended two-stage growth-law coefficients for steels in air and in marine environments "
        "(mean and mean + 2 SD; stress ratio below and at or above 0.5), the standard deviation of log10 A per "
        "stage, and the simplified m = 3 laws",
        implemented_in="critflaw.fatigue.laws (catalogue)",
        primary_reference="King1998",
        oracles=("c",),
        applicability="steels with yield or 0.2% proof strength <= 700 N/mm^2 (air) or <= 600 N/mm^2 (marine); "
        "air up to 100 C; seawater 5-20 C at 0.17-0.5 Hz",
    ),
    EquationRecord(
        id="fatigue.thresholds",
        description="Recommended growth thresholds dK0 for welded joints (63 N/mm^(3/2) air and with cathodic "
        "protection; 0 freely corroding; 21 aluminium) and for unwelded steel as a function of stress ratio "
        "(63 for R >= 0.5; 170 - 214 R for 0 <= R < 0.5; 170 for R < 0), capped at 63 for surface flaws under 1 mm",
        implemented_in="critflaw.fatigue.laws",
        primary_reference="King1998",
        oracles=("c",),
        applicability="as the laws; the lower-bound high-R value for all welded joints",
    ),
]

EQUATIONS: dict[str, EquationRecord] = {r.id: r for r in _RECORDS}


def cite(equation_id: str) -> str:
    """Human-readable citation for an equation, honest about what has been checked."""
    record = EQUATIONS[equation_id]
    if record.primary_reference is None:
        return f"{record.description} [no primary reference given by the source standard]"
    suffix = "" if record.checked_against_primary == "yes" else " (as cited by the source standard)"
    return f"{record.description}: {REFERENCES[record.primary_reference]}{suffix}"
