# Items for the owner's decision

## 1. Implemented items for which the source standard cites no primary reference

These are implemented and used. Their register records carry `primary_reference=None`, and results that depend on them say so. No citation has been invented. The evident origin of the FAD items is the R6 procedure, which the standard lists in its bibliography but does not cite in the defining clause.

| Register id | What | Suggested action |
|---|---|---|
| `fad.cutoff` | Plastic collapse cut-off `Lr,max` | Confirm R6 as the citation |
| `fad.option1.continuous` | Option 1 assessment line, continuous yielding | Confirm R6 as the citation |
| `fad.option1.discontinuous` | Option 1 assessment line, discontinuous yielding | Confirm R6 as the citation |
| `material.luders_strain` | Lüders strain estimate from yield strength | Identify source |
| `fad.kr` | Definition of the fracture ratio with `ρ` or `V` | Confirm R6 as the citation |
| `residual.as_welded` | As-welded residual stress and relaxation with primary load | Identify source |
| `residual.pwht` | Residual stress levels after PWHT | Identify source |
| `k.general` | General form `K = (Yσ)√(πa)` with primary and secondary parts | None needed (standard textbook form) |
| `k.plate.through` | Through-thickness flaw in a plate, secant finite-width correction | Identify source (a classical result) |
| `fad.option2` | Option 2 assessment line from the true stress-strain curve | Identify source |
| `fad.option2.monotone` | Monotone Option 2 line by a non-decreasing secant plastic compliance (this library's rule) | None needed (the library's own monotonicity rule) |
| `material.power_law` | Stress-strain estimate from yield strength, tensile strength and a hardening exponent | Identify source |
| `toughness.k_from_j` | Fracture toughness from the J-integral | None needed (standard fracture-mechanics relation) |
| `toughness.k_from_ctod` | Fracture toughness from CTOD with the constraint factor m | Identify source |
| `toughness.subsize` | Thickness correction of cleavage toughness from sub-size specimens | Identify source |
| `toughness.charpy_upper_limit` | Upper limit on Charpy-based toughness estimates | Identify source |
| `toughness.charpy_subsize` | Transition temperature shift for sub-size Charpy specimens | Identify source |
| `toughness.mote` | Minimum-of-three-equivalent toughness | Identify source |
| `toughness.lower_bound` | Lower-bound toughness by a one-sided tolerance limit (computed exactly) | Identify source |
| `k.cylinder.bulging_through` | Bulging factor of an axial through-wall flaw in a curved shell under internal pressure | Identify source |

## 2. Items cited to a primary reference that has not been checked against that reference

`checked_against_primary` is `no` for every record. The coefficients were verified against the printed standard (PDF), not against the papers below. Results cite them "as cited by the source standard".

| Register id | Primary reference as cited |
|---|---|
| `k.plate.surface` | Newman and Raju, NASA Technical Memorandum 85793 |
| `sref.plate.surface`, `sref.plate.through` | Willoughby and Davey 1989, ASTM STP 1020 |
| `fad.rho.simplified` | Ainsworth 1986, Engng Fract Mech 24 |
| `fad.v.simplified` | Ainsworth, Smith and Wiesner 1999, SINTAP/NE/018 |
| `fad.kjs.plastic_zone` | R6 Revision 4 |

## 3. Publication question

The implemented coefficient sets were transcribed from a copyrighted standard. Whether each may ship in a public release (as opposed to being loaded from a user-supplied file) has not been decided. Items whose primary reference is openly published are the safest to ship once checked against that reference.

## 4. Choices made where the source is ambiguous or silent

| Point | Choice |
|---|---|
| Boundary points left unassigned by strict inequalities on both sides (`χ = 5.2`, `Lr = 1.05`) | The functions are continuous there; assigned to the upper branch |
| Estimated Lüders strain applicability | The stricter limit is enforced (yield strength below 946 MPa) |
| Flaw with `a/2c > 1` | Treated as outside validity, although the source permits using the `a/2c = 1` solution |
| Finite-width factor "applicable up to `2c/W = 0.8`" | Enforced as a validity limit |
| Finite-width factor only "where flaw area exceeds 10% of the section" | Always applied |
| Secondary stress intensity factor | No finite-width factor, as printed |
| Self-balancing term denominator printed as an undefined symbol | Read as the fracture toughness |
| Negative primary K at the deepest point under bending (deep flaws) | Set to zero and flagged; no rule in the source |
| Plastic-zone correction for secondary stress acting alone | Plane stress by default (more conservative); recorded in the trace |
| Yielding behaviour "unknown" | Pointwise minimum of the two Option 1 forms (the literal hybrid can step upward) |

## 5. Apparent misprints in the printed standard (found by calculation, print verified at high resolution)

| Where | Printed | What the evidence says | What the library does |
|---|---|---|---|
| Probabilistic annex, Master Curve median equation | exponent coefficient `0.01055` | The Master Curve annex and the Charpy annex use `0.019`; the Charpy annex's own worked example (90.8 MPa√m) is reproduced with `0.019` and gives 70.5 with `0.01055` | Uses `0.019` |
| Tolerance-factor table for the lower-bound toughness, 16 results | `1.307` | The exact one-sided tolerance factor (non-central t) reproduces every other entry of the table to its printed precision and gives `1.301` here | Computes the factor exactly; does not use the table |
| Self-balancing residual stress term | divided by an undefined symbol | Evidently the fracture toughness | Divides by the fracture toughness |
| Alternative plasticity-interaction function `h(Lr)` | discontinuous at `Lr = 1.07` as printed | Not implemented in this library yet | Listed for when it is |

## 6. Phase 2 modelling choices where the source is silent

| Point | Choice |
|---|---|
| Yield plateau in measured stress–strain data | Declared by the user (`yielding="plateau"` with the Lüders strain); never detected from the data, because no detector was reliable in review |
| Option 2 line that rises after a Lüders increment | Made monotone by not letting the secant plastic compliance decrease (conservative; the difference is reported, 0.015 to 0.03 for a lower-yield-point steel) |
| Plastic strain below 2% of the yield strain in measured data | Treated as zero (noise level); raises the line by at most about 0.012 below `Lr = 0.75` for clean data |
| Sampling a measured curve with random yield strength | The plastic strain as a function of stress/yield is the fixed shape, which preserves the offset definition of yield |
| Positive quantities with a normal distribution | Must be written as an explicitly truncated normal |

## Phase 3, Part A: modelling choices of this library (design note v6.1, marked (choice))

The source gives no rule for these; each is printed with the result where it applies.

- Validity statements of the growth-law catalogue (temperature, environment, strength) are carried as text; nothing checks them (F3).
- Correlation between the two stages' log A is a required argument with no default; results report a sensitivity at 0 and 1 (F4).
- Whether a random load scale is shared between the cyclic ranges and the extreme load is a required argument with no default (F5).
- Spectrum order: a repeating pattern is averaged (expectation form); an ordered histogram compares three declared orders and does not search for the most damaging order (F6).
- Surface-flaw K formulas evaluated at a = B so that the boundary set is closed (F8).
- Re-characterisation at the aspect-ratio validity limit is a conservative device, not a physical leak (F7).
- Order of operations at a step boundary: fracture check, inspection, repair (F15).
- A through flaw is detected with probability 1 unless a second PoD is given; successive inspections are independent given the size; no false calls (F15, F16).
- Within a load-process step the order of a leak and a load-induced failure is a convention (F12).
- The quadrature finish for a through flaw near the width limit: panel-wise Gauss-Legendre with the kinks of the rate as grid points and in-panel solves for interior sizes (F8, section-9 amendment); the "remaining time shorter than the step" trigger is withdrawn.
- A repair clears the removed flaw's event record; the repair flaw's own breakthrough is a new leak event; the removed flaw's history is reported separately (F15).
- The deterministic life (`grow`) follows the headline treatment of `solve`: first failing or out-of-validity checkpoint on either side of a flaw-type change, end of the growth solution's validity, or a leak when leaks count; inspection policies are not simulated deterministically.
- Tabulated multilinear interpolation of table cells and the "suspect" cell marking are Part B items (see the Part B note).

## Phase 3, Part B: items with no reference in the source, and modelling choices (design note v3.1)

Items the source gives without a primary reference (`primary_reference=None`, or the nearest cited reference used for the family):

- The bulging factor of an axial through-wall flaw under internal pressure (the surface-flaw form cites its references; the through-wall form cites none). Its only independent check is the comparison against the tabulated through-wall factors over 5 <= r_m/B <= 100.
- The finite-width factor of the embedded flaw (given with the embedded solution; cited to the same reference as the solution).
- The source's statement that the plate solutions may be applied to circumferential external surface flaws and to embedded flaws in shells (no reference).
- The reference-stress forms P.9.3, P.9.5 (extended axial flaws), P.10.6 (embedded flaws in cylinders): no reference in their headings; recorded under the family's reference.

Apparent misprints and inconsistencies (entered as printed; the implemented reading stated):

- Table M.6, r_m/B = 50: the two entries at lambda = 15.143 are anti-monotonic; r_m/B = 5 at lambda = 0.177 (and 0.355) jumps above the r_m/B = 10 column. Table M.7: the surface-point columns are identical in both halves and two cells have M_b(s) > M_m(s). Table M.1, r_m/B = 10: seven printed entries with a gap between lambda = 8.048 and 12.072. All are `suspect` cells: each evaluation that interpolates through one carries the `suspect` status and a result whose samples touched one carries a flag; the reported mass of such samples in probabilistic runs and the sensitivity re-run (design B13) are not implemented (owed).
- Table M.2, a/c = 0.05 block at a/B >= 0.6 (three cells, large difference between the two curvature columns) and the Table M.3 a/B = 0.8 pair: the diff reduced the doubt (the ratios track the extended-flaw table) but the transcriber flagged them as printed anomalies, so they are marked `suspect` as well (the conservative side, design G5). Table M.5, a/B = 0 printed 1.120 (polynomial value, not an error): not marked.
- The radius substitution for external axial flaws is printed "the inner radius for the outer radius" in P.9.4/P.9.5 and the reverse in P.10.4 (amended text); implemented as "the outer radius replaces the inner one" in both families.
- P.9.7's closing sentence refers to an internal flaw in the external-flaw clause; implemented as the external extended case (1/M_ex = 0).
- The embedded-flaw width limit is printed as 2c/W < 0.5 (conditions) and 0.8 (finite-width note); the stricter is enforced.
- The P.10.2 case conditions are lost in the OCR markdown; two independent readings of the PDF page agree.

Modelling choices (marked (choice) in the design note):

- A bending-loaded embedded flaw is refused (no bending magnification at the major-axis ends is given); the bending coefficient sets are implemented and tested but dormant.
- `near_surface` is a declaration, required whenever bending is present; the near minor-axis end is the assessed depth position.
- a/2c > 1 is outside validity for the embedded flaw (the Phase 1 surface-flaw rule), not evaluated at a/2c = 1.
- The static assessment of an embedded flaw is the source's direct assessment; the optional re-characterisation relaxation after an unacceptable result is not implemented (every embedded result says so). Disputed by red team A as unverifiable from the paraphrase: `reviews/DISPUTES.md` D-1.
- The crack dimension of the secondary-only plastic-zone correction is the `a` of the K solution (half-height for the embedded flaw).
- The pressure-to-stress factors of `PressureLoad` are the user's; the K solutions exclude crack-face pressure.
- The global moment acts at the flaw position farthest from the bending axis (M.25).
- Through-wall bending is tensile on the outside unless declared otherwise.
- Pressure-only route: M = 1 for extended flaws; the range 5 <= r_m/B <= 100 of the tabulated through-wall solutions is adopted as the validity of the pressure-only kinds; the lambda range is not.
- Table interpolation is multilinear on the printed axes; the corner envelope is a bound only under within-cell monotonicity (checked at the nodes); interpolation uncertainty is otherwise excluded.
- The F3 threshold cap (surface flaws under 1 mm) is not applied to embedded flaws (the less conservative side for a shallow embedded flaw).
- Growth of a through-wall flaw under the tabulated route: the interpolable end of the lambda range depends on r_m/B (each printed column ends at its own last lambda; the end at a query is the smaller of the two bracketing columns' ends), so the range-end boundary is column dependent, not the global 12.211 / 15.143; the last tenth of that range is exempt from the pass-growth guard as the Part A terminal zone is (the terminal step lands on the end within one pass).
- Embedded flaws in cylinders: neither the static assessment nor the growth is implemented (both paths raise `NotImplementedError("TODO(verify): ...")`); the static path would use the plate solution as the source directs, and benchmark B19 (the three-transition cascade) is owed with the growth path.
- The half-angle of a part-circumferential surface flaw in the thick-wall net-section limit load (P.11): the source prints no radius for the flaw's arc (the through-wall length is defined at mid-thickness) and no validity range. Readings: the flaw's own surface (c / r_i internal, c / r_o external) and mid-thickness (c / r_m). They differ in the net-section Lr by more than the Phase 1 tolerance everywhere, so by the design rule (G7) the conservative member of each pair is implemented: c / r_i for internal and c / r_m for external flaws (the larger angle, the smaller limit load; conservative at every grid point to within 2e-4, the exception being the source's branch (ii) with closure ignored, where L_N rises slightly with the angle). Sampled maxima of the relative difference between the two readings of a pair over the grid B/r_i in {0.02 .. 1}, a/B in {0.05 .. 0.8}, flaw half-angle c/r_i in {0.02 .. 0.9} pi, load ratio from pure tension to pure moment (1800 cases, tested): 10 % (internal) and 10 % (external) for B/r_i <= 0.2, 31 % and 36 % for B/r_i <= 1 (the differences grow with wall ratio, depth and length). These are sampled maxima on that grid, not proven bounds. The earlier record of "less than 5 %" sampled one depth and one load ratio and was wrong.
- Kinds whose K is singular at a = B ((M.22) axial surface flaw under the pressure-only route; (M.26) fully circumferential external flaw) are ended at a/B = 0.9 with leak_assumed (trigger 17): a declared end, not a source rule.
- A through-wall bending declared tensile on the inside surface negates the tabulated bending term at both surfaces: linear superposition of the source's outside-tensile convention (derived, not a coefficient).
- The circumferential external surface flaw (the plate solution, as the source directs) carries the plate finite-width factor with W the cylinder length under both routes.
- Table interpolation treats a query within 1e-12 (relative) of a printed edge as on the edge (rounding, not extrapolation).
