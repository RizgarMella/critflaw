# 2. Stress intensity factor and reference stress solutions

The FAD coordinates of [01_fad.md](01_fad.md) need, for each flaw geometry, a stress intensity factor solution
(for $K_r$) and a reference stress solution (for $L_r$). This chapter describes the geometries implemented, the
general form in which they are combined, their validity ranges, and the rule that a solution is never evaluated
outside its range. Implementation: `critflaw.solutions` (`plate`, `embedded`, `cylinder`, `cylinder_tables`,
`cylinder_sref`, `cylinder_dispatch`).

The coefficient sets, tables and closed forms of the source standard are not reproduced here: each solution is
described by its structure and role, with the implementing module and its register id. Only the general symbolic
form of $K$, textbook limiting values and the validity ranges a user needs to set up a problem are stated. The
coefficients were transcribed from the licensed source standard, transcribed a second time independently from the
printed page and diffed, and are recorded in the equation register with the primary reference the standard cites
([references.md](references.md)). Every register entry
currently has `checked_against_primary = "no"`: the coefficients were verified against the printed standard, not
against the primary papers, and are cited "as cited by the source standard". Their knowledge-base status is in
[08_knowledge_base.md](08_knowledge_base.md).

---

## 2.1 Validity and refusal

Every solution has a validity range taken from its source. The library applies four rules:

1. **Never extrapolated.** A request outside the range is not evaluated. A deterministic `Assessment.run()` raises
   `OutOfValidityError`; with `on_invalid="mask"` (used by the probabilistic layer) the sample is returned with
   `valid = False`, every float blanked to NaN, `acceptable = False` and exit `invalid`. Internally, invalid samples
   are evaluated on a stand-in geometry inside the range so that no warnings arise, and all their outputs are then
   discarded.
2. **Never dropped.** In probabilistic runs the out-of-validity mass is reported and the failure probability is
   given as a bound pair ([04_probabilistic.md](04_probabilistic.md) §4.6). In fatigue, reaching a validity limit
   triggers re-characterisation or the end of the sample ([05_fatigue.md](05_fatigue.md) §5.3.3).
3. **Per realisation.** Constraints that depend on random variables (for example an embedded flaw's
   $2a + 2p \le B$ with a random ligament) are checked for every realisation, never once for the nominal values.
4. **The stricter reading.** Where the source states two limits for one solution, or is ambiguous, the stricter
   one is enforced and the other recorded (`docs/unsourced_items.md` §4 and Part B lists).

Table interpolation treats a query within $10^{-12}$ (relative) of a printed edge as on the edge: that is rounding,
not extrapolation.

## 2.2 The general form of K

All plate and cylinder routes assemble $K$ in one general form [register `k.general`; no open primary reference
identified (a standard textbook form); see `docs/unsourced_items.md`]:

$$ K_I = (Y\sigma)\sqrt{\pi a}, $$
$$ (Y\sigma)_p = M\,f_w\left\{k_{tm} M_{km} M_m P_m + k_{tb} M_{kb} M_b\left[P_b + (k_m - 1)P_m\right]\right\}, \qquad (Y\sigma)_s = M_m Q_m + M_b Q_b. $$

$M_m$ and $M_b$ are the membrane and bending magnification factors of the geometry at the crack-front position,
$f_w$ the finite-width correction, $M$ the bulging factor (1 for plates), $k_{tm}, k_{tb}$ stress concentration
factors, $M_{km}, M_{kb}$ weld-toe magnification factors and $k_m$ the misalignment factor. The static
`Assessment` always uses 1 for $k_{tm}, k_{tb}, M_{km}, M_{kb}$ and $k_m$: it has no input for weld-toe
magnification, stress concentration or misalignment (the low-level `critflaw.solutions.plate.stress_intensity`
accepts them, but `Assessment` does not pass them). Only fatigue stress ranges accept `ktm`, `ktb` and `km`,
through `Spectrum` ([05_fatigue.md](05_fatigue.md)).
The secondary term carries no finite-width factor (as printed in the source). The model factor `k_model`
multiplies both parts ([01_fad.md](01_fad.md) §1.1).

A negative primary K at a position is set to zero and flagged ([01_fad.md](01_fad.md) §1.5). Crack-face pressure
is not part of any K solution here; for cylinders it is excluded by the solutions themselves and the trace says so.

## 2.3 Plates

### 2.3.1 Semi-elliptical surface flaw

Depth $a$, surface length $2c$, plate thickness $B$, width $W$; assessed at the deepest point ($\theta = \pi/2$) and
the surface points ($\theta = 0$). [register `k.plate.surface`, primary reference: Newman and Raju, NASA TM 85793
[NewmanRajuTM85793]]

The magnification factors have the form

$$ M_m = \left[M_1 + M_2\left(\tfrac{a}{B}\right)^2 + M_3\left(\tfrac{a}{B}\right)^4\right]\frac{g\,f_\theta}{\Phi},\qquad M_b = H\,M_m,\quad H = H_1 + (H_2 - H_1)\sin^q\theta, $$

with $M_1, M_2, M_3, g, f_\theta, H_1, H_2, q$ functions of $a/c$, $a/B$ and $\theta$, and $\Phi$ an approximation to
the elliptic integral of the second kind (a power-law fit in $a/c$ for $a/c \le 1$, the $c/a$ form above 1). The
coefficient functions are implemented in `critflaw.solutions.plate` and not reproduced here. Two coefficient sets
apply, for $a/c \le 1$ and $a/c > 1$; the branch change at $a/c = 1$ (a jump of
order $2\times10^{-3}$ in the coefficients) is an event of the fatigue integrator. The finite-width factor is

$$ f_w = \left\{\sec\!\left[\frac{\pi c}{W}\sqrt{\frac{a}{B}}\right]\right\}^{1/2}. $$

**Validity.** $0 < a/2c \le 1$; for long, shallow flaws a depth limit that depends on $a/c$, otherwise $a/B < 1$;
$2c/W \le 0.8$ (the finite-width note's limit, enforced as a validity limit). The exact depth limit is in
`critflaw.solutions.plate` (register `k.plate.surface`). The finite-width factor is always applied: the source's
condition on flaw area for applying it is not used to switch it off. A flaw with $a/2c > 1$ is
outside validity, although the source would allow the $a/2c = 1$ solution to be used.

**Definition-pinning checks** (kind (b)): a shallow, long flaw tends to the edge-crack value; a shallow semicircular
flaw tends to the penny-shaped factor; finite-width and bending terms reduce correctly. An FE comparison of this
solution over a 34-case design is in [06_fe_option3.md](06_fe_option3.md) §6.2 (D3): it agrees within 3 % at most
stations, with a systematic negative deviation at the deepest point of deep flaws ($a/B = 0.6$) that is mesh
converged and awaits comparison with the primary reference's own tabulated values.

### 2.3.2 Through-thickness flaw

Total length $2a$ (so $a$ is the half-length), $M_m = M_b = M = 1$, one assessed position (the tip), and

$$ f_w = \left[\sec\frac{\pi a}{W}\right]^{1/2}, \qquad \text{valid for } 2a < W. $$

[register `k.plate.through`; no open primary reference identified (a classical secant correction); see
`docs/unsourced_items.md`]. The wide-plate limit $\sigma\sqrt{\pi a}$ with $a$ the half-length is a
definition-pinning test.

### 2.3.3 Embedded elliptical flaw

Height $2a$, length $2c$, ligament $p$ to the **nearer** surface, effective thickness $B' = 2a + 2p$; positions are
the minor-axis end nearest the surface ($\theta = \pi/2$) and the major-axis ends ($\theta = 0$). [register
`k.plate.embedded`, primary reference [NewmanRajuTM85793]]

$$ M_m = \left[M_1 + M_2\left(\tfrac{2a}{B'}\right)^2 + M_3\left(\tfrac{2a}{B'}\right)^4\right]\frac{g\,f_\theta}{\Phi},\qquad f_w = \left\{\sec\!\left[\frac{\pi c}{W}\sqrt{\frac{2a}{B'}}\right]\right\}^{1/2}, $$

with two branches at $a/2c = 0.5$ ($a/c$ forms below, $c/a$ forms above). **Validity:** $p > 0$;
$2a + 2p \le B$ ($p$ is the smaller ligament; equality is mid-thickness); $0 < a/2c \le 1$;
for long, shallow flaws a depth limit on $a/B'$ that depends on $a/c$ (no depth limit is given for other aspect
ratios and none is invented; exact limit in `critflaw.solutions.embedded`, register `k.plate.embedded`);
$2c/W < 0.5$ (the stricter of the two width limits the source gives; the other is recorded).

**Bending is refused.** The source gives a bending magnification $M_b$ at the minor-axis ends only, over part of
the aspect-ratio range, and none at the major-axis ends. An embedded flaw with any
non-zero bending component ($P_b$, $Q_b$, a bending range in a spectrum, or bending in the fracture-check loads) is
therefore refused with `NotImplementedError("TODO(verify): M_b at the major-axis ends of an embedded flaw")`:
assessing the major axis with $M_b = 0$ would be an invented value. The $M_b$ sets are implemented and tested
(continuity at their thresholds, $0 \le M_b \le M_m$, $M_b/M_m \to 1 - 2p/B$ as $a/B \to 0$) but dormant.
`near_surface` ("tension" or "compression": the side on which the declared bending is tensile) is required whenever
bending is present so that problems are complete when the path opens.

**Static assessment.** The embedded flaw is assessed directly (minimum over its positions). The optional ligament
re-characterisation that may relax an unacceptable result is not implemented; every embedded result carries the
flag `embedded_recharacterisation_relaxation_not_applied`. The omission is conservative under the reading that the
re-characterisation is an optional further step after an unacceptable result; that reading is an open item.

**Definition-pinning checks:** the infinite-body limits as $2a/B' \to 0$ on both branches (continuous at $a/c = 1$);
the penny-shaped value $2/\pi$ at $a = c$ within the $\Phi$ fit's error (about $7\times10^{-4}$); the Griffith limit
as $a/c \to 0$; $f_w(p = 0)$ equal to the through-flaw factor; the embedded $M_m$ below the surface-flaw deepest-point
$M_m$ at the same $a, c$ on a grid (a free surface raises K).

Embedded flaws in cylinders are not implemented (static and growth paths raise `TODO(verify)`).

### 2.3.4 Extended (full-width) surface flaw

A flaw of uniform depth $a$ across the width, with polynomial $M_m(a/B)$ and $M_b(a/B)$ and $f_w = 1$, valid for
$a/B \le 0.6$. [register `k.plate.extended`, primary reference: Rooke and Cartwright [RookeCartwright1976]] It is
used for extended flaws in cylinders under the pressure-only route (§2.4).

## 2.4 Cylinders: stress intensity factors

`Cylinder(r_i, thickness, length, k_route, sref_route="thin")`. The **K route is an explicit argument with no
default**. The two routes are different models of the same shell; they are never selected from the loads or switched
within a run, and the route is printed in every trace and is part of every cache key. A test evaluates one
pressure-only case under both routes and records the (non-zero) difference.

Flaw objects: `CylinderSurfaceFlaw(a, two_c, orientation, position)` (axial or circumferential, internal or external),
`CylinderThroughFlaw(two_a, orientation)` (circumferential length measured at mid-thickness), and
`ExtendedFlaw(a, orientation, position)` (fully circumferential, or extended axial over the full length).

### 2.4.1 Pressure: one declared record

Internal pressure enters through `PressureLoad(pressure, hoop_per_pressure, axial_per_pressure)`. The two
pressure-to-stress factors are the user's (the source gives no conversion); the membrane stress due to pressure is
the hoop factor times the pressure for axial flaws and the axial factor times the pressure for circumferential flaws.
The **same** pressure feeds the thick-wall limit pressures (§2.5), so $K$ and $L_r$ always describe one loading.
Other named components are a declared membrane stress `pm`, a global-moment membrane stress `pm_moment` (applied at
the flaw position farthest from the bending axis, where the source's figures place the flaw) and the through-wall
bending `pb`, tensile on the outside unless declared `bending_tensile_surface="inside"` (then the whole bending term
changes sign at both surfaces by linear superposition).

### 2.4.2 Route `"pressure_only"`: plate factors with bulging

The flat-plate $M_m$, $M_b$ with a bulging factor $M$ and the plate $f_w$ with $W$ the cylinder length.

The bulging factors are implemented in `critflaw.solutions.cylinder`; their constants are not reproduced here.

- **Axial through-wall flaw** (half-length $a$): a Folias-type factor, increasing from 1 with $a^2/(r_m B)$
  [register `k.cylinder.bulging_through`; no open primary reference identified; see `docs/unsourced_items.md`]. Its
  only independent check is the comparison against the tabulated through-wall factors over $5 \le r_m/B \le 100$
  (a kind (b) test with the difference asserted within a band).
- **Axial surface flaw**: a ratio built from $a/B$ and a Folias-type factor $M_T$ in $c^2/(r_m B)$, equal to 1 for a
  vanishing flaw and increasing with depth [register `k.cylinder.bulging_surface`, primary
  reference: Kiefner et al. [KiefnerEtAl1973]]. It is singular at $a = B$; a growing flaw of this kind is ended at
  $a/B = 0.9$ ([05_fatigue.md](05_fatigue.md) §5.3.3).
- $M = 1$ for circumferential and extended flaws.

**Eligibility.** The route carries the pressure-induced membrane stress only; a realisation with any mechanical
primary stress or moment is out of validity (never re-routed). The source prints no range for the bulging factors
(the referenced plate limits apply); the library adopts the plate limits and, in addition, the tabulated through-wall
range $5 \le r_m/B \le 100$, so that the bulging factors are never applied to a shell the source never tabulated.
The shell parameter range is not adopted; the trace prints the largest
$\lambda = [12(1-\nu^2)]^{1/4} a/\sqrt{r_m B}$ at which a through-wall bulging factor was used when it exceeds the end
of its only independent check.

### 2.4.3 Route `"tabulated"`: interpolated factors

$M = f_w = 1$ and the magnification factors come from tables printed in the source, interpolated **multilinearly in
the printed variables**, without transformation:

| Family | Axes | Range (as printed) | Register / primary reference |
|---|---|---|---|
| axial and circumferential through-wall: $M_1 \dots M_4$, with $M_m = M_1 \pm M_2$, $M_b = M_3 \pm M_4$ at the outside (+) / inside (−) surface | $\lambda$, $r_m/B$ | $5 \le r_m/B \le 100$; $\lambda$ up to each column's last tabulated value | `k.cylinder.tables.through` [FranceGreenSharples1996] |
| axial internal / external surface; extended axial internal | $a/c$, $a/B$, $B/r_i$ | $0 \le a/B \le 0.8$, $0.05 \le a/c \le 1$, $0.1 \le B/r_i \le 0.25$, $2c/W \le 0.15$ | `k.cylinder.tables.axial_surface` [RajuNewmanVessels] |
| extended axial external | $B/r_i$, $a/B$ | $0 \le a/B \le 0.8$, $0.001 \le B/r_i \le 0.333$ | `k.cylinder.tables.axial_extended_external` [AndersonEtAlWRC471] |
| circumferential internal surface | $a/c$, $a/B$, $B/r_i$ | $0 \le a/B \le 0.8$, $0.1 \le a/c \le 1$, $0.1 \le B/r_i \le 0.2$ | `k.cylinder.tables.circumferential_surface` [DedhiaHarris1983] |
| fully circumferential internal | $a/B$, $B/r_i$ | $0 \le a/B \le 0.8$, $0.1 \le B/r_i \le 0.2$ | `k.cylinder.tables.circumferential_extended` [JosephErdogan1989] |

A circumferential through-wall flaw also carries a membrane multiplier $\beta_c \ge 1$ on $P_m$, a function of
$a/r_m$ that tends to 1 for a short flaw (implemented in `critflaw.solutions.cylinder`). The through-wall tables are sparse columns: each
$r_m/B$ column is printed at its own $\lambda$ values, so the value is interpolated in $\lambda$ within each of the two
bracketing columns and then linearly in $r_m/B$; the interpolable end of the $\lambda$ range at a query is the
smaller of the two bracketing columns' ends.

The circumferential **external** surface flaw uses the plate solution under both routes, as the source directs, with
the plate $f_w$ and $W$ the cylinder length [`k.plate.surface`]. The fully circumferential **external** flaw uses a
closed-form membrane solution in $r_i/r_o$ and $a/B$ with bending stresses added to the membrane ones
[register `k.cylinder.extended_external`, primary reference [RookeCartwright1976]]; it is singular at $a = B$.

**Corner envelope.** Every tabulated evaluation also returns the interval $[\min, \max]$ of the containing cell's
corner values, labelled *bound under within-cell monotonicity (checked at the nodes, not proved)*. For
$M_1 \pm M_2$ and $M_3 \pm M_4$ the envelopes are combined sign-aware. It is never called an error; interpolation
uncertainty is listed among the excluded sources until calibrated by published interior values or FE.

**Suspect cells.** Cells that the independent transcription or the diff flagged as printed anomalies (anti-monotone
entries, a printed gap in one column, identical columns in two halves of a table, a bending factor above the
membrane factor at one position) are marked `suspect`. An evaluation that interpolates through one carries the
`suspect` status, the result is flagged, and the knowledge-base status of the table for that evaluation is
`unverified`. The reported probability mass of such samples and a sensitivity re-run with each suspect cell replaced
by the interpolant of its neighbours are **not implemented**. An alternative "upper envelope" interpolation option is
not implemented either.

**Checks** (kind (b)): every $a/B = 0$ row of the surface tables reproduces the plate $M_m$ to three decimals; every
table is asserted monotone along every axis in the direction physics requires, except at the listed suspect cells;
the difference to the plate value decreases as curvature decreases; finite-flaw $M_m \le$ extended-flaw $M_m$ at the
same $(a/B, B/r_i)$; bulging factors tend to 1 as $a^2/(r_m B) \to 0$, and the coefficient of the through-wall
bulging factor is pinned against the tabulated factors; the fully circumferential external solution gives the
edge-crack factor at zero depth.

## 2.5 Reference stress solutions

The reference stress is $\sigma_{ref} = L_r\,\sigma_Y$, the stress that the primary loading represents relative to
plastic collapse of the flawed section. Every solution tends to the applied stress as the flaw vanishes and to the
net-section collapse value as it fills the section; these limits are definition-pinning tests.

### 2.5.1 Plates

The closed forms are implemented in `critflaw.solutions.plate` and `critflaw.solutions.embedded` and are not
reproduced here.

**Surface flaw** [register `sref.plate.surface`, primary reference: Willoughby and Davey [WilloughbyDavey1989]].
A net-section expression combining $P_m$ and $P_b$, in which the flaw enters through an effective area fraction
$\alpha''$ of the section; $\alpha''$ has one form for a plate wide enough relative to the flaw and another, based on
$2c/W$, otherwise. With negligible bending restraint (`bending_restraint="pin_jointed"`) the bending term is
augmented by a contribution from $P_m\alpha''$ (the eccentricity of the net section).

**Through-thickness flaw** [register `sref.plate.through`, [WilloughbyDavey1989]]: the same combination of $P_m$
and $P_b$ divided by the net-section fraction $1 - 2a/W$.

**Embedded flaw** [register `sref.plate.embedded`, [WilloughbyDavey1989]]: the same structure with an additional
ligament term in the denominator and $\alpha''$ formed with the flaw height $2a$.

**Extended embedded flaw** (full width; no K solution exists for it) [register `sref.plate.embedded_extended`,
primary reference: Lei and Budden [LeiBudden2004]]: a closed form in $a/B$, the ligament and $P_b/P_m$, valid within
a depth limit that depends on the ligament; NaN (out of validity) beyond. The implementation uses an algebraically equivalent form
that is stable at $P_m = 0$.

### 2.5.2 Cylinders

`Cylinder(..., sref_route=...)` selects the family: `"thin"` (thin-wall reference stresses, the default),
`"thick_pressure"` (thick-wall limit pressures for axial surface and extended flaws; requires a `PressureLoad`) or
`"thick"` (thick-wall limit loads). Branches are alternatives and are never combined.

| Branch | What | Register / primary reference |
|---|---|---|
| axial, thin wall | through-wall, surface and extended flaws under membrane and through-wall bending, with an empirical multiplier on the membrane term and a Folias-type factor $M_T$ (or $M_s$ for surface flaws) | `sref.cylinder.axial_thin` [Folias1984] |
| axial, thick wall, pressure | limit pressures of internal and external axial surface flaws (crack-face pressure included) and their extended forms; $L_r = p/p_L$ expressed as $\sigma_{ref} = \sigma_Y p/p_L$ | `sref.cylinder.axial_pressure` [Lei2008] |
| axial external, thick wall, membrane and bending | normalised limit force and moment; valid up to a geometry-dependent limit on $2c/W$ | `sref.cylinder.axial_thick_bending` [Lei2004] |
| circumferential, thin wall | through-wall under tension, bending and pressure ($2a/(\pi r_i) \le 0.5$, $B/r_i \le 0.2$); surface ($c/(\pi r_i) \le 0.8$); fully circumferential | `sref.cylinder.circumferential_thin` [KastnerEtAl1981] |
| circumferential, thick wall | net-section limit force and moment under tension and global bending: $L_r = F/F_L = M/M_L$ at the applied ratio $M/(r_m F)$ | `sref.cylinder.circumferential_thick` [JonesEshelby1990] |

Eligibility is enforced per realisation: the thick-wall pressure branch refuses any mechanical primary stress; the
thick-wall circumferential branch refuses through-wall bending.

**Readings chosen where the source is inconsistent.** Each is implemented as one reading with the alternative
recorded in `docs/unsourced_items.md`; where the readings differ by more than the tolerance anywhere, the
conservative one is implemented:

- for external axial flaws, the outer radius replaces the inner one in $M_T$;
- the half-angle of a part-circumferential surface flaw in the thick-wall net-section limit: $c/r_i$ for internal
  flaws and $c/r_m$ for external flaws (the larger angle, the smaller limit load). The two readings of each pair
  differ by sampled maxima of about 10 % for $B/r_i \le 0.2$ and 31–36 % for $B/r_i \le 1$ on the recorded grid
  (sampled maxima, not proven bounds; the source prints no validity range for this solution).

## 2.6 Crack-front length for toughness

The Master Curve toughness depends on the crack-front length $l$ ([03_materials_toughness.md](03_materials_toughness.md)
§3.2): $2c$ for a surface flaw, $4c$ for an embedded flaw, $2B$ for a through-thickness or extended flaw, always capped
at $2B$. The cylinder dispatch supplies the length per flaw kind.

## 2.7 Assumptions and limits of this chapter

- Membrane and bending stresses are linear through the thickness. Arbitrary through-wall stress profiles are not
  supported (there is no input for them); only membrane plus bending.
- Residual stress is membrane (plus bending if declared as $Q_b$).
- No flaw interaction; no spheres or nozzles.
- Embedded flaws: membrane loading only, plates only.
- Cylinder pressure-to-stress factors are the user's; the K solutions exclude crack-face pressure.
- Tabulated cylinder factors: multilinear interpolation; interpolation error is not quantified beyond the corner
  envelope.
- Kinds (c) and (d) of verification cannot catch a coefficient that is wrong in the printed source; no K solution is
  yet compared with its primary reference's own published values ([08_knowledge_base.md](08_knowledge_base.md)).
