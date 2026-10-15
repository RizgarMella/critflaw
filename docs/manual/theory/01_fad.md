# 1. The failure assessment diagram

This chapter describes the deterministic engine: how a flaw is placed on the failure assessment diagram (FAD),
which assessment lines are available, how secondary stress is treated, how acceptance is decided and what the
reserve factors mean. The stress intensity factor and reference stress solutions that feed it are in
[02_solutions.md](02_solutions.md); toughness and tensile laws are in
[03_materials_toughness.md](03_materials_toughness.md). Implementation: `critflaw.fad` (`curves`, `option2`, `kr`,
`margins`), `critflaw.assessment`, `critflaw.reserve`.

**Primary references.** The FAD definitions themselves — the cut-off, the Option 1 lines, the definition of `Kr`
with `ρ` or `V` — carry `primary_reference=None` in the register: no open primary reference identified; see
`docs/unsourced_items.md`. The simplified `ρ` is cited to Ainsworth (1986)
[Ainsworth1986], the simplified `V` to Ainsworth, Smith and Wiesner (1999) [AinsworthSmithWiesner1999], and the
plastic-zone-corrected secondary K to R6 Revision 4 [R6]. Option 2 and its monotone construction: no open
primary reference identified; see `docs/unsourced_items.md` (the monotone rule is this library's own).

---

## 1.1 The two coordinates

A flaw under load is represented by a point $(L_r, K_r)$.

**Load ratio.** The proximity to plastic collapse:

$$ L_r = \frac{\sigma_{ref}}{\sigma_Y} $$

where $\sigma_{ref}$ is the reference stress of the flawed section under the *primary* loads, obtained from a
limit-load solution ([02_solutions.md](02_solutions.md) §2.5), and $\sigma_Y$ is the yield or 0.2 % proof
strength. Secondary stresses do not enter $L_r$.

**Fracture ratio.** The proximity to fracture, in its simplest form

$$ K_r = \frac{K_I}{K_{mat}} $$

with $K_I$ the mode-I stress intensity factor and $K_{mat}$ the fracture toughness. With secondary stress present
$K_r$ is modified by a plasticity interaction term (§1.5).

Both coordinates are evaluated at every assessed crack-front position (deepest and surface points of a surface
flaw; the tip of a through-thickness flaw; the near minor-axis end and the major-axis ends of an embedded flaw;
the positions supplied by the cylinder dispatch). The acceptance test (§1.6) must pass at every position.

**Model factors.** An `Assessment` accepts three multiplicative model-uncertainty factors, all 1 by default
("not accounted for"): `k_model` on the primary and secondary stress intensity factors, `sref_model` on the
reference stress, `fad_model` on the assessment line. A factor `m` on the line is applied by assessing
$(L_r, K_r/m)$ against the unmodified line, which is exactly equivalent (including at the cut-off) and keeps the
margin bracket of §1.7 valid. A factor that is held fixed leaves the corresponding item in the result's
list of excluded uncertainty sources.

## 1.2 The assessment line and the cut-off

The acceptable region is bounded by an assessment line $K_r = f(L_r)$ and a vertical cut-off at
$L_r = L_{r,max}$.

**Cut-off.** [register `fad.cutoff`; no open primary reference identified; see `docs/unsourced_items.md`]

$$ L_{r,max} = \frac{\sigma_Y + \sigma_U}{2\sigma_Y} $$

so that $L_{r,max}\,\sigma_Y$ is the flow strength. In a deterministic assessment the line and the cut-off use
the supplied (mean) properties; in a probabilistic run every realisation builds its own line and its own
$L_{r,max}$ from its sampled $\sigma_Y$ and $\sigma_U$. The library requires $\sigma_U > \sigma_Y$; when both are
random the ordering must hold by construction, which is why the tensile strength may be given as a `Ratio` of the
yield strength with support above 1 ([04_probabilistic.md](04_probabilistic.md) §4.1).

**Properties every line has.** Each curve object

- is non-increasing on its domain (analytic curves by construction; tabulated laws are made so, §1.4);
- satisfies $f \le 1$ and $f = 0$ for $L_r \ge L_{r,max}$;
- declares its domain and its discontinuities (`step_locations`) and is never evaluated outside the domain;
- has $f(0)$ defined by a limit, never by evaluating $0/0$.

These properties are what make the radial margin of §1.7 well defined.

## 1.3 Option 1

Option 1 needs only $\sigma_Y$, $\sigma_U$ and $E$. The **yielding behaviour is a required input with no
default**: `"continuous"`, `"discontinuous"` or `"unknown"`. A wrong assumption changes $f(1)$ by up to a factor of
about two, so the library will not guess it.

The closed forms and fitted coefficients of the Option 1 lines are not reproduced here; they are implemented in
`critflaw.fad.curves.Option1Curve` and recorded in the register under the ids given below.

**Continuous yielding** [register `fad.option1.continuous`; no open primary reference identified; see
`docs/unsourced_items.md`]. For $L_r \le 1$ the line is the elastic term $(1 + L_r^2/2)^{-1/2}$ multiplied by a
factor that decays from 1 with $L_r$ at a rate set by $E/\sigma_Y$; for $1 < L_r < L_{r,max}$ it is a power of
$L_r$ starting from $f(1)$, with an exponent set by $\sigma_Y/\sigma_U$; it is 0 for $L_r \ge L_{r,max}$. The line
is continuous at $L_r = 1$. The material dependence of the decay rate distinguishes this form from the older
fixed-coefficient line; a regression test pins the difference.

**Discontinuous yielding** (a Lüders plateau) [register `fad.option1.discontinuous`; no open primary reference
identified; see `docs/unsourced_items.md`]. For $L_r < 1$ the line is the elastic term; at $L_r = 1$ it drops to a
value set by the normalised Lüders strain $E\,\Delta\varepsilon_L/\sigma_Y$; beyond, it is the same power branch
as the continuous form, starting from the post-step value. The step at $L_r = 1$ is vertical and the value at
$L_r = 1$ is the post-step value. $\Delta\varepsilon_L$ is the Lüders strain, supplied by the user or estimated from
$\sigma_Y$ by a linear estimate (`critflaw.fad.curves.luders_strain_estimate`) that is valid only below a yield
strength limit [register `material.luders_strain`; no open primary reference identified; see
`docs/unsourced_items.md`]. Above that limit the estimate is refused and the strain must be supplied.

**Unknown yielding.** The rule for unknown yielding assumes no plateau below $L_r = 1$ and a plateau from
$L_r = 1$. Taken
literally, the hybrid line can step *upward* at $L_r = 1$ (for high yield strengths with the estimated Lüders
strain, or for a small supplied Lüders strain), which would make it increasing. The library
therefore defines it as the pointwise minimum of the two forms: the continuous form for $L_r < 1$ and, for
$L_r \ge 1$, the power branch starting from the smaller of the two $f(1)$ values. It is never less conservative
than either form and it is non-increasing for every input.

## 1.4 Option 2

Option 2 builds the line from the material's true stress–strain curve [register `fad.option2`]. Write
$s = \sigma/\sigma_Y$ and the normalised true strain $e = E\varepsilon/\sigma_Y$. The line is

$$ f(L_r) = \left[\frac{e(L_r)}{L_r} + \frac{L_r^3}{2\,e(L_r)}\right]^{-1/2}, \qquad L_r < L_{r,max}, $$

and $0$ beyond. A material law supplies the plastic true strain $\varepsilon_p(s) \ge 0$, so that
$e(s) = s + (E/\sigma_Y)\,\varepsilon_p(s)$ and $f \le 1$. For a linear-elastic law ($\varepsilon_p \equiv 0$) the
line reduces to $(1 + L_r^2/2)^{-1/2}$, an oracle used in the tests.

**The monotone construction** [register `fad.option2.monotone`; this library's rule]. Write
$r(s) = e(s)/s = 1 + \kappa\,q(s)$ with $\kappa = E/\sigma_Y$ and $q(s) = \varepsilon_p(s)/s$ (the secant plastic
compliance). Then $f^{-2} = r + s^2/(2r)$ and

$$ \frac{d f^{-2}}{ds} = r'\left(1 - \frac{s^2}{2r^2}\right) + \frac{s}{r}. $$

The unmodified line can therefore rise only where $q$ decreases: after a Lüders increment, or over a stretch of
nearly constant plastic strain. The library replaces $q$ by its running maximum
$\hat q(s) = \max_{t\le s} q(t)$:

$$ \hat e(s) = s\,(1 + \kappa\,\hat q(s)), \qquad h(s) = \left[\frac{\hat e}{s} + \frac{s^3}{2\hat e}\right]^{-1/2}. $$

Under the **validity condition** $e(s) \ge s^2/\sqrt2$ (with the unmodified $e$) the construction is

- *monotone*: $d h^{-2}/ds \ge 0$ because $\hat r' \ge 0$ and $s \le \sqrt2\,r \le \sqrt2\,\hat r$;
- *conservative*: at fixed $s$, $f^{-2}$ increases with $r$ on $[r, \hat r]$, so $h \le f$ everywhere;
- *exact and cheap*: $\varepsilon_p$ is linear between data points, so $q$ is monotone on each interval and the
  running maximum is attained at a knot; $\hat q$ does not depend on $\kappa$ and is computed once per law.

The condition always holds for $s \le \sqrt2$ (because $e \ge s$). Beyond $\sqrt2$ — reachable only when
$\sigma_U/\sigma_Y > 1.83$ — it requires the material to have strained plastically by $1.41\sigma_Y$. A tabulated
law is checked exactly at its knots (the condition is linear against a convex bound); a power law on 200 points
with a 0.1 % margin. A deterministic assessment whose law violates it raises; a realisation that violates it is
out of validity. The largest lowering $\max(f - h)$ is reported in the trace (0.015 to 0.03 for a
lower-yield-point power law with a Lüders strain); for tabulated data a lowering above 0.05 is an error (the data
are suspect).

**Laws.** Three laws implement the Option 2 protocol:

- `PowerLaw(offset, n, luders_strain=None)`: $\varepsilon_p = \text{offset}\cdot s^n$, plus a Lüders strain step
  at $s = 1$ if declared. The offset must be chosen explicitly (0.002 for a 0.2 % proof strength, 0 for a lower
  yield point). [register `material.power_law`; no open primary reference identified]
- `TabulatedLaw.from_engineering(...)` / `.from_true(...)`: a measured record, cleaned in a fixed, reported
  sequence ([03_materials_toughness.md](03_materials_toughness.md) §3.6.2). A tabulated law ends at its last point;
  if a realisation's $L_{r,max}$ lies beyond it, the last point acts as the cut-off (exit `domain_end`).
- `WarpDeformationLaw(E, ν, σ0, n)`: the uniaxial law of the WARP3D deformation-plasticity material, so that
  Option 2 and the FE-based Option 3 describe one material ([03_materials_toughness.md](03_materials_toughness.md)
  §3.6.3). Its curve is built with `Option2Curve.from_warp_law(law, uts)`, which takes $E$ and $\sigma_Y := \sigma_0$
  from the law; the direct constructor refuses a mismatch.

**Sampling.** With random tensile properties the plastic-strain shape $\varepsilon_p(s)$ is held fixed and
$\kappa = E/\sigma_Y$ comes from each realisation. This preserves the offset definition of yield.

**Sensitivity to data.** Option 2 lines from measured data are sensitive to measurement noise: a mean change of
about 0.015 in $f$ (up to 0.04) was measured for 1 MPa stress noise and $5\times10^{-5}$ strain noise. Coarse
stress quantisation produces a stepped line about the unrounded one. These are properties of Option 2 with noisy
data, not of the implementation.

## 1.5 Secondary stress and the plasticity interaction

Primary stresses ($P_m$, $P_b$) contribute to plastic collapse; secondary stresses ($Q_m$, $Q_b$: residual,
thermal) are self-equilibrating and do not. Both contribute to $K_I$. Their interaction with plasticity is
represented by $\rho$ (additive) or $V$ (multiplicative), chosen by the user (`interaction="rho"`, the default, or
`"V"`); the two do not give the same $K_r$. Define

$$ \chi = \frac{K_I^s\,L_r}{K_I^p}, $$

the load-independent ratio $K_I^s/(K_I^p/L_r)$, well conditioned at any non-zero primary load.

**Decision table** [register `fad.kr`]:

| Case | $K_r$ |
|---|---|
| $K_I^s < 0$ | $K_I^s$ set to 0 ($\rho = 0$, $V = 1$) |
| no secondary stress | $K_I^p/K_{mat}$ |
| primary and secondary, $\rho$ route | $(K_I^p + K_I^s)/K_{mat} + \rho(\chi, L_r)$ |
| primary and secondary, $V$ route | $(K_I^p + V(\chi, L_r)\,K_I^s)/K_{mat}$ |
| $\chi$ above the simplified route's preferred range (`critflaw.fad.kr.CHI_SIMPLIFIED_LIMIT`) | the simplified route is still applied (it is over-conservative there) and the result is flagged |
| secondary only ($P_m = P_b = 0$ exactly) | $K_J^s/K_{mat}$ with the plastic-zone-corrected $K_J^s$ |

A self-balancing secondary term $K_{sb}/K_{mat}$, if supplied, is added outside the interaction in every row. It is
available only in the low-level `critflaw.fad.kr.fracture_ratio` (argument `k_self_balancing`); `Assessment` does
not expose it.

**Simplified $\rho$** [register `fad.rho.simplified`; Ainsworth1986; `critflaw.fad.kr.rho_simplified`].
$\rho = \rho_1(\chi)$ at low $L_r$, tapers linearly to zero over a transition band of $L_r$, and is zero above
it. $\rho_1$ is a polynomial in $\chi$ up to a limiting $\chi$ and is held at a constant cap above.
**Simplified $V$** [register `fad.v.simplified`; AinsworthSmithWiesner1999; `critflaw.fad.kr.v_simplified`]. $V$ is
the smaller of a form increasing in $L_r$ and $\chi$ and a linear bound decreasing in $L_r$, below the same upper
$L_r$ limit, and $V = 1$ above it. Both are continuous at the boundary points that the source leaves unassigned;
those points are assigned to the upper branch without changing the value. The breakpoints and coefficient sets are
not reproduced here.

**Secondary stress acting alone** [R6]. $\rho$ and $V$ are undefined when there is no primary load. The secondary
K is corrected for the plastic zone:

$$ a_{eff} = a + \frac{1}{2\pi\beta_{pz}}\left(\frac{K_I^s}{\sigma_Y}\right)^2,\qquad K_J^s = \sqrt{a_{eff}/a}\;K_I^s \ \ge K_I^s, $$

with $\beta_{pz} = 1$ (plane stress, the default because it is more conservative) or 3 (plane strain); the choice
is recorded in the trace. The plane-stress default over-predicts $K_r$ by up to about 13–16 % relative to plane
strain for thick sections. The crack dimension $a$ is the $a$ of the K solution: the depth of a surface or extended
flaw at both positions, the half-length of a through flaw, the half-height of an embedded flaw.

**Vanishing versus absent primary load.** $K_r$ is discontinuous between "a vanishing primary load" and "no primary
load". This is a property of the procedure, and it is pinned by a test at $10^{-12}$ of nominal load.

**Negative primary K.** The bending magnification factor is negative at the deepest point of deep flaws, so a
tensile bending stress can give $K_I^p < 0$ there; the source gives no rule. The library sets $K_I^p = 0$
(assessing the crack as not closed by the primary load: conservative) and flags the result. Because a primary
load is present ($L_r > 0$) this is not the secondary-only row: $\chi$ is unbounded and the interaction takes its
limiting value ($\rho_1$ at its cap; $V$ on its linear bound). Primary stresses are tensile by validation and a negative
secondary K is zeroed, so $K_r \ge 0$ always.

### 1.5.1 Welding residual stress

`Residual` adds a uniform membrane secondary stress [registers `residual.as_welded`, `residual.pwht`; no open
primary reference identified; see `docs/unsourced_items.md`]:

- **as-welded**: $Q_m = \sigma_Y'$, the appropriate yield strength supplied according to flaw orientation (the
  lesser of weld and parent for a flaw parallel to the weld, the material containing the flaw for a transverse
  flaw, the greater of the two if it spans both), defaulting to the material yield strength; a room-temperature
  value may replace it in the unrelaxed term when the assessment temperature is below ambient;
- **relaxation with primary load** (explicit option, off by default): $Q_m$ is reduced below $\sigma_Y'$ by an
  amount that grows linearly with $\sigma_{ref}/\sigma_f'$ (the ratio clipped to a bounded interval), where
  $\sigma_f'$ is the flow strength of the appropriate material, and never exceeds $\sigma_Y'$; when on, $Q_m$ is
  recomputed inside the engine from the sampled strengths and $\sigma_{ref}$. The constants are not reproduced here
  (implementation: `critflaw.model.Residual`, register `residual.as_welded`);
- **post-weld heat treated**: a fraction of the appropriate yield strength depending on flaw orientation and the
  furnace temperature range, for the steels and furnace conditions the rule covers (register `residual.pwht`).

## 1.6 Acceptance: the direct test

A point is **acceptable** if and only if

$$ K_r < f(L_r) \quad\text{and}\quad L_r < L_{r,max} $$

at **every** assessed crack-front position. The inequalities are strict: a point on the line, on the cut-off or on
the step is unacceptable. The test is evaluated directly (`fad.margins.acceptable`); $f$ is evaluated only inside
the cut-off. This is the single authoritative failure indicator used by the deterministic engine, by every
probabilistic solver ([04_probabilistic.md](04_probabilistic.md) §4.3), by the time-dependent first-passage
definition ([05_fatigue.md](05_fatigue.md) §5.5) and by the emulator-based Pf ([07_pf_emulator.md](07_pf_emulator.md)
§7.1).

The failure set is not a series system of smooth components. With the line split into segments $i$ on $L_r$
intervals $I_i$ (a step location belonging to the post-step segment),

$$ \text{fail} = \bigcup_i \{L_r \in I_i,\ K_r \ge f_i(L_r)\} \ \cup\ \{L_r \ge L_{r,max}\}, $$

which is exactly the complement of the acceptable set. A step is not an event in itself. This structure is why
FORM is not applied to FAD problems ([04_probabilistic.md](04_probabilistic.md) §4.4.3).

## 1.7 The radial margin

The radial margin $F$ is the factor on the ray from the origin through $(L_r, K_r)$ at which the point first leaves
the acceptable region. Because every line is non-increasing with $f \le 1$, the acceptable set along the ray is an
interval $t < F$, so bisection on the acceptance predicate is valid even across a step. The implementation:

- brackets $F$ by $t_{hi} = \min(L_{r,max}/L_r,\ 1/K_r)$, rigorous because $f \le 1$ and $f = 0$ at the cut-off
  (no arbitrary cap, no widening);
- uses the exact values on the axes: $F = f(0)/K_r$ on the $K_r$ axis ($f(0) = 1$ for every line implemented),
  $F = L_{r,max}/L_r$ on the $L_r$ axis, $F = +\infty$ at the origin;
- evaluates $f$ only at trial points inside its domain;
- records whether the exit is through the curve, a step or the cut-off (see the glossary, [index.md](index.md) §3.1).

$F$ is accurate to a few ulp, so a point within rounding of the boundary can have $F = 1$ on either side.
**Acceptance is never taken from $F$.** In the probabilistic layer $F - 1$ serves only as a driving variable for
ordering samples (its sign is forced to agree with the direct indicator where they differ by rounding, and a larger
disagreement is an internal error).

The **governing position** of a sample is the crack-front position with the smallest $F$; where margins tie (for
example both positions leave through the cut-off) the position with the larger $K_r$ governs. It is reported per
sample; acceptance itself requires all positions to pass.

## 1.8 Reserve factors and limiting flaw size

Acceptability need not be monotone in a scanned parameter: residual-stress relaxation, the interaction term and
through-wall stress variation can each produce more than one crossing. The functions in `critflaw.reserve`
therefore **scan** a stated range on a stated grid (default 400 points), polish every change of acceptability by
bisection, report the smallest value at which the assessment is not acceptable, and say what they could not
resolve. They take a scalar assessment.

| Function | Scanned quantity | Grid |
|---|---|---|
| `reserve_factor_on_load` | factor on $P_m$ and $P_b$ (secondary stress held; $L_r$, $K_r$, $\rho$ and a relaxed $Q_m$ re-evaluated at each load) | log-spaced, $10^{-3}$ to 100 |
| `reserve_factor_on_toughness` | divisor on $K_{mat}$ | log-spaced, $10^{-3}$ to 1000 |
| `limiting_flaw_size` | surface flaw: depth $a$ at fixed $a/c$ up to $B$; through flaw: length $2a$ up to $W$ | linear, from $10^{-4}$ of the upper end |

The result (`Crossing`) states: the smallest not-acceptable value; whether one was found in range (`bounded`);
whether acceptability changed more than once (`unique`); every crossing found; whether the assessment was already
unacceptable at the bottom of the range (`below_range`, searched downward by decades); whether the value is where a
solution leaves its validity range rather than a predicted failure (`limited_by_validity`); and the scan actually
performed. For the load factor, exactly zero primary load is assessed first; if that is unacceptable the result is
0 with `unacceptable_at_zero` (no reserve on load exists, although positive loads may be acceptable because of the
discontinuity noted in §1.5).

**Limits.** Non-acceptable bands narrower than the grid spacing can be missed; the result states the resolution.
The property "radial margin equals load reserve factor without secondary stress" holds and is tested, but is not
counted as evidence of correctness of either.

## 1.9 The trace

Every `Result` carries a `Trace`: the source edition tag, the register ids of the equations used (with citations
"as cited by the source standard" until checked against the primary), the options chosen (line option, yielding,
interaction, bending restraint, plastic-zone correction, route choices, the Option 2 correction and cleaning notes),
the flags raised, the knowledge-base status of each equation ([08_knowledge_base.md](08_knowledge_base.md)) and the
uncertainty sources the result does not account for. `Assessment.run(strict=True)` refuses a result that used any
equation below `cross-checked`.

## 1.10 Assumptions and limits of this chapter

- The FAD lines are the generic Options 1 and 2. Option 3 exists only as an experimental emulator for one geometry
  ([06_fe_option3.md](06_fe_option3.md)).
- `ρ` and `V` are the simplified forms; above their preferred range of $\chi$ they are over-conservative and
  flagged. The alternative
  plasticity-interaction function based on $K_J^s$ with $h(L_r)$ is not implemented.
- The yielding behaviour is the user's declaration; a yield plateau is never detected from data.
- Option 2 requires a material that has strained plastically by $1.41\sigma_Y$ when the cut-off lies beyond
  $\sqrt2$.
- Equality on the line is failure.
