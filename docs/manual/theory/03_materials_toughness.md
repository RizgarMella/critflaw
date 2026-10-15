# 3. Materials: fracture toughness and tensile laws

This chapter covers the material inputs of the FAD: the fracture toughness $K_{mat}$ (from fracture-mechanics test
results, from the Master Curve, or from Charpy correlations) and the tensile description used by the assessment
line (yield and tensile strength for Option 1, a stress–strain law for Option 2 and for the FE pipeline).
Implementation: `critflaw.toughness`, `critflaw.model` (`Material`, `MasterCurve`, `Ratio`), `critflaw.fad.option2`
(`PowerLaw`, `TabulatedLaw`), `critflaw.fe.warp_law` (`WarpDeformationLaw`).

Public functions take and return SI units (Pa, m, Pa√m, J/m², J, °C). The correlations are written in the units of
their sources (MPa√m, mm, J) and converted inside each function.

---

## 3.1 The `Material` record

`Material(yield_strength, uts, modulus, toughness, yielding, fad_option=1, luders_strain=None, law=None)`:

- `yielding` is required ([01_fad.md](01_fad.md) §1.3);
- `uts` may be a `Ratio(value)` of the yield strength with `value > 1` (support above 1 when random), which
  guarantees $\sigma_U > \sigma_Y$ in every realisation; with independent distributions for both strengths the
  library requires their supports not to overlap and otherwise refuses the problem;
- `toughness` is $K_{mat}$ in Pa√m (a number, array or distribution) or a `MasterCurve` record (§3.2);
- `fad_option=2` requires a `law`; `yielding` must then be `"discontinuous"` exactly when the law has a declared
  plateau, and the Lüders strain belongs to the law.

Every positive quantity that is random must have a distribution whose support is positive; an unbounded
distribution must be truncated explicitly, e.g. `Truncated(Normal(...), lo=0)`. Nothing is clipped silently.

## 3.2 Master Curve

[register `toughness.master_curve`, primary reference: Wallin and Rintamaa [WallinRintamaa1997]]

For ferritic steels in the lower shelf and transition, the toughness at cumulative probability $p$ follows a
three-parameter Weibull distribution with a fixed threshold and fixed shape, whose scale grows exponentially with
$T - T_0$ ($T$ the temperature, $T_0$ the reference temperature, °C) and is scaled to the crack-front length $l$ by
a fourth-root size law relative to a reference length; $l$ is capped at twice the section thickness. The constants
are not reproduced here (implementation: `critflaw.toughness.master_curve`). The distribution's threshold is a
lower bound on $K_{mat}$, which is shown in results.

**In the engine** a `MasterCurve(t0, temperature, probability)` record is resolved inside the assessment: the
cumulative probability is normally a `Uniform(0, 1)` random variable, and $K_{mat}$ is computed from the crack-front
length of the flaw being assessed ($2c$ surface, $4c$ embedded, $2B$ through-thickness and extended, capped at $2B$;
[02_solutions.md](02_solutions.md) §2.6). In fatigue the length follows the growing flaw. A sampled probability that
rounds to exactly 1 is moved to the largest double below 1. A random $T_0$ and a temperature margin are not combined.

**Exponent coefficient.** Two printed forms of the exponent coefficient differ; the library adopted one of them
(the reasoning is in `docs/unsourced_items.md` §5).

## 3.3 Charpy correlations

All are for ferritic steels and are labelled as estimates in their docstrings and register descriptions. They are
standalone functions returning arrays; they do not enter an `Assessment` trace. None applies to ductile (upper-shelf
tearing) behaviour except the upper-shelf correlation.

| Function | What it gives | Register / primary reference |
|---|---|---|
| `t0_from_charpy(T, energy=27 or 40)` | $T_0$ as a fixed offset below the 27 J (or 40 J) Charpy transition temperature; the correlation's standard deviation is exported as `T0_CHARPY_STANDARD_DEVIATION` | `toughness.charpy_t0` [Wallin1989] |
| `charpy_master_curve(T, T0, B, probability=0.05, temperature_margin=25)` | the Master Curve form with the section thickness $B$ as the length and a temperature margin $T_K$ (°C) for the scatter of the correlation | `toughness.charpy_master_curve` [Wallin1989] |
| `charpy_lower_bound(Cv, B)` | a lower-bound estimate, increasing with $\sqrt{C_v}$ and scaled to the thickness by a fourth-root size law, from the Charpy energy at the service temperature (lower shelf and transition); a non-positive result is refused | `toughness.charpy_lower_bound` [INSTA1991] |
| `charpy_upper_limit(Cv, σY)` | a linear cap in $C_v$ on Charpy-based estimates; refused for yield strength at or above 480 MPa | `toughness.charpy_upper_limit` — no open primary reference identified; see `docs/unsourced_items.md` |
| `charpy_upper_shelf(Cv_us, E)` | initiation toughness at 0.2 mm tearing from the upper-shelf energy, through a J correlation and $K = \sqrt{EJ/(1-\nu^2)}$ | `toughness.charpy_upper_shelf` [LuconEtAl2005] |
| `subsize_charpy_temperature_shift(B_ss)` | upward shift of the transition temperature for sub-size specimens of thickness in (0.625, 10] mm | `toughness.charpy_subsize` — no open primary reference identified; see `docs/unsourced_items.md` |

Coefficient values are in the code with their register entries.

## 3.4 Toughness from test results

**From J** [register `toughness.k_from_j`; no open primary reference identified]:

$$ K_{mat} = \sqrt{\frac{E\,J_{mat}}{1-\nu^2}}. $$

**From CTOD** [register `toughness.k_from_ctod`; no open primary reference identified]:

$$ K_{mat} = \sqrt{\frac{m\,\sigma_Y\,\delta\,E}{1-\nu^2}}, $$

with the constraint factor $m$ a power-law correlation in $\sigma_Y/\sigma_U$ for $0.3 < \sigma_Y/\sigma_U < 0.98$
and a constant outside. The yield and tensile strengths here are those **of the tested material at the test
temperature**, separate arguments from the (possibly sampled) assessment properties; for steels and deeply notched
high-constraint specimens.

**Sub-size specimens** [register `toughness.subsize`; no open primary reference identified]. For cleavage results
from specimens thinner than the component, `subsize_correction` scales the toughness above the Master Curve
threshold by a fourth-root law in the specimen-to-component thickness ratio (constants not reproduced here). The
caller must have checked limited prior ductile extension and that $K_b$ is below the small-scale-yielding limit
computed by `k_jc_limit` from the ligament, the test yield strength and the modulus; a value above it is replaced by
the limit.

**Minimum of three equivalent** [register `toughness.mote`; no open primary reference identified].
`minimum_of_three_equivalent` takes 3 to 15 results and returns the lowest, second lowest or third lowest according
to the number of results (the mapping is in the function, not reproduced here).

**Lower-bound statistic** [register `toughness.lower_bound`; no open primary reference identified]. From $n$ results
with mean $\bar K$ and standard deviation $S$ (divisor $n-1$), the bound $\bar K - kS$ lies below a stated population
quantile with a stated confidence (both settable; defaults in the signature of `tolerance_factor`). The tolerance factor is computed **exactly** from the
non-central t distribution,

$$ k = \frac{t'_{\,n-1,\ \delta}(\text{confidence})}{\sqrt n},\qquad \delta = z_{1-\text{percentile}}\sqrt n, $$

rather than read from a table: this reproduces the source's tabulated factors for 3 to 20 results to their printed
precision except one entry, which is evidently a misprint (`docs/unsourced_items.md` §5). `lognormal=True` applies the same to
$\ln K$, which is preferred for cleavage data in the transition (a normal fit can give very low or negative values;
a non-positive normal bound is refused).

## 3.5 Tensile properties for Option 1

Option 1 uses $\sigma_Y$, $\sigma_U$ and $E$ only, plus the Lüders strain for discontinuous yielding (supplied, or
estimated from $\sigma_Y$ below a yield strength limit; [01_fad.md](01_fad.md) §1.3). In probabilistic runs every realisation forms
its own line and cut-off.

## 3.6 Stress–strain laws for Option 2

All laws expose the plastic true strain through the secant compliance $q(s) = \varepsilon_p(s)/s$ with
$s = \sigma/\sigma_Y$, its running maximum $\hat q$, their discontinuities and their domain end
([01_fad.md](01_fad.md) §1.4).

### 3.6.1 Power law

`PowerLaw(offset, n, luders_strain=None)`: $\varepsilon_p(s) = \text{offset}\cdot s^n$, plus a step equal to the
Lüders strain at $s = 1$ when declared. `offset` must be chosen explicitly (0.002 when $\sigma_Y$ is a 0.2 % proof
strength, 0 when it is a lower yield point) and $n \ge 1$. The hardening exponent `n` (and the offset) is a fixed,
user-supplied value of the law: it cannot be a random variable, and the engine never re-derives it from sampled
strengths. With random tensile properties the plastic-strain shape is held fixed and only $\kappa = E/\sigma_Y$
varies ([01_fad.md](01_fad.md) §1.4). A helper, `hardening_exponent_lower_bound(yield_strength, uts)`, returns a
lower-bound estimate of the exponent from $\sigma_Y/\sigma_U$ that the user may call to choose `n`; it is stated for
ferritic steels with $0.65 < \sigma_Y/\sigma_U < 0.95$ and yield 300–1000 MPa (the function does not check this
range) [register `material.power_law`; no open primary reference identified; see `docs/unsourced_items.md`].

### 3.6.2 Tabulated law from a measured curve

`TabulatedLaw.from_engineering(stress, strain, yield_strength, modulus, yielding, luders_strain=None)` (or
`.from_true`) cleans the data in a fixed order and records every step in the law's `notes`, which the trace prints:

1. **Sort by strain.**
2. **Yield plateau: declared, never detected.** `yielding` is required: `"continuous"` uses the data as they are;
   `"plateau"` requires the Lüders strain, removes the points whose engineering strain lies inside the declared
   plateau (an error if any differs from $\sigma_Y$ by more than 5 %) and represents the plateau as a step, not as
   the ramp true-stress conversion would make of it. A plateau cannot be told from low hardening by inspecting data;
   each detection error moves the line at $L_r = 1$ by a factor of two or more, so no detector is used. A plateau
   record declared continuous gives a line that is too high near $L_r = 1$.
3. **Monotone stress.** Stress is made non-decreasing in strain by isotonic least-squares regression (pool adjacent
   violators), which averages measurement noise where a running maximum would ratchet upward. An adjustment above
   5 % of $\sigma_Y$ is an error (a load drop, not noise). Of a run of points at equal stress both ends are kept, so
   the line steps there; coarse stress quantisation therefore gives a stepped line around the unrounded one.
4. **True stress and strain;** $\varepsilon_p = \varepsilon - \sigma/E$.
5. **Elastic range.** Plastic strain below 2 % of the yield strain is set to zero up to the last such point (it is
   noise level; this can raise the line by up to about 0.012–0.02 below $L_r = 0.75$ for clean data). Plastic strain
   below $-0.25$ of the yield strain is an error (the modulus is too low). The data must include a point with no
   plastic strain. $\varepsilon_p$ is then made non-decreasing in $s$.
6. **Declared plateau step** at $s_{step} = \min(1, \text{mean plateau stress}/\sigma_Y)$; data after the step are
   kept as measured.
7. **Coverage.** A note is raised if no point lies in some interval of the minimum evaluation set
   ($L_r$ = 0.7, 0.9, 0.98, 1.0, 1.02, 1.1); the data must extend beyond $\sigma_Y$.

$\varepsilon_p(s)$ is linear between points; the law ends at its last point (exit `domain_end` when that comes
before the collapse cut-off). When the yield strength is sampled the plastic-strain shape $\varepsilon_p(s)$ is held
fixed, which preserves the offset definition of yield.

### 3.6.3 The WARP3D deformation law

`WarpDeformationLaw(E, nu, sigma_0, n)` is the uniaxial law of the deformation-plasticity material used by the FE
pipeline ([06_fe_option3.md](06_fe_option3.md) §6.1.2), implemented so that Option 2 and Option 3 describe one material.
In the normalised plane $(x, y) = (\varepsilon/\varepsilon_0,\ \sigma/\sigma_0)$ with $\varepsilon_0 = \sigma_0/E$:

- linear, $x = y$, up to $y = K_1 = 0.95$;
- a circular arc, tangent to the linear branch at $(K_1, K_1)$ and at first contact with the power branch; its end
  $K_2$, centre and radius are derived from $n$ (unique for $n$ in 3–20, with $K_2 \approx$ 1.003–1.023);
- the pure power law $x = y^n$ for $y \ge K_2$.

The arc construction is the one detail the WARP3D manual does not fully specify (it refers to a 1991 MIT thesis by
Wang, which has not been obtained). The measured difference between this construction and WARP3D's one-element
response is the recorded bound: at most 0.14 % (at $s = 1.0$) in the arc, and within the output print resolution
($5.7\times10^{-4}$ relative, against a band of $7\times10^{-4}$) on the linear and power branches
([06_fe_option3.md](06_fe_option3.md) §6.2, D6).

**Meaning of $\sigma_0$.** $\sigma_0$ is the law's reference stress (where the power branch meets the elastic line),
**not** a 0.2 % proof stress. The implied proof stress (`proof_stress_0p2()`) is 1.02–1.52 $\sigma_0$ over the FE
parameter box and is reported with every FE result; `sigma0_from_proof_stress` inverts it for users who specify
$R_{p0.2}$ (using $R_{p0.2}$ in place of $\sigma_0$ would shift the $L_r$ axis by 4–52 % over the box). The law has
no tensile strength: the user's `uts` sets $L_{r,max}$ and nothing else; the Considère stress implied by the power
branch, $s^n = 1/(n\varepsilon_0)$, is reported for comparison and never used. For Option 2,
$q(s) = \varepsilon_0(x(s)/s - 1)$, which is non-decreasing, so $\hat q = q$; the curve is built with
`Option2Curve.from_warp_law(law, uts)`, which takes $E$ and $\sigma_Y := \sigma_0$ from the law and refuses a
mismatch. A Ramberg–Osgood power law or a measured curve given to Option 3 is refused (no exact mapping exists: the
Ramberg–Osgood law has no linear range).

## 3.7 Assumptions and limits of this chapter

- The Master Curve and Charpy correlations are for ferritic steels; validity statements (temperature range,
  strength) are enforced where the code can check them (yield limits) and otherwise printed.
- Charpy-based toughness carries the scatter of the correlation; the temperature margin and a random $T_0$ are not
  to be combined.
- Option 2 tensile data must include the elastic range and extend beyond yield; noise sensitivity is a property of
  Option 2 ([01_fad.md](01_fad.md) §1.4).
- Constraint effects on toughness (beyond the CTOD constraint factor and the crack-front length of the Master Curve)
  are not modelled.
