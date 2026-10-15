# 6. The finite element pipeline and the experimental Option 3 emulator

Option 3 builds the assessment line of a specific geometry from elastic-plastic J-integral analyses. `critflaw`
provides (i) a WARP3D-based FE pipeline for a semi-elliptical surface flaw in a flat plate under proportional tension
and bending, with layered quality assurance and a record of its verification evidence, (ii) extraction of the
deepest-point Option 3 curve from one FE case, and (iii) an **experimental** Gaussian-process emulator of that curve
over a five-dimensional input box. Failure probability with the emulator is in [07_pf_emulator.md](07_pf_emulator.md).
Implementation: `critflaw.fe` (`fecase`, `crack_mesh`, `mesh`, `deck`, `runner`, `parser`, `elastic_case`,
`plastic_case`, `convergence`, `warp_law`, `units`) and `critflaw.option3` (`curve`, `design`, `gp`, `prior`,
`emulator`, `validation`, `interface`).

**Status in one paragraph.** The FE pipeline is complete for its one geometry family and every case passes or fails a
named set of QA checks. The family verification is **incomplete**: two published elastic-plastic benchmarks (D4,
D5) are blocked on references that have not been obtained, and the formal mesh-convergence criterion (D7) is
unresolved for the family. No FE result is therefore labelled `fe-verified` (`FAMILY_VERIFIED = False`). The emulator
is trained on these unverified data; every emulator record and prediction carries `verification_status="unverified"`
and `assessment_grade=False`, applies to the deepest point only, and is refused for assessments and Pf unless the
caller passes `experimental=True`, in which case the result is labelled experimental. It misses its RMSE target
(leave-one-curve-out 0.0241 against 0.02), its leave-one-curve-out coverage is below nominal (0.935 against 0.95),
and the held-out checks it passes rest on two curves (§6.5).

---

## 6.1 The FE pipeline

### 6.1.1 Solver and build

WARP3D Release 18.5.0 (build 5120, a pinned source commit), built in Docker with the Intel ifx compiler and MKL. The
build identity (source commit, compiler and MKL versions, build flags), the image digest and the manual version the
commands were checked against (the 18.5.0 User Manual, April 2026 update [WARP3D manual]) are recorded with every
run. Every command the deck writer emits is cited to a manual section in `docs/manual/warp3d_commands.md`; a test tokenises every
emitted deck and asserts that each command keyword appears there. The Dockerfile is shipped; the image is not
redistributed. One job runs at a time on 4 threads with a declared memory ceiling and a wall-clock timeout.

### 6.1.2 The case record

`FECase(a, c, B, W, L_far, E, nu, sigma_0, n, beta, level, phi_s_deg=10, n_steps=12, ...)` is an immutable record with
a canonical JSON form and a case id (SHA-256 prefix). It enforces at construction:

| Parameter | Box |
|---|---|
| hardening exponent $n$ | [3, 20] |
| $\sigma_0/E$ | [$10^{-3}$, $5\times10^{-3}$] |
| $\nu$ | [0.25, 0.35] |
| loading mix $\beta = P_b/(P_m + P_b)$ | [0, 1] (proportional loading; $\beta = 1$ pure bending) |
| $a/B$ | [0.2, 0.6] |
| $a/c$ | [0.4, 1.0] |
| $c/W$ | ≤ 0.1 |
| remote length $L_{far}$ | ≥ max(20 B, 2 W) |
| mesh level | one of the four declared levels |

The box is an algorithmic range, not a claim of physical validity; the strain caps of §6.1.5 bound the physics. Other
geometry families raise `NotImplementedError("TODO(verify): ...")`.

**Material: exactly one family**, WARP3D's `deformation` (deformation-plasticity, small-strain) law: linear to
$0.95\sigma_0$, a tangent circular transition arc, then $\varepsilon/\varepsilon_0 = (\sigma/\sigma_0)^n$; the library's
`WarpDeformationLaw` implements the same uniaxial law ([03_materials_toughness.md](03_materials_toughness.md) §3.6.3), so
the Option 2 and Option 3 curves describe one material. $\sigma_0$ is the law's reference stress, not a proof stress,
and is the library's $\sigma_Y$ for this family. Ramberg–Osgood laws and measured curves are refused for Option 3.

### 6.1.3 Mesh

A purpose-built parametric crack block of 20-node bricks (no external mesher; the plate is a box), a pure function of
the case (node and element numbering deterministic; a topology hash recorded):

- a tube around the elliptical front of radius $r_t = \min(0.5\,a^2/c,\ 0.5(B - a),\ 0.1\,a)$, constant along the front,
  with rings graded $r_i = r_t(i/n)^{1.6}$ and a blend radius $1.5\,r_t$;
- collapsed wedge tip elements: for the elastic mesh tied, with quarter-point mid-side nodes (checked to $10^{-12}$);
  for the plastic mesh untied, in the keyhole-collapsed-to-a-point pattern (one distinct node per angular position of
  the collapsed edge, shared by the two adjacent elements, so no element repeats a node), with the crack-plane
  constraint on the ligament-side tip node only so that the others can blunt;
- front nodes at fixed parametric angles, with a node exactly at the near-surface evaluation angle $\varphi_s$
  (parametric, Newman–Raju convention; default 10°, physical angle $\arctan((a/c)\tan\varphi_s)$, 4.0° at
  $a/c = 0.4$) on every level, so that comparisons across levels refer to the same physical point; the free-surface
  vertex $\varphi = 0$ is excluded from every criterion;
- a radial blend to a square (hence the number of sectors is a multiple of 4), an O-grid core, a mapped outer block,
  a structured slab and a graded far field whose first layer equals the blend thickness and grows by at most 1.4;
- **levels** as integer vectors (rings, sectors, front divisions, front divisions to $\varphi_s$, blend layers):
  `level_1` = (6, 12, 12, 3, 3) is the production mesh; `level_2` = (8, 16, 16, 4, 4); `level_3` = (10, 20, 20, 5, 5);
  `level_4` = (12, 24, 24, 6, 6); exact refinement ratios 4/3, 5/4, 6/5 in every entry. At the mid-box point on the
  converged far field the first three levels have 34,029 / 56,258 / 85,176 nodes.

### 6.1.4 Loading and the J-integral

**Loading** is proportional: one unit pattern of consistent nodal loads for remote tension and bending
($P_m + P_b = 1$ MPa at the declared mix, with the linear profile $P_m + P_b(1 - 2y/B)$ through the thickness), scaled
by a declared cumulative load list. The parser reads the cumulative factors back from the output and checks them
against the declared list.

**Admissibility.** Before any nonlinear run an elastic case on the tied mesh of the same level must give a signed
$K_I > 0$ at every interior front station (no crack-face contact); otherwise the nonlinear run is refused.

**Elastic K.** Signed $K_I, K_{II}, K_{III}$ and $T$ from WARP3D's interaction integrals, always with the plane-strain
auxiliary field. Plane-strain $E' = E/(1-\nu^2)$ is used wherever a J-to-K conversion is reported (information only).

**J.** Domain integrals at every corner station of the front after every step; J at a station is the mean over rings
2..n (ring 1, adjacent to the tip, is excluded); the spread over those rings is the path-independence measure. The
deepest-point domain uses q-function type a; the free-surface station type c with its nodes listed so that the vertex
is last. Plastic decks request WARP3D's binary packets file so that J per ring is read in full double precision (the
printed tables carry four significant digits, which is the size of level-to-level differences); the printed tables are
checked against the packets to $10^{-3}$.

**Elastic J for $K_r$.** $J_e$ comes from the **same** plastic mesh, by scaling the first, essentially elastic, step:

$$ J_e(\lambda) = J_1\left(\frac{\lambda}{\lambda_1}\right)^2, $$

subject to two QA conditions: (a) every Gauss point with $\sigma_{vm} > 0.95\sigma_0$ after step 1 lies in the first
ring; (b) $J_1$ agrees within 0.5 % with an elastic run on the same untied mesh (a large-yield material, so the band
measures the plastic-zone contribution only).

**$L_r$ from one declared limit load.** $L_r$ for both the FE curve and the assessment point comes from the plate
surface-flaw reference stress of [02_solutions.md](02_solutions.md) §2.5.1 divided by $\sigma_0$; a limit load is
never read from the reactions of a strain-hardening run. Then, at each step,

$$ L_r = \lambda\,\frac{\sigma_{ref}(\text{unit pattern})}{\sigma_0},\qquad K_r = \left(\frac{J_e}{J}\right)^{1/2}. $$

### 6.1.5 Load schedule and the small-strain caps

The deformation law is small-strain. Two caps form a kinematic-error budget (the small-strain error at strain $e$ is of
order $e/2$), consistent with the 3 % band of the verification:

- **schedule end**: the smaller of $L_r = 1.5$ and the $L_r$ at which the reference strain $\varepsilon_0 L_r^n$
  reaches 2 %. Over the box this ends the curve between $L_r$ = 1.07 (at $n = 20$, $\sigma_0/E = 5\times10^{-3}$) and
  1.5;
- **local cap**: the maximum effective strain over the elements whose centroid is at least $5r_t$ from the front
  (the far region; with $r_t = 0.1a$ this excludes $0.5a$ around the front). Once it exceeds 5 % the step and every
  later step are marked `outside_small_strain`; they are stored but never used for a curve. The strains are output in
  element chunks at three steps (the first step with $L_r \ge 0.8$, the last with $L_r \le 1.2$, the last step);
  because strains are monotone in the load under proportional loading of a deformation-plasticity solid, steps between
  checked steps inherit the decision conservatively.

The schedule is 12 steps: geometric from $L_r = 0.05$ (or half the end, if smaller) to 0.8, then linear to the end.

### 6.1.6 Units, run handling and caching

The case is SI; the deck is written in N, mm, MPa (a Pa deck was mis-read by WARP3D's number scanner, and its
fixed-width prints overflow for Pa-sized stresses), and every parsed quantity is converted back
($1\ \text{MPa}\sqrt{\text{mm}} = 10^6\sqrt{10^{-3}}\ \text{Pa}\sqrt{\text{m}}$; $1\ \text{N/mm} = 10^3\ \text{J/m}^2$).

A run is `complete` only with a zero exit, the normal-termination marker, all declared steps completed and convergence
at every step; a step that WARP3D's adaptive driver completes through sub-increments is converged and recorded
`adaptive`. A non-converging step ends the job (`truncated`); unparsable output is `failed`; a killed container is
`timeout`. Infrastructure failures (container killed for memory) are not recorded as case results.

The **solve cache** key is the SHA-256 of every generated solver input, the entry deck, the executable, the thread
count, the memory ceiling, the timeout and the image digest: a rebuilt image is always a new key. The canonical case
JSON, library, mesher and template versions are searchable metadata, so a parser or QA change re-parses stored raw
output without re-solving. Failed, truncated and timed-out runs are stored and never re-run silently (an explicit
retry policy keeps earlier attempts). The store is append-only.

### 6.1.7 QA layers

Every check is a named value with a band and a pass flag; a case failing any check is `fe-failed` with the first failing
check named. No automatic re-meshing.

1. **Mesh**: positive Jacobian at every 2×2×2 Gauss point of every element, including the collapsed tips; scaled
   Jacobian ≥ 0.2 for ordinary bricks outside the tube and over rings 2..n; tip-element count and coincident-node
   pattern confined to the front; quarter-point positions (elastic) and the untied pattern (plastic); aspect-ratio
   extremes reported (thin footprint layers up to about 90 and far-field columns up to about 600 are reported, not
   banded: the J domains never reach them); domain extents not reaching a model boundary.
2. **Run**: convergence at every step; steps completed equal declared; warnings against a deny-list; load factors
   read back equal the declared list; equilibrium — the force and moment resultants of the applied nodal loads are
   formed exactly in Python and the parsed reactions balance them (residuals within $10^{-4}$ on dimensional scales).
3. **Mechanics**: J path independence within 2 % over rings 2..n at every position and step; $K_I > 0$; the $J_e$
   conditions (a) and (b); the far-region strain; the remote stress profile within 0.5 % at half the remote length
   (a Saint-Venant check of $L_{far}$).

A case that passes every check is `unverified` while the family verification is incomplete, and would become
`fe-verified` only when the family verification is complete: D4, D5, D6, D7 and D8 passed for the family and the D3
deep-flaw / bending band resolved (§6.2, §6.6).

## 6.2 Verification evidence and limits

The verification plan separates checks of the pipeline from checks of the mechanics. A K solution is never marked
`fe-verified` by an FE pipeline that was itself validated only against that solution
([08_knowledge_base.md](08_knowledge_base.md) §8.5).

| ID | What | Reference | Result |
|---|---|---|---|
| D1 | parser on a single-edge-notch bend regression deck | the deck's stored WARP3D J | reproduced to the four printed significant digits |
| D2 | path independence, corner-crack deck | rings 2–8 | ≤ 0.25 % for the thirteen interior domains at every step; the two domains at each end of the front (rings reaching the boundary) reach 0.6–1.6 % late in the run (reported, not banded) |
| — | deck regeneration | the shipped deck on the same image | worst relative J difference $9.4\times10^{-4}$ over 12 steps × 17 domains (band $1.5\times10^{-3}$; J printed to four digits) |
| D3 | elastic $K_I$ of the mesher, 34-case design | the plate surface-flaw solution at the same parametric angle ([02_solutions.md](02_solutions.md) §2.3.1) | 25 pass (within 3 %), 8 unresolved (outside 3 % but within the fit's documented 6 %), 1 fail; see below |
| D3b | elastic $K_I$ against an independent 8-node surface-crack model from the WARP3D verification suite (test87), $a/B = 0.4$, $a/c = 0.4$ | its plane-strain interaction-integral $K_I$, with the crack-face traction term added to its domains | the mesher is 1.3–2.3 % above at every interior station except one (+3.1 % at 3.33°), within 3 % at $\varphi_s$ and the deepest point; the reference states that its values lie 1–3 % below 20-node results |
| D4 | fully plastic $h_1$ reproduction (plane-strain SE(T), CC(T)) | EPRI NP-1931 (Kumar, German and Shih, 1981) | **blocked**: the reference has not been obtained |
| D5 | published 3-D surface-crack J in a plate | candidate: Lei, IJPVP 81 (2004) 21–30 (tension) and 31–41 (bending) | **blocked**: not obtained |
| D6 | constitutive response, one-element uniaxial tension | `WarpDeformationLaw` in Python | linear and power branches within the print resolution ($5.7\times10^{-4}$ against $7\times10^{-4}$); arc residual 0.14 % at $s = 1.0$ (band 0.5 %) |
| D7 | nonlinear mesh convergence | Richardson/GCI with observed order | see below; **unresolved for the family** |
| D7b | far-field refinement at the mid-box point | the production level | pass: doubling the far divisions changes J by at most 0.449 % (band 0.5 %; ≤ 0.17 % at the deepest point); doubling $L_{far}$ to 40 B by 0.003 % |
| D8 | load application and boundary conditions | exact resultants; closed-form remote profile | reactions balance to residuals ≤ $10^{-5}$ at every step; remote profile within 0.5 % |
| D9 | solver-tolerance sensitivity | a 10× tighter Newton tolerance on levels 2 and 3 | J changes ≤ $1.3\times10^{-5}$ relative, while level-to-level differences are ≥ 38× larger (mostly ≥ 100×) |

**D3 detail.** Ring spread of $K_I$ ≤ 0.11 % in every case. 25 tension cases at $c/W = 0.05$: 22 pass, 3
unresolved, all at $a/B = 0.6$, worst at the deepest point (−3.0 to −3.8 %). Four tension corners at $c/W = 0.1$: 3
pass, 1 unresolved (deepest −3.4 %). Five bending cases: 4 unresolved (worst 4.0–5.3 % at near-surface stations), 1
fail ($a/B = 0.6$, $a/c = 1.0$, deepest point −7.6 % against the fit; the bending K there is 0.12 of
$\sigma_b\sqrt{\pi a}$, so the absolute difference is 0.9 % of that scale). The deepest-point deviation is
systematically negative and grows with $a/B$ (−2.2 % at 0.2 to −3.0/−3.8 % at 0.6); it is mesh converged (0.05 % then
0.01 % over three levels) and grew by about 1 % when the remote length went from 5 B to 20 B, which points to the
plate length and end condition of the comparison basis. Attribution needs the primary reference's tabulated values
and stated end conditions; until then the band stays unresolved and the one case failed, as the decision rule says.

**D7 detail (mid-box point: $a/B = 0.4$, $a/c = 0.4$, $n = 10$, $\sigma_0/E = 1.7\times10^{-3}$, tension; three
levels).** Under $J(h) = J^* + Ch^p$ with $d_{12} = J_1 - J_2$, $d_{23} = J_2 - J_3$, the observed order solves

$$ \frac{d_{12}}{d_{23}} = \frac{r_{21}^p\,(1 - r_{21}^{-p})}{1 - r_{32}^{-p}} $$

(non-constant ratios, ASME V&V 20 / Celik et al. 2008), and the production-level error is the coarse-grid GCI
estimate $1.25\,|J_1 - J_2|\,r_{21}^p/(r_{21}^p - 1)$ relative to $J_2$. The asymptotic test requires (a) monotone
differences, (b) $p \in [0.8, 3]$, (c) the fourth level predicted within 20 % and (d) $p$ from levels 2–4 within 0.5
of $p$ from levels 1–3. At the **deepest point** the sequence is monotone at every step with $p$ in [0.89, 1.82] for
$0.8 \le L_r \le 1.2$ and a level-1 GCI error of 0.22–0.29 %; adding the one-sided far-field shift (+0.09 to +0.17 %),
**the production error at the deepest point is 0.34–0.42 % for $0.8 \le L_r \le 1.2$** (band 3 %). The strain-capped
last step ($L_r = 1.28$) converges slowly ($p = 0.22$) and is not accepted. At **$\varphi_s$** the differences give
$d_{12}/d_{23}$ below the model's infimum (order undefined, $p \to 0$): the near-surface station converges slowly
(the free-surface intersection), with a three-level spread of 0.04–0.82 % for $L_r \le 1.2$ and 1.3 % at the last step;
it is labelled **"unresolved: mesh-sensitive"** and excluded from any verified or training use. The fourth level
does not fit in the workstation's memory on the converged far field (123k nodes, about 8 GB working set, a projected
2.3 h with paging), so clauses (c) and (d) are not evaluable on this hardware and D7 cannot pass as defined.
Production-level and level-2 runs also exist at the box corner ($n = 20$, $\sigma_0/E = 5\times10^{-3}$; used as
held-out validation cases in §6.5); the corner-point convergence evidence is not tabulated here. D7 remains unresolved
for the family.

**What the evidence supports.** The pipeline (writer, runner, parser, units, loads and boundary conditions, constitutive
law, path independence, solver tolerance) is checked. Elastic K agrees with an independent mesh and, mostly, with the
handbook solution. The deepest-point J at the mid-box point is converged to about 0.4 % up to $L_r = 1.2$, which is
provisional evidence, not a family-wide statement. Not established: agreement with published elastic-plastic J
(D4, D5), convergence over the whole box, the near-surface station, and the deep-flaw/bending K band.

## 6.3 The Option 3 curve of one case

`option3.curve.from_summary` turns a complete, QA-passing FE result into the deepest-point curve
`Option3Curve(x, level, case_id, lr, j_over_je)` with $x = (a/B,\ a/c,\ n,\ \sigma_0/E,\ \beta)$:

- **coordinate**: the FE load steps $k = 1..12$; $J/J_e = 1$ at step 1 by the definition of $J_e$;
- **support**: the prefix of steps with $k \le 11$ (the strain-capped last step was not accepted) **and** $L_r \le 1.2$
  (the range of the accepted convergence evidence) **and** inside the small-strain cap; steps beyond are absent, never
  filled; at least two steps are required;
- **QA**: $J/J_e$ must not decrease between supported steps by more than $2\times10^{-4}$ (relative); smaller dips are
  solver noise in the elastic range and are removed by a running maximum; larger ones fail the case;
- $K_r = (J/J_e)^{-1/2}$.

## 6.4 The emulator

### 6.4.1 Inputs and validity

$x = (a/B, a/c, n, \sigma_0/E, \beta)$ over $a/B \in [0.2, 0.6]$, $a/c \in [0.4, 1.0]$, $n \in [3, 20]$,
$\sigma_0/E \in [10^{-3}, 5\times10^{-3}]$, $\beta \in [0, 1]$. Declared validity bounds that are not inputs: $\nu = 0.3$
and $c/W \le 0.05$ (trained at 0.05). $L_r$ is referred to $\sigma_0$; a user's 0.2 % proof stress is converted with
`sigma0_from_proof_stress` ([03_materials_toughness.md](03_materials_toughness.md) §3.6.3). $L_{r,max}$ is computed from
the user's tensile strength and sampled properties, never emulated; the curve is used up to
$\min(L_{r,end}(x), L_{r,max})$. The schedule $L_r$ of the steps is a closed-form function of $x$, so the emulator
works at step index $k$ without interpolation.

### 6.4.2 The emulated quantity: log increments

$$ y_k = \log\left(\frac{J}{J_e}(k) - \frac{J}{J_e}(k-1)\right),\quad k \ge 2;\qquad \frac{J}{J_e}(k) = 1 + \sum_{i=2}^{k} e^{y_i},\qquad K_r(k) = \left(\frac{J}{J_e}(k)\right)^{-1/2}. $$

Every function draw therefore has $K_r \le 1$ and $K_r$ non-increasing in $L_r$ **by construction**; no draw is
rejected to impose a constraint. Elastic-range increments below $10^{-5}$ are set to that floor before the logarithm
(declared; it limits the resolution of $K_r$ near 1 to about $10^{-4}$).

### 6.4.3 Option 2 physics prior

The GP models the FE log increments **minus** those of the Option 2 curve of the same `WarpDeformationLaw`
([01_fad.md](01_fad.md) §1.4), with $R_0 = K_{r,\text{Opt2}}^{-2}$ normalised to 1 at step 1:

$$ y_k^{GP} = y_k - \log\max(\Delta R_{0,k},\ 10^{-5}). $$

The prior removes the dominant material dependence ($n$ and $\sigma_0/E$) and leaves geometry and constraint effects to
the GP. It changes the mean only; the constraints by construction are unchanged. It was adopted by leave-one-curve-out
model selection (Kr RMSE 0.024 → 0.019 on the first 17 curves; 0.024 against 0.026 without it on 20). Logarithmic
$n$ and $\sigma_0/E$ inputs were tried and rejected (0.0255).

### 6.4.4 The Gaussian process

One joint GP over the six inputs $(x, k)$, so the correlation of the curve along $L_r$ is modelled and preserved in
joint draws (no principal-component reduction, hence no reconstruction residual). Inputs are scaled to the unit box
by declared bounds and outputs standardised. The covariance is the anisotropic Matérn-5/2 kernel

$$ k(z, z') = s^2\left(1 + \sqrt5\,r + \tfrac53 r^2\right)e^{-\sqrt5\,r},\qquad r = \left\lVert \frac{z - z'}{\ell}\right\rVert, $$

with a constant mean, signal variance $s^2$, length scales $\ell_j$ and a noise variance $\sigma_n^2$ (plus a floor
of $10^{-8}$; the FE outputs are deterministic up to the solver tolerance). Hyperparameters maximise the log marginal
likelihood

$$ \log p(y\mid\theta) = -\tfrac12 y^T K^{-1} y - \tfrac12\log|K| - \tfrac n2\log 2\pi,\qquad K = K_{s}(\theta) + (\sigma_n^2 + \text{jitter})I, $$

with its analytic gradient, multi-start L-BFGS-B, length scales in [0.02, 20] (scaled units), signal variance in
[$10^{-3}$, $10^2$] and noise in [$10^{-10}$, 1] (standardised). The posterior of the latent function is

$$ \mu(z) = k(z, Z)K^{-1}y,\qquad \operatorname{cov}(z, z') = k(z, z') - k(z, Z)K^{-1}k(Z, z'), $$

and leave-one-out predictions use the closed form $y_i - \mu_{-i} = [K^{-1}y]_i/[K^{-1}]_{ii}$,
$\sigma_{-i}^2 = 1/[K^{-1}]_{ii}$ (hyperparameters fixed). The engine is small (NumPy/SciPy) by design; an independent
OpenTURNS kriging model is the cross-check (§6.5).

### 6.4.5 Discretisation discrepancy

The target is the converged FE curve, not the production mesh. Paired cases at levels 1 and 2 give
$d(x, k) = y_{L1} - y_{L2}$. Richardson extrapolation with level 1 the coarser gives

$$ y^* = y_{L1} + R\,(y_{L2} - y_{L1}) = y_{L1} - R\,d,\qquad R = \frac{r^p}{r^p - 1},\ r = 4/3, $$

with the order $p$ sampled uniformly on [0.9, 1.8] (the observed deepest-point range, $R \approx 2.5$–4.4). $d$ is
modelled per step by a linear regression in the scaled inputs when at least 8 pairs reach the step (an intercept
otherwise), with its residual standard deviation widened by the regression's own uncertainty at $x$,
$\sqrt{1 + \text{leverage}(x)}$, and **one** standard normal shared along the curve (one mesh pair makes the whole
difference). A one-sided far-field allowance multiplies $J/J_e - 1$ by $1 + \phi$ with $\phi$ uniform on [0, 0.25 %]
(from D7b). Both terms enter every draw. They are evidenced at the mid-box point only, which the experimental label
covers.

### 6.4.6 Support and statuses

A prediction never raises for a realisation; it carries a status:

| Status | When |
|---|---|
| `supported` | $x$ inside the box and the queried $L_r$ inside the support |
| `outside_box` / `invalid` | $x$ outside the box / not finite |
| `beyond_support` | $L_r$ beyond the support given by the schedule rule ($k \le 11$, $L_r \le 1.2$) |
| `cap_uncertain` | $L_r$ beyond the support because the small-strain cap rule cut it |

**Small-strain cap rule.** The emulator cannot know in advance where a new input's own FE strain field would exceed
the local cap. It applies a case-specific rule: a step whose reference strain $\varepsilon_0 L_r^n$ reaches the
smallest reference strain at which the cap cut a training case is not advertised (and if fewer than two steps remain,
the whole curve is `cap_uncertain`); validation fails on any held-out support overrun; and the support never exceeds
step 11 and $L_r = 1.2$, where the far-region strain at the mid-box point was ≤ 1.3 % against the 5 % cap. Whether a
rule that is safe a priori exists is an open item.

Between steps, `Prediction.kr_at` interpolates linearly in $J/J_e$ (which keeps monotonicity) as information. Assessments
and Pf use the rigorous between-step bounds of [07_pf_emulator.md](07_pf_emulator.md) §7.2 instead.

### 6.4.7 Use in an assessment

`assessment_curve(prediction, lr_max, experimental=False)` refuses a prediction that is not assessment-grade unless
`experimental=True`, refuses a non-supported prediction, cuts the curve at $L_{r,max}$, and returns it with labels:
the verification status, "deepest point only (the surface position is unresolved: mesh-sensitive)", and
"experimental". A deepest-point curve is never silently applied to the whole flaw.

## 6.5 Validation (V9) and its results

Criteria were fixed before the held-out runs:

- held-out $K_r$ RMSE over supported steps ≤ 0.02;
- coverage of the 95 % intervals per curve (a point within $10^{-4}$ in $K_r$ counts as covered, the resolution of the
  increment floor), reported with its curve-level standard error;
- the **energy score** of the whole supported curve vector,
  $ES = E\lVert X - y\rVert - \tfrac12 E\lVert X - X'\rVert$ (a proper multivariate score), against a mean-only
  baseline, and the **variogram score**
  $VS_{1/2} = \sum_{i<j}\left(|y_i - y_j|^{1/2} - E|X_i - X_j|^{1/2}\right)^2$ as the dependence diagnostic;
- leave-one-curve-out (LOCO) on the training set;
- an independent OpenTURNS kriging model (Matérn-5/2, constant trend, its own hyperparameters) on the same data: its
  held-out predictions inside the GP's 95 % intervals at ≥ 90 % of points;
- zero support overruns.

Global coverage alone is not accepted as evidence that a Pf interval is calibrated.

**Data.** 22 level-1 cases were solved of 48 planned (scrambled Sobol' design in the five-dimensional box, fixed seed);
20 were accepted (one failed the remote-profile check, one is excluded). No further FE was run: V9 uses
LOCO over every accepted training curve (each curve held out in turn, the emulator refitted on the rest) and two
existing independent level-1 runs held out, the D7 mid-box and corner points (same production mesh and far field).

| Criterion | Result | Verdict |
|---|---|---|
| LOCO $K_r$ RMSE ≤ 0.02 | **0.0241** (maximum absolute error 0.118, at edge-of-box curves: low $n$, $a/c \to 1$) | **not met** |
| LOCO 95 % coverage (curve level) | 0.935 (20 curves; no standard error recorded) | below the nominal 0.95 |
| held-out (D7 mid, corner) RMSE ≤ 0.02 | 0.0093 | met on 2 curves only (weak evidence) |
| held-out coverage / support overruns | 1.00 / 0 | met on 2 curves only (22 points; weak evidence) |
| held-out energy score, emulator vs mean-only baseline | 0.031 vs 0.029 | not met: the emulator scores worse than the mean-only baseline (2 curves) |
| OpenTURNS agreement ≥ 0.9 | 1.00 | met |

The LOCO RMSE (0.0241) misses the 0.02 target and LOCO coverage (0.935) is below nominal; the held-out checks that
pass rest on two curves. With 20 training curves in five inputs the miss is consistent with too little data, but the
data do not establish that more cases would close it. V9 is therefore **partially met** and the emulator stays
experimental.

## 6.6 Limits of the FE pipeline and the emulator

- One geometry family: a semi-elliptical surface flaw in a flat plate, proportional tension and bending, one mesh
  topology, $a/B \in [0.2, 0.6]$, $a/c \in [0.4, 1.0]$; $\nu$ and $c/W$ fixed in the emulator.
- Deformation plasticity, small strain; curves end where the strain caps end them ($L_r$ between 1.07 and 1.5 by the
  schedule rule, and at most 1.2 in the emulator).
- Deepest point only; the near-surface station is mesh-sensitive and unresolved.
- No case is `fe-verified`; D4 and D5 are blocked on references; D7 is unresolved for the family; the deep-flaw /
  bending K band is unresolved.
- The emulator was trained on FE data whose family-level verification is incomplete, so it is experimental, and so
  is everything computed with it.
- The discrepancy model is evidenced at the mid-box point only; GP hyperparameter uncertainty is not propagated.
