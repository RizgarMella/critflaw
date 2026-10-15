# WARP3D commands emitted by `critflaw.fe.deck` (gate (i) deliverable)

Each command keyword the writer emits, with the User Manual 18.5.0 section it was checked against (manual text is
not reproduced). The test `tests/test_fe_deck.py::test_every_emitted_command_is_listed_here` tokenises every
emitted deck and asserts the keyword is in this table.

| keyword (first two words) | manual section | note |
|---|---|---|
| `structure` | 2.1 | model name |
| `number of` | 2.1 | `number of nodes N elements M` |
| `material` | 2.2.1 | material definition |
| `properties mises` | 3.5 / 3.6 (mises) | elastic runs: `yld_pt` far above any stress (test87 practice) |
| `properties deformation` | 3.4, Table 3.4.1 | `e nu yld_pt n_power` |
| `elements` | 2.3 | element type and properties |
| `all type` | 2.3 | `all type q3disop linear material ...` |
| `order 2x2x2` | 2.3, Table 3.1.1 | integration order and output location (`nodpts short`) |
| `*input` | 1.3 | `*input from 'file'` |
| `*echo` | 1.3 | `*echo off` at the head of every generated include file |
| `blocking automatic` | 2.6.2 | |
| `loading` | 2.8.1 | loading name |
| `nonlinear` | 2.8.4 | step loads (increments of pattern factors) |
| `step` | 2.8.4 | `step k pattern increment` |
| `nonlinear analysis` | 2.10 | `nonlinear analysis parameters` |
| `solution technique` | 2.10.2 | sparse direct / sparse iterative |
| `time step` | 2.10.3 | large value for a static analysis |
| `maximum iterations` | 2.10.4.1 | |
| `minimum iterations` | 2.10.4.2 | |
| `extrapolate on` / `extrapolate off` | 2.10.7 | |
| `nonconvergent solutions` | 2.10.4.3 | `stop` / `continue` |
| `convergence test` | 2.10.4.5 | `norm res tol` |
| `trace solution` | 2.10.12 | |
| `adaptive solution` | 2.10.6 | |
| `line search` | 2.10.5 | |
| `divergence check` | 2.10.4.4 | |
| `batch messages` | 2.10.9 | |
| `compute displacements` | 2.11 | `compute displacements for loading L step k` |
| `output wide` | 2.12.1 | `output wide eformat reactions <list>` (the fixed format overflows above 1e4 N) |
| `output wide` | 2.12.1 | `output wide eformat stresses/strains for elements <list>` (Gauss points; the fixed format overflows) |
| `output strains` | 2.12.1 | `output strains for elements <list>` |
| `output wide` | 2.12.1 | `output wide eformat displacements for nodes <list>` |
| `domain` | 4.4 | domain definition start |
| `normal plane` | 4.4 | |
| `tangent vector` | 4.4 | the `phi = 0` node only |
| `symmetric` | 4.4 | J doubled for symmetry (checked by the parser against the declared planes) |
| `front nodes` | 4.4 | corner and mid-side nodes, `verify` |
| `q-values automatic` | 4.4 | rings |
| `function type` | 4.4 | a / b / c / d |
| `plane strain` / `plane stress` | 4.5.7.5 | `plane strain output for interaction integral` |
| `print domain` | 4.4 (revision note, domain extents) | `print domain extents` |
| `compute domain` | 4.4 / 4.5.7.5 | `compute domain integral` / `compute domain interaction integrals` |
| `binary packets` | 2.12 (Binary Packets File), 2.10 | `binary packets on file '<name>'` inside the nonlinear analysis parameters; full-precision results (Appendix F) |
| `output packets` | 4.4.13 | `output packets J` in a domain definition: packet type 17 (Appendix F.19: per ring dm1..dm8, total J, killed elements; then average/min/max; then K from J) |
| `stop` | 1.3 | |
| `coordinates` | 2.4 | the coordinate file (`*echo off` then `coordinates`) |
| `incidences` | 2.5 | the incidence file (three continuation lines per element) |
| `constraints` | 2.7.1 | the constraint file: `<node> u/v/w 0.0` |
| `element loads` | 2.8.3, 3.1.5 | `<element> face <n> pressure <value>` (positive into the face) |
| `nodal loads` | 2.8.2 | `<node> force_x/y/z <value>` (additive per node) |
