# References

Citations in this manual use the bibliography keys of the equation register (`src/critflaw/provenance.py`,
`REFERENCES`). A register citation means **"as cited by the source standard"**: every record currently has
`checked_against_primary = "no"`, i.e. the implemented coefficients were verified against the printed standard, not
against the publication below ([08_knowledge_base.md](08_knowledge_base.md) §8.1). Items for which the source
standard cites no primary reference carry `primary_reference=None` and are listed in `docs/unsourced_items.md`; no
citation has been supplied for them here.

## 1. Primary references cited by register records

| Key | Reference | Register records |
|---|---|---|
| AEA2002 | AEA Technology Plc. Partial safety factors for SINTAP procedure. HSE Offshore Technology Report 2000/020. London: The Stationery Office, 2002. | `partial_factors.table` |
| Ainsworth1986 | Ainsworth, R.A. The treatment of thermal and residual stresses in fracture assessments. Engng Fract Mech, 1986, 24: 65-76. | `fad.rho.simplified` |
| AinsworthSmithWiesner1999 | Ainsworth, R.A., Smith, S.D. and Wiesner, C.S. Treatment of thermal and residual stresses in the SINTAP defect assessment procedure. British Energy Generation Report EDP/GEN/REP/0423/99 and SINTAP/NE/018, 1999. | `fad.v.simplified` |
| AndersonEtAlWRC471 | Anderson, T.L., Thorwald, G., Revelle, R.J., Osage, D.A., Janelle, J.L. and Fuhry, M.E. Development of stress intensity factor solutions for surface and embedded cracks in API 579. WRC Bulletin 471, 2002. | `k.cylinder.tables.axial_extended_external` |
| DedhiaHarris1983 | Dedhia, D.D. and Harris, D.O. Improved influence functions for part-circumferential cracks in pipes. In: Circumferential Cracks in Pressure Vessels and Piping, ASME PVP (1983). | `k.cylinder.tables.circumferential_surface` |
| Folias1984 | Folias, E.S. An axial crack in a pressurized cylindrical shell. International Journal of Fracture 26 (1984) 251-260. | `sref.cylinder.axial_thin` |
| FranceGreenSharples1996 | France, C.C., Green, D. and Sharples, J.K. New stress intensity factor and crack opening area solutions for through-wall cracks in pipes and cylinders. AEA Technology Report AEAT-0643, 1996; addendum 1999. | `k.cylinder.tables.through` |
| INSTA1991 | INSTA Technical Report. Assessment of Structures Containing Discontinuities. Stockholm: Materials Standards Institution, 1991. | `toughness.charpy_lower_bound` |
| JonesEshelby1990 | Jones, M.R. and Eshelby, J.M. Limit solutions for circumferentially cracked cylinders under internal pressure and combined tension and bending. Nuclear Electric report TD/SID/REP/0032, 1990. | `sref.cylinder.circumferential_thick` |
| JosephErdogan1989 | Joseph, P.F. and Erdogan, F. Surface crack problem in plates. International Journal of Fracture 41 (1989) 105-131. | `k.cylinder.tables.circumferential_extended` |
| KastnerEtAl1981 | Kastner, W., Rohrich, E., Schmitt, W. and Steinbuch, R. Critical crack sizes in ductile piping. International Journal of Pressure Vessels and Piping 9 (1981) 197-219. | `sref.cylinder.circumferential_thin` |
| KiefnerEtAl1973 | Kiefner, J.F., Maxey, W.A., Eiber, R.J. and Duffy, A.R. Failure stress levels of flaws in pressurized cylinders. Progress in Flaw Growth and Fracture Toughness Testing, ASTM STP 536 (1973) 461-481. | `k.cylinder.bulging_surface` |
| King1998 | King, R. A review of fatigue crack growth rates in air and seawater. HSE Report OTH 511. London: Health & Safety Executive Books, 1998. | `fatigue.recommended_laws`, `fatigue.thresholds` |
| Lei2004 | Lei, Y. J-integral and limit load analysis of semi-elliptical surface cracks in plates under combined tension and bending. International Journal of Pressure Vessels and Piping 81 (2004) 43-56. | `sref.cylinder.axial_thick_bending` |
| Lei2008 | Lei, Y. A review of limit load solutions for cylinders with axial cracks and development of new solutions. International Journal of Pressure Vessels and Piping 85 (2008) 825-850. | `sref.cylinder.axial_pressure` |
| LeiBudden2004 | Lei, Y. and Budden, P. Limit load solutions for plates with embedded cracks under combined tension and bending. International Journal of Pressure Vessels and Piping 81 (2004) 589-597. | `sref.plate.embedded_extended` |
| LuconEtAl2005 | Lucon, E., Wallin, K., Langenberg, P. and Pisarski, H. The use of Charpy/fracture toughness correlations in the FITNET procedure. OMAE 2005-67569, Proc. OMAE 2005, Halkidiki, Greece. | `toughness.charpy_upper_shelf` |
| NewmanRajuTM85793 | Newman, J.C. and Raju, I.S. Stress intensity factor equations for cracks in three-dimensional finite bodies subjected to tension and bending loads. NASA Technical Memorandum 85793. | `k.plate.surface`, `k.plate.embedded` |
| ParisErdogan1963 | Paris, P.C. and Erdogan, F. A critical analysis of crack propagation laws. Journal of Basic Engineering (Trans. ASME) 85 D(4) (1963) 528-534. | `fatigue.paris_law` |
| R6 | R6 Panel. R6: Assessment of the integrity of structures containing defects. Revision 4, as amended. Gloucester: EDF Energy, 2001. | `fad.kjs.plastic_zone` |
| RajuNewmanVessels | Raju, I.S. and Newman, J.C. Stress intensity factor influence coefficients for internal and external surface cracks in cylindrical vessels. In: Aspects of Fracture Mechanics in Pressure Vessels and Piping, ASME PVP (1982). | `k.cylinder.tables.axial_surface` |
| RookeCartwright1976 | Rooke, D.P. and Cartwright, D.J. Compendium of stress intensity factors. London: The Stationery Office, 1976. | `k.plate.extended`, `k.cylinder.extended_external` |
| Wallin1989 | Wallin, K. Simple theoretical Charpy V-K_IC correlation for irradiation embrittlement. In: Innovative Approaches to Irradiation Damage and Fracture Analysis. New York: ASME, 1989. PVP-170. | `toughness.charpy_t0`, `toughness.charpy_master_curve` |
| WallinRintamaa1997 | Wallin, K. and Rintamaa, R. Statistical reevaluation of the ASME K_IC and K_IR fracture toughness reference curves. 23rd MPA Seminar, Stuttgart, October 1997. | `toughness.master_curve` |
| WilloughbyDavey1989 | Willoughby, A.A. and Davey, T.G. Plastic collapse at part wall flaws in plates. In: R.P. Wei, ed. Fracture mechanics: perspectives and directions. Proc. 20th national symposium. STP 1020. Philadelphia: ASTM, 1989: 390-409. | `sref.plate.surface`, `sref.plate.through`, `sref.plate.embedded` |

The FAD definitions themselves (`fad.cutoff`, `fad.option1.continuous`, `fad.option1.discontinuous`, `fad.kr`) have
no open primary reference identified (`docs/unsourced_items.md`).

## 2. In the register's bibliography but not cited by any current record

Kept in `REFERENCES` for related solutions; listed for completeness.

| Key | Reference |
|---|---|
| AndrasicParker1984 | Andrasic, C.P. and Parker, A.P. Dimensionless stress intensity factors for cracked thick cylinders under polynomial crack face loadings. Engineering Fracture Mechanics 19 (1984) 187-193. |
| BergmanBrickstad1990 | Bergman, M. and Brickstad, B. Stress-intensity factors for circumferential cracks in pipes by line-spring calculations. SA/FoU Report 90/05, Swedish Plant Inspectorate, 1990. |
| Lei2007 | Lei, Y. Limit load solutions for cylinders with axial cracks. British Energy report E/REP/BBGB/0019/GEN/07, 2007. |
| Lei2008EAN | Lei, Y. Local limit load solutions for plates with surface cracks under combined tension and bending. British Energy engineering advice note E/EAN/BBGB/0014/GEN/08, 2008. |
| Miller1988 | Miller, A.G. Review of limit loads of structures containing defects. International Journal of Pressure Vessels and Piping 32 (1988) 197-327. |
| NewmanRaju1981EFM | Newman, J.C. and Raju, I.S. An empirical stress intensity factor equation for the surface crack. Engineering Fracture Mechanics 15 (1981) 185-192. |
| NewmanRaju1983 | Newman, J.C. and Raju, I.S. Stress intensity factor equation for cracks in three-dimensional finite bodies. In: Fracture Mechanics: Proc. of 14th Symposium. Vol. I: Theory and Analysis. STP 791. Philadelphia: ASTM, 1983: 238-265. |
| RajuNewman1981 | Raju, I.S. and Newman, J.C. Stress intensity factor equations for cracks in three dimensional finite bodies. NASA TM-83200, 1981. |

## 3. FE software and verification references named in the repository

| Reference | Where used | Status |
|---|---|---|
| WARP3D User Manual, Release 18.5.0 (April 2026 update), distributed from the WARP3D project's downloads page | every deck command is cited to a manual section in `docs/manual/warp3d_commands.md`; [06_fe_option3.md](06_fe_option3.md) §6.1 | obtained |
| WARP3D verification suite: the single-edge-notch bend regression deck and the `test87` surface-crack model (an FEACrack-generated 8-node mesh) | D1 and D3b, [06_fe_option3.md](06_fe_option3.md) §6.2 | used (test87 with the crack-face traction term added to its domains, stated) |
| Kumar, German and Shih, 1981, EPRI NP-1931 (fully plastic $h_1$ tables) | D4 | **not obtained**; D4 blocked |
| Lei, Y., International Journal of Pressure Vessels and Piping 81 (2004) 21–30 (tension) and 31–41 (bending): published 3-D surface-crack J (candidate references; distinct from the Lei2004 register entry) | D5 | **not obtained**; D5 blocked |
| Wang, 1991 MIT thesis (the reference the WARP3D manual gives for the deformation law's transition arc) | `WarpDeformationLaw` | **not obtained** (the download needs a browser verification step); the D6 residual is the recorded bound |
| NASA TM 85793 tabulated boundary-correction factors (entry NewmanRajuTM85793) | attribution of the D3 deep-flaw / bending band | not yet compared |
| ASME V&V 20 / Celik et al. 2008: the observed-order procedure for non-constant refinement ratios and the grid convergence index | D7, [06_fe_option3.md](06_fe_option3.md) §6.2 | named in the design notes; full citation not recorded in the repository |
| OpenTURNS | independent cross-check of the probabilistic transform and FORM, and of the emulator's GP (kriging) | software |

## 4. Statistical and reliability methods named in the code and design notes

The repository names these methods and, where shown, a source; full bibliographic details are not recorded in the
repository and have not been added here.

| Method | Named source in the repository | Used in |
|---|---|---|
| Pathwise conditioning of Gaussian processes (random-Fourier-feature prior plus exact update) | "Wilson et al. 2020" (`src/critflaw/option3/gp.py`) | [07_pf_emulator.md](07_pf_emulator.md) §7.3 |
| Empirical-Bernstein bound for the mean of bounded i.i.d. summands | "Audibert, Munos and Szepesvári 2009, Theorem 1" (`src/critflaw/prob/intervals.py`; `docs/design/phase2.md` D16) | [04_probabilistic.md](04_probabilistic.md) §4.4.2 |
| Second-order reliability approximation | "Breitung" (`src/critflaw/prob/form.py`) | [04_probabilistic.md](04_probabilistic.md) §4.4.3 |
| Improved HL-RF iteration for the FORM design point | "improved HL-RF" (`src/critflaw/prob/form.py`) | [04_probabilistic.md](04_probabilistic.md) §4.4.3 |
| Clopper–Pearson exact binomial interval | named (`src/critflaw/prob/intervals.py`) | [04_probabilistic.md](04_probabilistic.md) §4.4, [05_fatigue.md](05_fatigue.md) §5.5 |
| Gaussian copula / Nataf transformation | named (`docs/roadmap.md`, `bs7910_probabilistic_library_proposal.md`) | [04_probabilistic.md](04_probabilistic.md) §4.2 |
| Kish effective sample size | named (`docs/design/phase3.md` F16) | [05_fatigue.md](05_fatigue.md) §5.6.2 |
| Matérn-5/2 covariance, maximum-marginal-likelihood GP regression | named (`src/critflaw/option3/gp.py`) | [06_fe_option3.md](06_fe_option3.md) §6.4.4 |
| Energy score and variogram score (proper scores for multivariate forecasts) | named (`src/critflaw/option3/validation.py`) | [06_fe_option3.md](06_fe_option3.md) §6.5 |
| Scrambled Sobol' sequences | named (`src/critflaw/option3/design.py`) | [06_fe_option3.md](06_fe_option3.md) §6.5 |
| Richardson extrapolation | named (`src/critflaw/option3/emulator.py`) | [06_fe_option3.md](06_fe_option3.md) §6.4.5 |

Standard textbook references for subset simulation, the Nataf transformation and Gaussian-process regression are not
named anywhere in the repository and are therefore not listed.
