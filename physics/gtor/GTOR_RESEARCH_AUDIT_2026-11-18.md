# GTOR — Research Audit 16 (18 November 2026 AEDT)

## Provenance, chronology, and status

Selective literature review, not exhaustive. Inspected arXiv gr-qc, quant-ph, and hep-th new feeds headed **Tuesday 17 November 2026**, plus the primary arXiv abstracts and public HTML for the six papers below. Crucial distinction: arXiv listing on 17 November is **not** submission on 17 November. The selected papers were submitted 5, 7, or 13 November. All six arXiv IDs were absent from the 14 prior GitHub audit files dated 4–17 November, verified by fetching their text and checking IDs (62 distinct previous arXiv-like identifiers were present). The two historical comparator papers below were not previously catalogued by title. These preprints are author claims, not peer-reviewed confirmations; DOI links on arXiv use the arXiv/DataCite prefix, with registration sometimes pending. No paywalled full text was accessed.

GTOR-specific action, metric, and observable have NOT been independently derived by these sources. The retained main criterion is a gauge-invariant prediction beyond standard quantum mechanics and a causal Lorentzian dynamics derived from GTOR's own ingredients.

## New primary studies (six)

1. **M. Baran Ökten, Ash S. Turan, Michael R. R. Good, Kuan-Nan Lin, Pisin Chen, "Transient Information in Moving-Mirror Quantum Correlations".** Submitted 5 Nov 2026; appeared in the 17 Nov new listing.
   - Primary https://arxiv.org/abs/2611.18695 ; full public HTML https://arxiv.org/html/2611.18695v1 ; DOI https://doi.org/10.48550/arXiv.2611.18695 .
   - Assumptions: 1+1-dimensional moving-mirror QFT, a common asymptotic Carlitz–Willey thermal trajectory, two compact and sign-reversed deformations of the ray map; outgoing Bogoliubov covariance measured using packet modes and a shared phase reference.
   - Result under those assumptions: identical asymptotic Hawking temperatures can conceal different transient histories. The signed linear-response squared right-moving packet correlation is fourth order in bandwidth; matched cross-sector correlations are cubic; a **relative** covariance phase responds linearly to packet bandwidth. Equation (74) gives a linear term i * omega_bar * H(0) * mu_0[g]^2 * Delta_omega/(2*pi) plus cubic corrections (paper's definitions).
   - GTOR consequence: distinguish dynamical memory via reference-calibrated correlations; this is **not** observation of absolute global wavefunction phase. A single thermal temperature or number spectrum is not sufficient to establish that two histories are physically equivalent.

2. **Faizuddin Ahmed, Ahmad Al-Badawi, İzzet Sakallı, "Kalb-Ramond black bounces: throat-blind shadows and throat-sensitive neutrinos".** Submitted 5 Nov 2026; newly listed 17 Nov.
   - Primary https://arxiv.org/abs/2611.18697 ; public HTML https://arxiv.org/html/2611.18697v1 ; DOI https://doi.org/10.48550/arXiv.2611.18697 .
   - Geometry: ds^2 = -f(R)dt^2 + dR^2/[h(R)f(R)] + R^2 dOmega^2, with h(R)=1-a^2/R^2 and f(R)=1-2M/R+Upsilon/R^n, n=2/lambda; a is the throat radius. Derived original Kalb-Ramond hair f(R), **then inserted** Simpson–Visser regularization. The resulting metric is NOT a vacuum solution of the original Kalb-Ramond equations and requires an effective exotic source; Eq. (12): rho+p_r = -2a^2*f(R)/(8*pi*G*R^4), negative outside the horizon where f>0.
   - The photon sphere equation 2f=R*f' and shadow b_c=R_ph/sqrt(f(R_ph)) are independent of a when R_ph>a. By contrast the photon-ring Lyapunov exponent and subring demagnification depend on h: gamma(a)=gamma(0)*sqrt(1-a^2/R_ph^2). Photon and neutrino propagation acquire different radial-measure factors.
   - GTOR falsifier: any proposed a-dependent shadow size when f(R) is held fixed and the above assumptions apply conflicts with this result; use subring demagnification, photon/neutrino lensing, or radial epicyclic frequencies to probe a instead. Observability and astrophysical feasibility remain separate issues.

3. **Tengyang Liu, "Finite Energy, Event Horizons, and a Universal Bound on the Singularity Order in Born--Infeld-type Black Holes".** Submitted 7 Nov 2026.
   - Primary https://arxiv.org/abs/2611.18711 ; public HTML https://arxiv.org/html/2611.18711v1 ; DOI https://doi.org/10.48550/arXiv.2611.18711 .
   - Theorem 5.1: in a specified Einstein + electrostatic interpolated Maxwell/Born–Infeld-type nonlinear electrodynamics with stated positivity, finite-energy and growth/truncation assumptions, the Kretschmann invariant cannot remain finite at the center. Generic nonzero mass-energy excess gives Schwarzschild-like K~r^-6; cancellation of that term at M=E gives the weakest K~r^-4 (possibly logarithmic), with other exponents between -6 and -4.
   - This is a **conditional obstruction** to obtaining a regular electric core in that NED sector; it is not a theorem that all charged regular black holes or all modified gravities are impossible. It is compatible with item 2's metric being supported by an additional nonstandard source, and with vacuum gravitational modified theories outside the theorem's hypotheses.
   - Baseline curvature identity from paper: K=[(r^2 A'')^2+4(r A')^2+4(A-1)^2]/r^4 for ds^2=-A dt^2+A^-1 dr^2+r^2 dOmega^2.

4. **Laura Shou, An Le, Twesh Upadhyaya, T. C. Mooney, Alexey V. Gorshkov, "Optimal Frobenius light cones for power-law systems in any dimension".** Submitted 13 Nov 2026.
   - Primary https://arxiv.org/abs/2611.18829 ; public HTML https://arxiv.org/html/2611.18829v1 ; DOI https://doi.org/10.48550/arXiv.2611.18829 .
   - Assumptions: quantum spin systems on d-dimensional lattices with pairwise power-law Hamiltonian interactions proportional to 1/r^alpha; the normalized **Frobenius/Hilbert–Schmidt norm** of commutators, not the operator norm and not a Lorentzian metric.
   - Claimed proved thresholds: Frobenius light cone changes from algebraic to linear at alpha=d+1, polylogarithmic to algebraic at alpha=d, and loses volume decay at alpha=d/2. For alpha>d, Theorem 1.1 bounds its characteristic time scale T_2(R) from below by R^[min(alpha-d,1)-epsilon], within its definitions. The **operator-norm** transition occurs at alpha=2d+1 in the cited earlier literature; do not conflate the norms.
   - GTOR implication: graph/emergent spacetime models require specifying interaction falloff, spatial dimension and operational signal norm. A Frobenius cone is an information-spreading bound, not automatic relativistic microcausality. Compare separately against the previously catalogued Creminelli–Mancini–McBlain retarded-propagator condition.

5. **Sergey Bondarenko, Raghvendra Singh, "Metric perturbations in Einstein-Hilbert gravity: An alternative perturbative formulation".** Submitted 5 Nov 2026.
   - Primary https://arxiv.org/abs/2611.18698 ; public HTML https://arxiv.org/html/2611.18698v1 ; DOI https://doi.org/10.48550/arXiv.2611.18698 .
   - Starting theory: conventional 4D Einstein–Hilbert action with a common background gauge. Compare exponential graviton fields and inverse-metric-density perturbations related nonlinearly. Off-shell quadratic operators differ on general curved backgrounds by a Ricci-dependent term from the nonzero background tadpole, which vanishes when Ricci-flat. At tree-level, off-shell cubic vertices differ by free-field-equation terms while on-shell three-point amplitudes agree when normalized in a common gauge.
   - The paper does not compute full quantum loops; measure, regulator and sources would be needed. GTOR must not call differences from invertible field-variable choices new measurable gravity. Beware of losing field equations when pre-restricting an ansatz before variation.

6. **Paolo Arnaudo, Benjamin Withers, "Schwarzschild scattering from CFT".** Submitted 13 Nov 2026.
   - Primary https://arxiv.org/abs/2611.18826 ; public HTML https://arxiv.org/html/2611.18826v1 ; DOI https://doi.org/10.48550/arXiv.2611.18826 .
   - In an AdS4/small-black-hole, large-N CFT3 framework on R x S^2, identify a high-frequency wave sector with finite near-boundary potential and take a controlled L->infinity flat-space limit while holding the first-return time T=(t-2r*_b)/R_h fixed. The extracted reflection coefficient tends to the asymptotically flat Schwarzschild partial-wave scattering coefficient, with resonance poles at the familiar quasinormal frequencies. A Schrödinger-like radial master field satisfies phi''+(omega^2-V)phi=0.
   - This is a theoretical boundary/bulk comparison, NOT an astrophysical detection, a complete result for every bulk field, or a computation of a GTOR interior. The small AdS black hole is not the dominant canonical thermal saddle, as the authors note.
   - GTOR test: before claiming altered ringdown, specify which effective exterior/scattering potential differs, compare reflection and phase shifts under fixed boundary conditions, and separate boundary preparation from interior dynamics.

## Historical foundational baselines added (not counted as new preprints)

- **E. H. Lieb and D. W. Robinson (1972), "The finite group velocity of quantum spin systems", Communications in Mathematical Physics 28(3), 251–257.** DOI https://doi.org/10.1007/BF01645779 ; institutional bibliographic abstract https://collaborate.princeton.edu/en/publications/the-finite-group-velocity-of-quantum-spin-systems/ . Establishes finite group-velocity/commutator bounds for finite-range spin systems. Checked institutional metadata/abstract; no inaccessible full-text claim.
- **A. Simpson and M. Visser (2019), "Black-bounce to traversable wormhole", JCAP 02 (2019) 042.** Open primary manuscript https://arxiv.org/abs/1812.07114 ; published DOI https://doi.org/10.1088/1475-7516/2019/02/042 . Original continuum family R(r)=sqrt(r^2+a^2) interpolates Schwarzschild, black bounce and wormhole. This is a phenomenological solution family with effective stress-energy, not derived GTOR dynamics.

## Cross-comparison / disagreements requiring explicit assumptions

| Construct | Assumed input | Observable invariant / distinction | What GTOR still lacks |
|---|---|---|---|
| Moving-mirror covariance | reference thermal ray map, compact history deformation, packet and phase reference | packet cross-covariance phase and signed finite-bandwidth response | a GTOR interaction that modifies covariance after ordinary QFT calibration |
| Projective quantum mechanics (prior audits) | states modulo nonzero complex scaling | normalized Born probabilities; Berry/Pancharatnam relative phases | nonzero invariant effect beyond U(1)/C* gauge |
| Black-bounce exterior | chosen two-function metric f,h plus effective stress | shadow b_c insensitive to a, photon-ring demagnification sensitive to a | derived GTOR metric, full stress tensor, perturbation stability |
| Electrostatic Born–Infeld class | specific NED + Einstein equations; finite-energy hypotheses | K diverges with weakest power r^-4 | a theory outside assumptions if claiming regular charged core |
| Power-law lattice | d, alpha, lattice, Frobenius norm | information cone threshold alpha=d+1 | continuum Lorentzian light cone and retarded Green-function support |
| Einstein-Hilbert field reparametrization | invertible metric coordinates and gauge matching | on-shell tree-level graviton amplitudes agree | an inequivalent action/physical spectrum |
| AdS/CFT scattering | holographic state, first return, flat-space limit | Schwarzschild reflection/QNM pole data | explicit GTOR perturbation operator and boundary prescription |

**Central tension:** "regular core" is not a property of coordinate substitution alone. The black-bounce ansatz can remove a central singularity geometrically while requiring an exotic effective stress tensor; the Born–Infeld theorem rules out regularity in a narrower charged-electric action class. Neither proves or refutes a presently unspecified GTOR gravity action. Likewise, measurable **relative** moving-mirror covariance phase and a nonzero geometric Berry phase do not make absolute global phase observable.

## Independent offline standard-library null controls

Executed the script scripts/gtor_audit16_null_controls.py on 18 Nov AEDT. It checks equations from the first black-bounce paper in the hairless f(R)=1-2M/R specialization and an illustrative local metric curvature expansion against the second paper's identity; it does NOT independently solve either article's entire field equations nor prove its theorems.

For M=1: R_ph=3, b_c=3*sqrt(3)=5.196152422707 for all tested a. Numerical 5-point differentiation of V=f/R^2 gives:
- a=0: lambda_L=0.192450089768, gamma=3.141592654218.
- a=0.5: lambda_L=0.189758347556, gamma=3.097652131392.
- a=1.5: lambda_L=0.166666666700, gamma=2.720699046895, matching lambda_L=1/6 and gamma=pi*sqrt(3)/2.
- Toy metric lapse A(r)=1-2*deltaM/r-2c yields K=48*deltaM^2/r^6+32*deltaM*c/r^5+16*c^2/r^4 exactly. With deltaM=0, measured log slope -4.000000; with deltaM=0.1, c=0.1 near r=1e-5, slope -5.999990. These check asymptotic powers, NOT the NED constitutive relation.

## Falsifiable next tests and decision order

1. **Phase branch:** implement a fixed-reference, physical POVM/covariance functional. Baseline (standard QFT) must be computed for paired histories and packet bandwidth. A GTOR extra term must survive independent allowed rephasings and be nonzero after calibration. If zero, the phase modification is pure description, not new dynamics.
2. **Relational causality branch:** propose an explicit lattice/graph Hamiltonian with an interaction falloff. Measure Frobenius and operator-norm commutator growth independently as a function of graph radius, time and system size; fit thresholds, then test causal metric reconstruction and the retarded continuum propagator. Do not claim Lorentzian emergence merely from a Frobenius light cone.
3. **Gravity branch:** derive the GTOR action and solution rather than regularize a chosen black hole by hand. Check its field equations, radial/tangential NEC and charged electric sector; compute K, parallel-propagated tidal frame and geodesic completeness. If the Einstein + NED assumptions of Liu's theorem apply, a regular electric origin is a falsified claim.
4. **Scattering branch:** derive and numerically solve the GTOR radial master equation and compute QNM/phase shifts under common sources. Verify invariance to field redefinition, exterior matching and time/phase reference; if the source or gauge change alone explains a claimed shift, reject the GTOR-specific attribution.

## Boundaries and execution

Public primary manuscripts and abstracts only; bibliographic peer-review status is kept distinct from theoretical proof claims. Selective survey, no empirical GTOR confirmation. No owner sign-in or approval was needed to read primary papers, run an offline test or commit to the existing non-merging research branch. No production merge, cloud provisioning or spending authorized or attempted. Reviewer requirements for any future merge remain intact.
