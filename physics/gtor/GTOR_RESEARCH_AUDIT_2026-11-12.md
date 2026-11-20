# GTOR — Research Audit 10 (12 November 2026 AEDT)

**Status:** Selective public-primary-source review. Four distinct recent preprints and two foundational additions, deduplicated against existing research audits dated 3–11 November 2026. All four recent papers are **unreviewed arXiv preprints**, not observational confirmation of GTOR. Full public arXiv HTML inspected for all four; primary publisher abstracts/metadata inspected for the two historical papers. No paywalled text claimed. No merge, cloud provisioning or spending.

## A. Kinematic quantum geometry is geometry of a *parametrized quantum state family*, not extra absolute phase

**Bernardo Barrera, Nishchhal Verma, Raquel Queiroz, Anatoli Polkovnikov and Anushya Chandran**, *Kinematic Quantum Geometry*, submitted **9 Nov 2026**. [Public full HTML v1](https://arxiv.org/html/2611.13089v1), [arXiv abstract](https://arxiv.org/abs/2611.13089), [DOI 10.48550/arXiv.2611.13089](https://doi.org/10.48550/arXiv.2611.13089) (arXiv DOI listed as pending DataCite registration at review).

This extends Born–Oppenheimer (BO) approximation with a fast subsystem following states |psi_n^(KBO)(x,v)> that depend on *slow coordinates x and velocities v*. The authors define a connected adiabatic-gauge-potential covariance in enlarged parameter space z=(x,v):
```
Q_ab = <A_a A_b>_c;
h_ab = Re(Q_ab),  K_ab = -(2/hbar) Im(Q_ab).
```
The mixed x-v Berry curvature and effective mass correction produce additional velocity-dependent forces under their superadiabatic approximation. Small-velocity dynamics includes
```
(M_eff)_ij x_j'' = -partial_i U_BO + v_j F_ij - Gamma_ijk v_j v_k,
(M_eff)_ij = M delta_ij + kappa_ij.
```
Here prime/double prime denote ordinary time derivatives; these are **quantum-parameter-space and effective slow-DOF** quantities, not Lorentzian spacetime curvature or a new absolute complex phase. Model predictions are compared to a mixed quantum/classical Schrödinger reference and depend on timescale separation and chosen test systems. The authors give spin-texture and graphene-circuit examples; no universal fundamental-gravity claim follows.

**GTOR gate:** define a physical enlarged parameter space, state/projector, connection, gauge transformations, and operational response. If GTOR's kinematics reproduce these effective equations, classify that as established physics rather than a novel physical prediction. A measured effect must survive |psi> -> exp(i chi(x,v)) |psi> and changes of coordinates, with measurement references transformed consistently.

## B. Expansion and contraction can be indistinguishable by particle counts, distinguishable by phase correlations

**Christian F. Schmidt and Stefan Floerchinger**, *Inverse Scattering & Isospectrality in (Analog) Cosmology*, submitted **10 Nov 2026**. [Public full HTML v1](https://arxiv.org/html/2611.13871v1), [abstract](https://arxiv.org/abs/2611.13871), [DOI 10.48550/arXiv.2611.13871](https://doi.org/10.48550/arXiv.2611.13871) (arXiv DOI registration pending).

In an asymptotically static FLRW background with scalar modes, a conformal field redefinition gives a 1D scattering problem. With D spatial dimensions and w=(D-1)/2:
```
[-d²/deta² + V(eta)] psi_k = -h(k) psi_k
V = w [a''/a - (3-D)/2 (a'/a)^2]
N_k = |r_+(k)/t(k)|².
```
For D=3, V=a''/a. Defining W=a'/a yields the supersymmetric partner pair V_plus=W²+W', V_minus=W²-W', corresponding to positive a and inverse scale factor 1/a. They have the same particle counts N_k but a relative pi shift in the relevant scattering/Bogoliubov phase. The paper's inverse reconstruction of regular asymptotically static histories is unique with phase-inclusive scattering data and specified assumptions/boundary information; **particle number alone is not unique**. The paper's correspondence between singular zeroes of a and bound states is within this scattering formulation, not a general relativistic singularity theorem. Numerical inversion of selected logistic histories is a paper-reported result, not reproduced here.

**GTOR gate:** a proposed geometry-to-observable mapping must state the *complete measurement data*, including interference phase against a physical reference. Spectrum-only fitting could falsely identify an expanding vs contracting asymptotic history as the same physical model. This is not about measuring absolute global ket phase.

## C. Quantum-bounce cosmology from LQG-inspired black holes requires a junction shell and nonstandard matter

**V. F. C. Belo, F. G. S. Menezes, S. M. Rupf, I. P. R. Baranov, H. A. Borges**, *Non-singular cosmology from effective LQG-inspired black holes*, submitted **10 Nov 2026**. [Public full HTML v1](https://arxiv.org/html/2611.14001v1), [abstract](https://arxiv.org/abs/2611.14001), [DOI 10.48550/arXiv.2611.14001](https://doi.org/10.48550/arXiv.2611.14001) (registration pending).

The authors *assume* an effective LQG-inspired black-hole geometry, match it across a dynamical thin shell to flat FLRW via Darmois–Israel junction conditions and derive (paper Eq. 24; geometric units)
```
H² = (x0+2m)/x³ - 2mx0/x⁴ + 8 pi sigma/x + 16 pi² sigma².
```
Here x is the junction/areal radius, m black-hole mass, x0 the minimum radius, sigma surface energy density. At x=x0, sigma_B=-1/(4 pi x0) gives H²=0, with (a''/a)_B=(2m-x0)/(2x0³)>0 provided 2m>x0. This is an algebraic bounce but depends on **negative surface energy density near the bounce** and a transient phantom-like effective scalar mapping. Their two-field matching to lower-energy inflation relies on further assumptions; neither global geodesic completeness, perturbative stability nor observational viability is established here. It is not directly equivalent to the AdS/non-Hermitian interior of Areán et al. (Audit 09) or the finite-area core of Lan et al. (Audit 03).

**GTOR gate:** if proposing a black-hole-to-cosmology transition, supply the two region metrics, Israel jump conditions, stress-energy (including signs), causal evolution, invariant curvature and physical boundary conditions. A minimum scale factor by itself does not prove singularity resolution.

## D. Charged regular-core sign check exposes a material ambiguity in a new Frolov black-hole preprint

**Ying Huang, Huajie Gong, Li-Gang Zhu and Jian-Pin Wu**, *Quasinormal overtones and grey-body factors of Frolov black holes*, submitted **10 Nov 2026**. [Public full HTML v1](https://arxiv.org/html/2611.13569v1), [abstract](https://arxiv.org/abs/2611.13569), [DOI 10.48550/arXiv.2611.13569](https://doi.org/10.48550/arXiv.2611.13569) (registration pending).

The preprint studies test **massless scalar** QNMs and transmission on a static spherical metric:
```
ds²=-f(r)dt²+dr²/f(r)+r²dOmega²,
f(r)=1-[(2 M r-q²) r²]/[r⁴+(2 M r+q²) alpha0²],
V_eff = f(r) [ l(l+1)/r² + f'(r)/r ].
```
With ingoing-horizon/outgoing-infinity QNM conditions, it reports Im(omega)<0 over scanned charge q and regularization alpha0 ranges, but with overtone-sensitive near-horizon changes. This establishes neither gravitational/metric-sector nor nonlinear/Cauchy-horizon stability; it is scalar-test-field mode analysis.

**New independent sign check requiring clarification:** The paper describes a de Sitter-like regular core. But applying its printed f(r) at fixed *nonzero* q yields
```
q>0, r->0: f(r)=1+r²/alpha0²+O(r³),
                  R(0)=-12/alpha0², K(0)=24/alpha0^4.
q=0, r->0: f(r)=1-r²/alpha0²+O(r^4),
                  R(0)=+12/alpha0², K(0)=24/alpha0^4.
```
R is computed in metric signature (-+++): R=-f''-4 f'/r+2(1-f)/r². Thus the **charged fixed-q central limit is locally anti-de Sitter sign**, whereas the *neutral* limit is de Sitter sign; both have bounded quadratic curvature. The limits q->0 and r->0 do not commute in the coefficient of r². The numerator changes sign at r=q²/(2M). This should be reconciled with the authors' "de Sitter-like" characterization and any assumed charged-core energy conditions, rather than asserting the paper has been fully invalidated.

**GTOR gate:** evaluate signs of Ricci/sectional curvature and q/r limiting procedures for every proposed charged extension. A fixed neutral regular metric cannot be used to assume a charged branch's same local vacuum curvature.

## E. Foundational phase references not previously logged as distinct primary entries

- **S. Pancharatnam**, *Generalized theory of interference, and its applications. Part I. Coherent pencils*, *Proceedings of the Indian Academy of Sciences A* **44**, 247–262 (1956), [original institutional repository record and paper](https://repository.ias.ac.in/31456/), [DOI 10.1007/BF03046050](https://doi.org/10.1007/BF03046050). Historical interference theory for *relative* phase between polarization states.
- **J. Samuel and R. Bhandari**, *General Setting for Berry's Phase*, *Physical Review Letters* **60**, 2339–2342 (6 June 1988), [publisher abstract](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.60.2339), [DOI 10.1103/PhysRevLett.60.2339](https://doi.org/10.1103/PhysRevLett.60.2339). Generalizes operational geometric phase beyond unitary cyclic trajectories, including interruptions by measurement. The publisher abstract, **not paywalled body**, was reviewed.

Prior controls (not new): Wootters distance, Ashtekar–Schilling projective Hilbert geometry, Aharonov–Anandan, Sontz ray probabilities, Avdoshkin–Popov Bargmann invariants, Bartlett et al reference frames, Powell holonomy, Lu–Yu covariant QFI. These remain necessary null benchmarks; neither a gauge-fixing of normalization nor an overall state phase alone produces new physics.

## F. Reproducible independent checks (offline, no model-validation claim)

[gtor_audit10_phase_core_checks.py](scripts/gtor_audit10_phase_core_checks.py) (numpy/scipy/sympy; run locally with Python) carries out three tasks:

1. For D=3 use the positive asymptotically static test a(eta)=1+0.3 tanh(eta), dual 1/a, and solve psi''+(k²-V)psi=0 on eta in [-14,14] for four dimensionless k values. At k=0.4, N[a]=0.262732677288 and N[1/a]=0.262732677283; Delta N=4.55e-12, and beta phase shifts by approximately +/-pi. At k=1.3, N[a]=0.00801989650660 and N[1/a]=0.00801989650692 (Delta N=3.18e-13). Norm control |alpha|²-|beta|²=1 held to around 1e-11 for the first test. **This reproduces the *basic duality* in a new toy profile, not the paper's inverse-scattering reconstruction or real cosmology.**
2. Symbolically evaluate the charged and neutral Frolov central limits of f, Ricci scalar R and Kretschmann K, confirming the sign switch and bounded K.
3. Substitute sigma_B into Belo et al.'s H² formula to check H(x0)=0 and read off the paper's positive-acceleration condition. This is an algebraic consistency check, **not** solution of shell evolution or stability.

## Contradictions, uncertainty and research priority

**Real ambiguity to resolve:** Huang et al.'s broad de Sitter-core description must be reconciled with the anti-de Sitter Ricci sign of its printed charged f(r) as r->0 at fixed nonzero charge. Request author clarification or audit their sign conventions/physical source before adopting a universal "de Sitter core" claim.

**No contradiction:** a gauge-invariant *relative* scattering phase that distinguishes inverse cosmologies is consistent with standard quantum mechanics' invariance under global rephasing; these are different quantities. KQG state-manifold curvature is not GR spacetime curvature. A scalar-test-field QNM damping result and a nonlinear inner-horizon mass-inflation result probe different degrees of freedom. Negative energy at an effective bounce does not mean no possible bounce in any alternative theory; it marks an explicit matter/energy-condition cost in this one.

**Next prioritized tests:**
1. GTOR quantum branch: supply a normalized projective ray model, a reference/measurement operator, and a gauge-covariant response or scattering phase; test independence of local U(1) gauge and positive normalization while retaining physically controlled relative phases. Compare to KQG/standard BO as the null model. Reject a purported new absolute-phase observable if measurable distributions are gauge-identical.
2. GTOR asymptotic branch: build paired inverse histories with identical spectra but different phase correlation, then test identifiability under finite noise, limited k samples and explicit clock calibration. A spectrum-only reconstruction claim is rejected if both histories fit.
3. GTOR gravity branch: specify an action or discrete-to-continuum dynamics, metric and physical junction stress. Test R, K, causal affine lengths, parallel-propagated tidal curvature, horizons and nonlinear perturbations in charged and neutral branches. Reject a complete-regularity claim for any found geodesic incompleteness, unbounded physical curvature or unavoidable instability.
4. Observable discrimination: compare full mode spectra and excitation conditions, not only a shadow or fundamental QNM; model source and detector uncertainties. Do not interpret scalar QNM stability as full black-hole stability.

No GTOR-specific physical prediction is yet derived. All novelty here is newly catalogued literature and independently checked mathematical distinctions. The scan is selective, not exhaustive; no paper is represented as paywalled full text read. No owner approval or personal sign-in was needed for this read/research-and-draft-branch operation. No cloud provisioning, spending or merge was performed.
