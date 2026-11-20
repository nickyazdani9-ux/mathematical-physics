# GTOR 2.0 — Research Audit 02 (4 November 2026)

**Status:** Source-checked literature notes plus an elementary symbolic benchmark. **Not a claim of physical validation.**  
**Base document:** [GTOR 2.0 working draft](../GTOR_2_0_WORKING_DRAFT.md), Audit 01 (3 November 2026).  
**Scope:** Quantum rays/reference frames, black-hole geodesic completeness, regularity, inner-horizon stability, asymptotic models.

## Delta relative to Audit 01

Audit 01 already covers the invariance of Born probabilities under global phase, normalization as choice of ray representative, singularity theorems as geodesic-incompleteness theorems, and the insufficiency of a coordinate remapping alone. **Do not count those as new research findings.** This audit provides a reproducible example for the last point and new comparator papers on the dynamical fate of regular black-hole interiors. The previously listed GW250114 spectroscopy article, Luminet singularities primer and 2025 quantum-gravity paper are retained as existing citations, not re-ingested as discoveries.

### A. Quantum geometry and relative phase — foundational comparators

- Ashtekar & Schilling, *Geometrical Formulation of Quantum Mechanics* (1997 preprint/1999 chapter), arXiv https://arxiv.org/abs/gr-qc/9706069 ; chapter DOI https://doi.org/10.1007/978-1-4612-1422-9_3 . Pure physical states are rays; projective Hilbert space carries Kähler/Fubini–Study geometry, not a featureless quotient.
- Anandan & Aharonov, *Geometry of quantum evolution*, Phys. Rev. Lett. 65, 1697 (1990), DOI https://doi.org/10.1103/PhysRevLett.65.1697 . Projective-space path length connects to integrated energy uncertainty.
- Bartlett, Rudolph & Spekkens, *Reference frames, superselection rules, and quantum information*, Rev. Mod. Phys. 79, 555 (2007), DOI https://doi.org/10.1103/RevModPhys.79.555 . Phase becomes operational through **relative** reference systems. This does not make multiplying the *entire closed system state* by one global phase measurable.
- Polchinski, *Weinberg's nonlinear quantum mechanics and the Einstein-Podolsky-Rosen paradox*, Phys. Rev. Lett. 66, 397 (1991), DOI https://doi.org/10.1103/PhysRevLett.66.397 . Candidate non-linear alternatives must explicitly check no-signalling and composition of distant subsystems.

**Exact two-level benchmark.** For |ψ⟩ = cos(θ/2)|0⟩ + exp(iφ)sin(θ/2)|1⟩, the metric
ds² = ⟨dψ|dψ⟩ − |⟨ψ|dψ⟩|² = 1/4 [dθ² + sin²θ dφ²] (real part understood). The common U(1) phase drops out, but the relative phase φ remains in the geometry. Any GTOR added absolute phase variable must define additional operational measurement outcomes beyond standard closed-system projectors; an external reference must be modelled explicitly.

### B. Black-hole interiors — new comparisons not present in Audit 01

1. Hayward, *Formation and Evaporation of Nonsingular Black Holes*, Phys. Rev. Lett. 96, 031103 (2006), DOI https://doi.org/10.1103/PhysRevLett.96.031103 . A solvable regular-core reference metric; not a demonstration that every nonsingular candidate survives perturbations.
2. Di Filippo, Kolář & Kubizňák, *Inner-extremal regular black holes from pure gravity*, Phys. Rev. D 111, L041505 (2025), DOI https://doi.org/10.1103/PhysRevD.111.L041505 . Motivates zero surface gravity at the inner horizon to evade the usual classical exponential instability.
3. Frolov & Zelnikov, *Regular black holes in quasitopological gravity: Null shells and mass inflation*, Phys. Rev. D 113, 084007 (2 April 2026), DOI https://doi.org/10.1103/nrd7-j6hs . Some regular geometries suppress significant amplification except at extremely fine-tuned near-inner-horizon shell interactions. Scope: null-shell model, not a generic global stability theorem.
4. Bonanno, Panassiti & Saueressig, *Cauchy horizon stability and instability of regular black holes*, Phys. Rev. D 114, 064027 (8 September 2026), DOI https://doi.org/10.1103/34w6-s2yl . Finds power-law rather than standard exponential mass growth for classes including Hayward; an asymptotic-safety-inspired example shows logarithmic growth. Power-law growth need not be bounded, so 'tamed' is not 'fully stable'.
5. Liang & Li, *Black Holes With Complete Asymptotic Throats* (arXiv submission 30 September 2026), https://arxiv.org/abs/2609.39186 , arXiv DOI https://doi.org/10.48550/arXiv.2609.39186 . **Preprint, not verified peer-reviewed.** Derives completeness conditions with two independent metric functions and shows radial NEC failure near a complete inner end under its assumptions; exterior can obey energy conditions. Bounded scalar invariants need not bound all parallel-propagated curvature components.
6. Feng, Liu, Ling, Jiang et al., *Global curvature control in a regular black hole with Minkowski core and inner-extremal horizon*, Eur. Phys. J. C 86, 1205 (22 October 2026), DOI https://doi.org/10.1140/epjc/s10052-026-16504-3 . Constructs a particular metric whose cubic inner-horizon zero gives polynomial blueshift ∝v^(3/2) and bounded Misner–Sharp mass in the studied shell/Ori models. Does **not** prove unrestricted nonlinear stability.
7. Tello-Ortiz, *Integrable singularities obstruct Cauchy horizons and mass inflation* (arXiv submission 28 October 2026; revised 30 October), https://arxiv.org/abs/2610.38826 , arXiv DOI https://doi.org/10.48550/arXiv.2610.38826 . **Preprint, not verified peer-reviewed.** Under specific spherical anisotropic-source/screening assumptions, excludes an inner Cauchy horizon and its associated mass-inflation channel. Retains an integrable *curvature singularity*; thus it is not a regular-core solution. Avoid conflating absence of that instability with full regularity.

**Physical distinction:** (i) no divergence in scalar invariants, (ii) geodesic completeness, (iii) controlled curvature in physical parallel-propagated frames, (iv) stable dynamics under specified perturbations and (v) observed exterior consistency are different tests. No one implies the other automatically.

### C. Independently checked elementary calculation — coordinates do not cure the Schwarzschild singularity

Take geometric units G=c=1 and Schwarzschild K=R_abcd R^abcd=48M²/r⁶. The GTOR draft's proposed bounded transformation r=L u/(1+u), with M,L,u>0, gives:

K(u) = 48M²(1+u)⁶/(L⁶u⁶).

Therefore K→∞ as u→0 and K→48M²/L⁶ as u→∞. A bounded coordinate radius is not a dynamical curvature regularization.

For a static spherical metric ds²=−f(r)dt²+dr²/f(r)+r²dΩ², conserved radial-null energy E=f dt/dλ gives (dr/dλ)²=E² wherever this chart is valid; analytically continue across coordinate horizons. Hence finite areal-radius intervals have finite affine-null length, even if the coordinate u tends to ∞. In particular u→∞ maps to r=L reached after finite Δλ=(L−r_initial)/|E|. This does not establish geodesic completeness.

By comparison, the Hayward toy metric f(r)=1−2Mr²/(r³+2Mℓ²) behaves f(r)=1−r²/ℓ²+O(r⁵) near its regular center. For Schwarzschild-form static spherical metrics,

K = [f''(r)]² + 4[f'(r)]²/r² + 4[f(r)−1]²/r⁴,

so the Hayward limit is K(0)=24/ℓ⁴, finite for positive ℓ. Curvature regularity still leaves questions about global extension, horizons, stress energy, perturbations and completeness.

A symbolic SymPy calculation confirmed both curvature limits and the metric expansion; this is an algebraic check of known formulae, **not** GTOR physical validation.

## Recommended reproducible test queue

1. **Quantum consistency / smallest task:** Reproduce the qubit Fubini–Study metric and invariance under |ψ⟩→e^(iα)|ψ⟩. Add a composite reference mode R and specify an *allowed observable* for a proposed GTOR distinction. Reject a physical claim if standard joint probabilities stay identical.
2. **Metric invariants:** Starting with an explicit GTOR interior metric (not merely r(u)), symbolically compute R, R_ab R^ab and K, and check all endpoints and horizons. Compare Schwarzschild and Hayward as negative/positive curvature baselines.
3. **Geodesic completeness:** Numerically integrate radial/tangential null and timelike geodesics to test whether affine/proper time diverges at a proposed inner asymptote. Separate finite coordinate radius from infinite physical length; check parallel-propagated curvature.
4. **Dynamics:** Locate inner horizons and compute surface gravities. Introduce incoming/outgoing null perturbations and compare mass inflation across Hayward, zero-surface-gravity models and no-inner-horizon models with stated flux assumptions.
5. **Empirical prediction:** After specifying exterior and perturbation equations, compute quasi-normal frequency shifts and compare with the already-catalogued 2026 GW250114 constraints; no 'confirmed GTOR' label without a quantitative fitted likelihood or exclusion bound.

## Evidence/limitations

Search is selective, not an exhaustive survey of every paper; publisher abstracts and open-access arXiv text were accessible. APS journal full text for several papers requires separate access; those findings above are explicitly limited to publisher abstract descriptions. Recent preprints are unreviewed unless independently established otherwise. All predictions from GTOR remain **unverified**. No cloud resources, API purchases or billable operations were requested or used.

