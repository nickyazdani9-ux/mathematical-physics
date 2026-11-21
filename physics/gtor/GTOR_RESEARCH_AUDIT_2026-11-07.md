# GTOR 2.0 — Research Audit 05 (7 November 2026)

**Status:** Selective public-primary-source audit; one newly published refereed paper, three previously uncatalogued preprints, two foundational comparisons, and independently reproduced toy checks. This is a research draft only. No observed validation of GTOR, no cloud provisioning or spending. **Deduplicated against Audits 01–04 (3–6 November)**: the Lan et al. single-horizon model, Peng et al. scalar-tensor no-go, Frolov/Zelnikov, Hayward, mass inflation without Cauchy horizons, Page–Wootters, Berry, Avdoshkin–Popov and helical flatness results are *not* presented as new.

## Most important delta: a further conditional incompleteness theorem

**Francesco Di Filippo**, *A Singularity Theorem for Quasi-Topological Gravities*, submitted **1 Nov 2026**, [arXiv:2611.02190](https://arxiv.org/abs/2611.02190), arXiv DOI [10.48550/arXiv.2611.02190](https://doi.org/10.48550/arXiv.2611.02190). **Open primary preprint; not verified peer reviewed. Full arXiv-rendered text reviewed.**

The author examines polynomial Birkhoff quasi-topological gravity in D≥5 and certain 4D nonpolynomial counterparts whose **time-dependent spherical** field equations share the form

```
ds² = γ_{μν}dx^μdx^ν + r(x)² dΩ²_{D−2}
ψ = [1−γ^{μν}(∇_μ r)(∇_ν r)]/r²
H(ψ) = ψ + Σ_{n≥2} c_n ψ^n
H'(ψ) R_ab k^a k^b = 8πG T_ab k^a k^b  (radial null k)
```

If the null energy condition holds and H'(ψ)>0 on the relevant future region, radial null convergence follows; Raychaudhuri focusing plus a trapped sphere, global hyperbolicity and a noncompact Cauchy slice imply **future null geodesic incompleteness**. A refined version assumes H'(ψ)>0 for ψ>0 and initial negative ingoing expansion. Regular vacuum curvature invariants alone therefore do not certify predictive, complete collapse. This is **conditional and model-class specific**, not an impossibility theorem for every proposed regular interior or every higher-curvature action. The argument may not apply if one or more hypotheses (NEC, topology/hyperbolicity, H' positivity, spherical symmetry, action class) fail; simply discarding one hypothesis is not a viable physical theory without a replacement initial-value analysis.

**Relation to Audit 04:** Peng et al., [arXiv:2611.06460](https://arxiv.org/abs/2611.06460), obtain a turning-point/tensor-hyperbolicity contradiction for specific **two-horizon scalar-tensor** interiors, not a generic focusing theorem. These constrain *different actions and hypotheses* and must not be merged as one no-go theorem. Relation to Audit 03: the Lan et al. **single-horizon finite-area-core** model is outside the ordinary regular-point-center parity reasoning; Di Filippo's QTG theorem is not automatically applicable to that model.

## New refereed rotating comparator, with decisive limitation

**J. Ovalle**, *Kerr black holes without primary hairs*, **European Physical Journal C 86, article 1268**, **published 5 Nov 2026**, fully open access [DOI 10.1140/epjc/s10052-026-16530-1](https://doi.org/10.1140/epjc/s10052-026-16530-1), [arXiv:2607.14910v2](https://arxiv.org/abs/2607.14910) (original preprint **16 July 2026**, revised **13 October**; so this is *not* a newly submitted November paper). **Open publisher HTML reviewed.**

The rotating Kerr–Schild/Gürses–Gürsey ansatz has `Δ(r)=r²−2rm(r)+a²`, with exterior `m(r)=M` for `r≥h=M+sqrt(M²−a²)`, and smooth matching `m(h)=M, m'(h)=0`. A simplest regular interior branch is

```
m_n(r) = M [(n+1)(r/h)^3 − 3(r/h)^(n+1)]/(n−2),  n>2.
```

The model's stationary branches can have bounded curvature, retain Kerr exterior/event-horizon parameters, and possess a second inner/Cauchy horizon. Large n squeezes that inner horizon towards the outer one even without a≈M; finite-n outer surface gravity remains Kerr-like. The n→∞ limiting mimicker develops a cusp at the matching surface and should not be carelessly called a smooth Killing horizon. Crucially, **stationary ansätze are not a demonstrated time-dependent collapse solution**, nor are inner-horizon stability or geodesic completeness established by curvature regularity. Article explicitly identifies dynamical matching as an outstanding challenge.

This is not a contradiction to the QTG theorem: Ovalle uses rotating axisymmetric Einstein-matter solutions and a stationary construction; Di Filippo considers spherical collapse for specific higher-curvature theories under hypotheses. Nor does an unchanged *stationary* exterior alone supply a new astrophysical observable. Compare perturbation dynamics and matching conditions.

## New quantum-constraint geometry comparator

**Alejandro Corichi**, *Group averaging and the geometry of physical quantum states: from normalizable solutions to solvable loop quantum cosmology*, first posted **16 Oct 2026**, revised **28 Oct 2026**, [arXiv:2610.24027v2](https://arxiv.org/abs/2610.24027), arXiv DOI [10.48550/arXiv.2610.24027](https://doi.org/10.48550/arXiv.2610.24027). **Unrefereed preprint; primary abstract and metadata reviewed, not full derivation independently checked.**

The distinction between normalizable **kinematical** vectors and constrained **physical** rays is essential: different seed rays can yield the same physical state after group averaging; their fibers exceed mere U(1) gauge orbits. Under orthogonal projection assumptions the physical ray space has a structured bundle of representatives; the more general distributional/loop quantum cosmology case uses test-space and clock choices and need not be Kähler isometric. This goes beyond GTOR's earlier blanket global-phase argument: not every representation ambiguity is "absolute phase", and projection itself may change the induced state-space metric. The physical Dirac observables must be identified.

**Concrete finite-dimensional reproduction:** `P=diag(1,1,0)`, seed `s=(1,0,0)` and `t=(1,0,1)/sqrt(2)`. Their preprojection Fubini–Study distance (convention `acos(|<s|t>|)`) is **π/4**, while normalized `Ps` and `Pt` have distance **0**. Seeds are not related by a global phase, yet represent the same constrained physical ray. This illustrative calculation validates only the toy geometry, **not** the general theorem or loop quantum cosmology results.

## New long-form geometric quantum hypothesis (low evidence weight)

**Alexey A. Kryukov**, *Classical and Quantum Dynamics in State Space: Projective Hilbert Space, Measurement, and Classical Emergence*, submitted **1 Nov 2026**, [arXiv:2611.02138](https://arxiv.org/abs/2611.02138), arXiv DOI [10.48550/arXiv.2611.02138](https://doi.org/10.48550/arXiv.2611.02138). **240-page unpublished monograph preprint; only primary abstract inspected**. Proposes stochastic unitary dynamics on projective Hilbert space and an account of Born-like transition probabilities from assumed Brownian/GUE increments with thermal, isotropy and coarse-graining conditions. This is a *model-dependent speculative derivation*, not a newly confirmed resolution of quantum measurement. Useful GTOR comparator: explicitly state its noise law and whether detector outcome distributions differ from standard Born statistics under independently specified noise.

## Two foundational anchors, not new physical discoveries

- **Abhay Ashtekar and Troy A. Schilling**, *Geometrical Formulation of Quantum Mechanics*, [arXiv:gr-qc/9706069](https://arxiv.org/abs/gr-qc/9706069), [arXiv DOI](https://doi.org/10.48550/arXiv.gr-qc/9706069). State rays form projective Hilbert space with Kähler (symplectic + metric) structure; Hamiltonian Schrödinger dynamics is a corresponding symplectic flow. This is a reformulation **equivalent to standard quantum mechanics**, not support for observable global phase.
- **Y. Aharonov and J. Anandan**, *Phase change during a cyclic quantum evolution*, **Phys. Rev. Lett. 58, 1593 (20 Apr 1987)**, [DOI 10.1103/PhysRevLett.58.1593](https://doi.org/10.1103/PhysRevLett.58.1593). A gauge-invariant nonadiabatic cyclic geometric phase is determined by a path in projective ray space, not the arbitrary common phase of a single vector. Important counterpoint to GTOR's pure-gauge `A=2πk du` on an interval (Audit 04): an open-path integral is *not* an intrinsic closed-loop holonomy.

## Contradiction / assumption ledger

| Question | Ordinary geometry / paper assumption | GTOR requirement / failure condition |
|---|---|---|
| Global phase or norm | Physical ray `[ψ]=[cψ]` for all nonzero complex `c`; normalized rays use Fubini–Study `d=acos(|<ψ|φ>|/(||ψ||||φ||))` | Demonstrate one **gauge-invariant, reference-specified** operational prediction beyond the standard quotient, or drop independent absolute-phase claims |
| Constrained physical states | Kinematical representatives project or group-average into fewer physical rays | Declare constraint `C`, physical inner product, choice of clock/test space, Dirac observables and what remains under projection |
| Spherical QTG collapse | NEC, `H'(ψ)>0`, suitable initial data/global hyperbolicity and trapped surface force null incompleteness | Give explicit GTOR action, verify each hypothesis; if absent, construct consistent causal Cauchy evolution rather than claiming automatic evasion |
| Stationary rotating interior | Ovalle: same Kerr exterior, controlled mass profile, typically inner horizon | Test eigenmodes, curvature measured in infalling frame, full maximal extension, energy conditions, dynamical matching; `m(h),m'(h)` alone do not establish them |
| Helical GTOR surface | Earlier independent result: `φ=v+2πku` removes `k`, intrinsic scalar curvature zero and signature Riemannian | Define real extra field/gluing/observable if winding is physical; never identify a Euclidean cone's slope with Lorentzian causality by fiat |

## Reproducible toy tests and preregistered failure checks

1. **Quantum projection/rephasing test** (`scripts/gtor_projection_and_kerr_audit.py`): verify `d_FS(s,t)=π/4`, `d_FS(Ps,Pt)=0`, and invariance under multiplying seeds by nonzero complex numbers. A nonzero reference-free prediction from common phase would contradict orthodox quantum mechanics and requires a fully defined measurement model; otherwise label it gauge artifact.
2. **Kerr interior roots** (same script): for M=1, a=0.7, h≈**1.7141428429**, `n=3` inner root≈**1.4743938303**, `n=20`≈**1.6676657696**; confirm `m(h)=M,m'(h)=0`. Illustrates approach to outer horizon; **not a dynamic stability check**.
3. **Next numerical falsifier:** evolve a smooth ingoing wave on specified GTOR interior + matter action, monitor inner trapping-horizon amplification and expansion `θ`; require finite affine continuation and a well-posed hyperbolic problem. Reject any candidate violating predeclared regularity, completeness or energy/causal requirements. Compare model-specific QTG hypotheses separately.
4. **Separation of research tracks:** use projective/constraint geometry to test quantum hypotheses, and explicit Lorentzian metrics to test black-hole hypotheses. No mapping between the two sectors can be claimed until an action and coupling are specified.

**Coverage/accounting:** Sources checked directly: entire open arXiv-rendered Di Filippo text, publisher open HTML Ovalle, arXiv record/abstract Corichi and Kryukov, arXiv abstract of Ashtekar–Schilling, APS abstract Aharonov–Anandan. Not exhaustive; Corichi/Kryukov proofs not fully checked, no paywalled reading claimed; independently reproduced tests are basic algebra/numerics, not empirical support. No cloud/account/spending changes.
