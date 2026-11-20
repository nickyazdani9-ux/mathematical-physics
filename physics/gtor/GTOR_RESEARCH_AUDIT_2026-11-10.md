# GTOR 2.0 — Research Audit 08 (10 November 2026, AEDT)

**Scope and provenance.** Selective review of public primary papers, checked against the GTOR working draft and Audits 01–07 (3–9 November). 'New' means new to this research ledger, not all first published today. Sources classified below as publisher abstract, public full HTML, or preprint. **No verified empirical support for GTOR, no merge, no provisioning, no spending.** The branch is an unmerged research draft.

## Material change A: recovering curvature from discrete causal relations has a quantified noise barrier

**Nauman Ibrahim and Luca Bombelli**, *The Weyl Tensor for Gravitational-Wave Causal Sets: Thickness of Geodesic Analogs and Geodesic Deviation*, **submitted 6 Nov 2026** (listed in arXiv's Mon 9 Nov gr-qc new feed), **unreviewed preprint**; full public [arXiv HTML v1](https://arxiv.org/html/2611.09056v1), [abstract](https://arxiv.org/abs/2611.09056), arXiv DOI [10.48550/arXiv.2611.09056](https://doi.org/10.48550/arXiv.2611.09056) (DataCite registration listed as pending when checked).

Assumptions: a causal set is a *locally finite partially ordered set* (not an arbitrary undirected graph); a Poisson-sprinkled reference Minkowski/linearized gravitational-wave spacetime; longest maximal chains are analogs of timelike geodesics; the target curvature is a Weyl-tensor component inferred from geodesic deviation. With idealized distance extraction and artificially high wave strain h0=10^-2, the authors estimate **at least ~10^8 elements** to overcome chain-tube thickness noise with their three-point estimator, versus their then-simulated ~10^6 (and ~4e6 on a large node). It is a *method-specific estimate*, **not** a no-go theorem for discretization or GTOR.

Useful explicit error accounting (paper's leading-order equation, when m and n points in two sample clusters are reused for m*n distances):
```
Var(mean-distance) = (1/m + 1/n)*delta^2/3 + O(delta^4/s^2).
```
Counting the m*n pairwise distances as independent would underestimate variance because they share the same points. **GTOR consequence:** shortest-path distance and an asserted continuum geometry do not suffice: specify causal order, dimension, appropriate timelike/null proxies, a convergent curvature estimator, and statistical error bars. Distinguish the 1+1 horizon proxy of Eichhorn–Gamito–Stokes (Audit 04) from this harder 3+1 curvature task.

## Material change B: all ordinary Born probabilities may already be expressible on projective space

**Stephen Bruce Sontz**, *All Quantum Probability viewed in Complex Projective Geometry*, first submitted **17 May 2026**, revised **23 May 2026**; **preprint, full public HTML read** [arXiv:2605.17578v2](https://arxiv.org/html/2605.17578v2), [abstract](https://arxiv.org/abs/2605.17578), DOI [10.48550/arXiv.2605.17578](https://doi.org/10.48550/arXiv.2605.17578).

The author proposes to express the quantum-probability rules directly through the Riemannian/Kähler geometry of complex projective Hilbert space, including a projective projection theorem. The paper is a **mathematical reformulation/proposal**; not evidence of new measured physics or proof that any alternative geometry reproduces every quantum prediction. It explicitly constructs
```
CP(H) = (H \ {0}) / C*,   equivalently S(H) / U(1),
d_FS([psi],[phi]) = arccos(|<psi|phi>|/(||psi|| ||phi||)).
```
For pure-ray transition probabilities, the exact standard relation is `P(phi|psi)=cos^2(d_FS)`. Rescaling either vector by its *own* nonzero complex number leaves this probability invariant. This directly sharpens GTOR's burden: normalization/global-phase quotient discards choice of representative, not measurable Born information. A new geometric model must exhibit a projectively well-defined probability, independent clock/measurement definition and a quantitative departure from QM, or explicitly remain a mathematically equivalent reformulation.

**Foundational control (not new research):** **J. P. Provost and G. Vallée**, *Riemannian structure on manifolds of quantum states*, **Commun. Math. Phys. 76, 289–301 (September 1980)**, DOI [10.1007/BF02193559](https://doi.org/10.1007/BF02193559). **Publisher abstract only** (subscription content blocked). Authors formulate an induced state-manifold metric and relate state geometry to quantum fluctuations. This precedes the recent GTOR terminology; do not attribute its full proof to an abstract review.

## Material change C: a genuinely observable geometric phase needs holonomy, not an arbitrary ket phase

**Andrew Powell**, *A Geometric Journey Through the Aharonov–Bohm Effect: Tonomura's Toroid, Pseudoscalar Phase, and Quantum Holonomy in Clifford Algebra*, **submitted 6 Nov 2026**, **preprint; full public HTML read** [arXiv:2611.09311v1](https://arxiv.org/html/2611.09311v1), [abstract](https://arxiv.org/abs/2611.09311), DOI [10.48550/arXiv.2611.09311](https://doi.org/10.48550/arXiv.2611.09311) (DataCite registration marked pending when checked).

In Euclidean Clifford algebra Cl(3,0), central pseudoscalar I satisfies I^2=-1, so a U(1) element may be represented by exp(I alpha). For linked paths and shielded flux, the gauge-invariant result is an **interference-path holonomy**:
```
U_Gamma = exp(I q Phi_B / hbar),  Delta theta = (q/hbar)*oint_Gamma A.dx (mod 2pi).
```
The algebra representation does not create a separately measurable absolute global phase. It describes the existing AB effect with explicit topological linking and an external physical flux/source. Foundational experimental/theoretical baseline: **Y. Aharonov and D. Bohm**, *Significance of Electromagnetic Potentials in the Quantum Theory*, **Phys. Rev. 115, 485 (1 August 1959)**, DOI [10.1103/PhysRev.115.485](https://doi.org/10.1103/PhysRev.115.485), **publisher abstract verified**.

**GTOR contradiction check:** previous Audit 04 found the manuscript's helix connection `A = 2*pi*k du = d(2*pi*k*u)`, globally exact on the stated cylinder. Thus `oint A=0` for every closed cycle; `int_0^1 A=2*pi*k` is an *open-path* integral, not invariant holonomy. No AB-like topological effect follows from that connection without an independent physical flux/bundle/gluing condition. Simply changing from complex notation to Clifford/quaternionic notation does not solve that.

## Material change D: causal completeness and experienced tidal force are independent criteria

**Xiaotian Zhang and Sijie Gao**, *Geodesic completeness, curvature singularities and infinite tidal forces*, **Chinese Physics C 50(9), 095103 (2026)**, publicly online **15 Sep 2026** per journal, DOI [10.1088/1674-1137/ae7960](https://doi.org/10.1088/1674-1137/ae7960); full [public journal HTML](https://hepnp.ihep.ac.cn/article/doi/10.1088/1674-1137/ae7960) and [arXiv:2507.04616](https://arxiv.org/abs/2507.04616) read. **Published research for specified spherically symmetric metrics**, not a theorem for arbitrary interiors.

They construct/examine cases with formally extendible geodesics but divergent curvature/tidal response, and a contested wormhole throat whose metric is not well defined. The magnitude of measured tidal acceleration also depends on the observer's velocity; an ultrarelativistic boost may magnify finite curvature components. Geodesic deviation is governed by `D^2 xi^a / d tau^2 = -R^a_{bcd} u^b xi^c u^d` (curvature sign convention dependent). Their stronger claims about extended particles are model-dependent physical interpretations; do **not** infer an absolute prohibition on passage for every object from point-geodesic algebra alone.

GTOR must check **at least**: scalar invariants, causal geodesic extendibility, *parallel-propagated* tidal components, and perturbative evolution. This adds a physical observer criterion to Audit 02/07 regularity and the asymptotic-throat tests. Original philosophical control: **Robert Geroch**, *What is a Singularity in General Relativity?*, **Ann. Phys. 48, 526–540 (1968)**, DOI [10.1016/0003-4916(68)90144-9](https://doi.org/10.1016/0003-4916(68)90144-9), bibliographic/abstract-level only; the Geroch singularity-definition problem is not new to this survey.

## Optional but distinct quantum-interior comparator

**Dmitry S. Ageev and Anastasia N. Zueva**, *Reflecting Baths and Black Hole Interior Islands in JT Gravity*, **submitted 5 Nov 2026**, preprint, full public [arXiv HTML v1](https://arxiv.org/html/2611.08597v1), [abstract](https://arxiv.org/abs/2611.08597), DOI [10.48550/arXiv.2611.08597](https://doi.org/10.48550/arXiv.2611.08597) (pending DOI registration). In 1+1-dimensional **Jackiw–Teitelboim gravity** with reflecting finite non-gravitating baths and massless Dirac matter, boundary-reflection correlations change entropy and quantum-extremal-surface/island saddle dominance. **This is not** a claim that physical 3+1 black-hole curvature singularities disappear. Compare only to a GTOR model with explicit matter entropy, bath and QES prescription; do not equate entropy information recovery with local curvature regularity.

## Independently reproduced offline baseline (stdlib Python)

[Reproducible local script](scripts/gtor_audit08_checks.py) tested 2,500 pairs of independently complex-scaled three-component quantum vectors: max `|P - cos^2(d_FS)| = 1.67e-16`; max independently complex-rescaled `Delta P = 1.11e-15`. **Exactly a standard-QM null test**, not evidence supporting GTOR.

For m=n=8 reusable sample points with independent uniform displacements delta=0.3 and separation s=10, 25,000 seeded Monte Carlo trials gave std(mean pair distance)=`0.087114` versus the covariance-aware prediction `0.086603`; a wrong independent-pairs estimate would give `0.030619`, under by a factor `sqrt(8)=2.828`. This is a *separate one-dimensional correlated-sampling toy*, **not** a reproduction of the causal-set Weyl simulation or its 10^8 estimate.

## Falsification queue and recommended next research

1. **Quantum gate:** implement the projective Born geometry `P=cos²(d_FS)`, independently rephase and rescale each ket, and add explicit physical flux and a closed-loop measurement when asserting an AB-style phase. Reject GTOR's claimed intrinsic global-phase/helix-k observable if it disappears under permissible rephasing, coordinate transformation or reference changes. A new model must specify one gauge-invariant experimental `Delta P` and uncertainty after calibrating clocks.
2. **Emergent-causality gate:** construct a finite causal order (not graph-only metric), reproduce longest-chain thickness and the *correct covariance-aware* variance scaling with increasing number of independent points. Compare null/timelike proxies and known smooth curvature before extrapolating to 3+1 Weyl recovery; use small offline simulations only. A null result at limited N does not falsify a continuum theory without a power/noise model.
3. **Interior gate:** give an action and Lorentzian GTOR metric with two independent radial functions and proper source/boundary conditions. Integrate null/timelike affine parameters; compute scalars plus parallel-propagated `R^a_{bcd}u^b u^d`, monitor charged/rotating extensions and finite-energy perturbations. Register horizon data, singularities and any energy-condition violations. If incompleteness or diverging physical curvature persists, reject any *strong* complete-and-regular-interior claim for that candidate.
4. **Distinct sectors:** do not extrapolate an AdS/JT island result, a flat-background causal-set signal, or a static 4D regular metric into one universal claim. Specify assumption mappings and a separate quantitative observable.

**Evidence / dedup statement.** Compared with seven prior audits: Lan, Peng, Di Filippo, Bueno, Shao, Hayward, Ovalle, Engelhardt–Nagar, Liang–Li, Tello-Ortiz, Heydari, Kibble/Wootters/Bartlett, Berry/Avdoshkin and existing helix tests are *prior findings*, not new entries here. Newly catalogued primary works are Ibrahim–Bombelli, Sontz, Powell, Zhang–Gao, Ageev–Zueva; foundational controls Provost–Vallée and Aharonov–Bohm (and Geroch primary bibliography). Source full HTML: those new arXiv papers plus Zhang–Gao journal, except Provost–Vallée, AB, and Geroch only journal metadata/abstract-level. The survey is not exhaustive, peer review is not presumed for preprints, and no paywalled access is asserted.
