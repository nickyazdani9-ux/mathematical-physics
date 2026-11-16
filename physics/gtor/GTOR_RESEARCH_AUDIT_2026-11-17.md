# GTOR — Research Audit 15 (17 November 2026 AEDT)

## Scope and provenance
Selective primary-literature review, **not exhaustive**. The arXiv gr-qc, quant-ph and hep-th *new* feeds examined on 17 November AEDT were headed **Monday 16 November 2026**. The five new preprints below were **submitted 12–13 November UTC**; submission and arXiv listing dates must not be conflated. I inspected each abstract plus publicly available arXiv HTML, not paywalled journal full text. The 13 prior audits, dated 4–16 November on the existing draft research branch, were fetched and searched for identifier deduplication. All five IDs are new to that ledger. Aharonov–Bohm (1959), DOI 10.1103/PhysRev.115.485, was **already cited in the old audits**, so it is NOT counted as new foundational material.

**Status:** arXiv preprints are author-reported theoretical results, not peer-reviewed facts or empirical validation of GTOR. An arXiv-issued DOI is shown on each arXiv record, generally with *DataCite registration pending* at inspection time. GTOR does not currently have in this audit a demonstrated Lorentzian action, independently derived metric, or gauge-invariant new experimental prediction.

## Five newly catalogued primary preprints

1. **Markus B. Fröb and Albert Much, _Local Quantum Metrology: Precision Limits for Observers in QFT_** (submitted **13 Nov 2026**).
   - Primary https://arxiv.org/abs/2611.17948 ; public full text https://arxiv.org/html/2611.17948v1 ; DOI https://doi.org/10.48550/arXiv.2611.17948 (arXiv-issued, registration indicated pending).
   - **Assumptions:** normal state families on arbitrary von Neumann algebras, including type-III local algebras of relativistic QFT; the accessible measurement algebra is specified. Those local algebras generally have neither trace nor density matrices.
   - **Claim established in paper under assumptions:** defines Uhlmann/Bures SLD Fisher information without a density matrix and proves an operational collective-measurement Cramér–Rao bound, sharp at the algebraic measurement level. Its key definition is
     `F_M^SLD(theta)=8 lim_(epsilon->0) [1-sqrt(P(omega_theta,omega_(theta+epsilon)))]/epsilon^2`,
     where `P` is Uhlmann's algebraic transition probability. The paper discusses which POVMs can be physically implemented; sharpness over the algebra is not a universal hardware guarantee.
   - **GTOR relevance:** pure-state/projective-Hilbert-space Fubini–Study geometry is an appropriate finite-dimensional baseline but cannot automatically be treated as the complete observable geometry for a *local QFT region*. If GTOR makes local-clock claims, specify the local observable algebra and state family before computing sensitivities. No observation of absolute global phase follows.

2. **Samuel Valach and Mile Vrbica, _The Spectral Anatomy of Complex-Time Singularities_** (submitted **13 Nov 2026**).
   - Primary https://arxiv.org/abs/2611.18520 ; public full text https://arxiv.org/html/2611.18520v1 ; DOI https://doi.org/10.48550/arXiv.2611.18520 (arXiv-issued, registration indicated pending).
   - **Assumptions:** large-N holographic boundary correlators; planar AdS-Schwarzschild with boundary dimension `d >= 3`; minimally coupled probe scalar; complexified radial WKB at high frequency/overtone. Their near-singularity exponent satisfies `0<gamma<1/2` for the modeled class.
   - **Result:** the asymptotic spectrum has distinct momentum scalings `k ~ omega^gamma` (inner regime, sensitive to the singular vicinity) versus `k ~ omega` (outer regime, complex bulk away from it), which control different singularities of `G(t,x)`. The paper quotes a small-momentum form `omega_n ~ (pi/Z)[n+(nu-1)/2+F(k/n^gamma)]` in its conventions.
   - **Important overlap:** the paper explicitly acknowledges substantial independent overlap with Jia–Rangamani, arXiv:2611.13261. Treat the two as related theoretical approaches, **not** two independent experimental confirmations. This also overlaps conceptually with the existing audit's Della Rocca–Pani–Sanna high-overtone paper, but the backgrounds, boundary conditions, and observables are different.
   - **GTOR test:** if a GTOR metric and perturbation equation exist, derive radial WKB/QNM data at both scalings, and check complex-time branch locations against direct correlator numerics. Do not equate a complex-time holographic singularity with a naked astrophysical curvature singularity.

3. **Paolo Creminelli, Michael Mancini and Warin Patrick McBlain, _Microcausality Bounds on Dispersion Laws_** (submitted **12 Nov 2026**).
   - Primary https://arxiv.org/abs/2611.17407 ; public full text https://arxiv.org/html/2611.17407v1 ; DOI https://doi.org/10.48550/arXiv.2611.17407 (arXiv-issued, registration indicated pending).
   - **Assumptions:** local relativistic QFT even if boosts are spontaneously broken by the state; tree-level non-dissipative, weakly coupled excitations with analytic dispersion `omega(k)`.
   - **Result:** microcausality requires analytically continued retarded poles to obey `Im omega <= |Im k|` (units `c=1`). Their analyticity/typically-real-function approach constrains higher-derivative dispersion coefficients using Toeplitz positivity and the finite radius of convergence. The allowed higher-order corrections become tightly constrained as `c_s -> 1`.
   - **GTOR test:** derive the linearized dispersion relation of each propagating GTOR degree of freedom; test the retarded propagator, pole bounds and hyperbolicity. This is a **conditional** constraint, not a no-go theorem for all Lorentz-violating or dissipative systems. Truncated EFT polynomial dispersion must not be extrapolated past its convergence radius.

4. **Jialun Zhao, Antonio De Felice and Shinji Tsujikawa, _Quasinormal modes of dyonic axion black holes with nonlinear electrodynamics_** (submitted **13 Nov 2026**).
   - Primary https://arxiv.org/abs/2611.18117 ; public full text https://arxiv.org/html/2611.18117v1 ; DOI https://doi.org/10.48550/arXiv.2611.18117 (arXiv-issued, registration indicated pending).
   - **Assumptions:** static electrically/magnetically charged black holes with a massless axion and specified quartic nonlinear electromagnetic interactions. The authors solve **five coupled dynamical perturbation sectors** with spectral and independent matching methods.
   - **Result:** axion coupling breaks selected gravitational/EM parity isospectrality; nonlinear-electrodynamic coefficients modify damping. At strong-coupling examples with `gbar=10, eta=0.1`, they report purely imaginary growing `l=2` frequencies `Omega = i*0.06720, i*0.13343, i*0.17560` for `Q/M=0.6,0.7,0.8`, respectively. Since their time convention is `exp(-i omega t)`, these are exponentially **unstable**, despite satisfying some local high-frequency consistency conditions.
   - **GTOR consequence:** even ghost/gradient-free local dynamics and bounded core curvature do not prove mode stability. If GTOR couples additional matter, compute the **complete coupled** perturbation spectrum, not only test-scalar QNMs. The paper's strong-coupling numbers are not GTOR predictions.

5. **Gabriel P. Martins, Herman Batelaan and Pablo L. Saldanha, _Optically-induced electric Aharonov-Bohm effect_** (submitted **12 Nov 2026**).
   - Primary https://arxiv.org/abs/2611.17435 ; public full text https://arxiv.org/html/2611.17435v1 ; DOI https://doi.org/10.48550/arXiv.2611.17435 (arXiv-issued, registration indicated pending).
   - **Assumptions:** electron double-slit interferometer using a pulsed, nonlinear optical crystal; quasi-DC field generated by optical rectification is concentrated between the paths, with small local field on the electron trajectories.
   - **Status:** **experimental proposal with a theoretical numerical design**, not evidence of a completed electric AB measurement under negligible local force.
   - **Operational phase:** the authors give `phi_AB=-(e/hbar) integral (V_a-V_b) dt`, yielding a fringe modulation proportional to `cos^2(k_y*y_c + phi_AB/2)`. The difference between a **fringe displacement** and a common residual-field momentum-kick **envelope displacement** is experimentally discriminating.
   - **GTOR consequence:** the reference is a closed-loop relative phase (gauge-invariant holonomy), not measurable absolute global phase. A GTOR helix connection `A=2*pi*k du` remains an exact one-form on a global real coordinate `u` and has zero integral around closed loops. A nontrivial topology/connection/physical coupling must be specified before claiming observable holonomy.

## Newly logged foundational peer-reviewed comparison

**David B. Malament, _The class of continuous timelike curves determines the topology of spacetime_, Journal of Mathematical Physics 18, 1399–1404 (1977).**
DOI: https://doi.org/10.1063/1.523436 ; bibliographic record https://philpapers.org/rec/MALTCO-34 . The mathematical result, under the stipulated smooth Lorentzian and causal-distinguishability assumptions, connects timelike curves/causal order with topology and conformal structure; it does **not** determine the metric's absolute conformal scale or derive dynamics from a generic graph. Compare with the previously logged Hawking–King–McCarthy theorem and Churilov's nine-arrival-time reconstruction. Only accessible metadata/abstract summaries were consulted; no claim of obtaining inaccessible paywalled full text.

## Cross-paper comparison and unresolved tensions

| Branch | Starting data | Invariant/test | Still needed for GTOR |
|---|---|---|---|
| Finite-dimensional quantum rays | nonzero vector psi modulo C* | normalized projector `P=psi psi†/(psi†psi)`; Born probabilities `Tr(PE)`; Fubini–Study/QFI | a new physical observable surviving phase and norm changes |
| Local QFT metrology (Fröb–Much) | normal state on local von Neumann algebra, potentially type III | algebraic fidelity `P(omega1,omega2)` and local measurement Fisher bounds | an explicit local algebra, state family and realizable measurement |
| AB interferometry (Martins et al.) | two coherent paths, physical electromagnetic coupling | relative path phase and fringe displacement | non-exact topology/connection and physical coupling beyond coordinate winding |
| Causal reconstruction (Malament) | already smooth distinguishing Lorentzian causal structure | topology/conformal structure | metric scale, volume/clock data, action and existence of Lorentzian limit |
| Microcausality (Creminelli et al.) | local relativistic dynamics, controlled analytic excitations | retarded Green-function pole/dispersion constraints | GTOR's principal symbol and retarded response |
| Holographic interior (Valach–Vrbica) | AdS planar horizon + probe scalar + large-N WKB | asymptotic complex frequencies and analytic boundary correlators | GTOR geometry, perturbation operator, consistent boundary conditions |
| Axion/NED black holes (Zhao et al.) | Einstein+axion+NED charged background | all coupled mode frequencies; `Im omega > 0` signals growth | a global stability test, not only local regularity |

**Reconciliation, not contradiction:** A local QFT lacks a density-matrix description of its full type-III algebra, while finite qubit projective geometry remains correct for a finite subsystem. Different spatial momentum/overtone limits can encode different interior information without contradicting earlier regular-black-hole QNM claims. Microcausality constraints concern support/analyticity of retarded propagation; they do not, by themselves, rule out axion-induced global growing modes. Observable AB holonomy is a relative phase associated with paths and gauge structure, not absolute global phase.

**Potential falsifiers:** (1) If a claimed GTOR quantum correction vanishes upon nonzero complex rescaling/rephasing or after reference calibration, there is no new observable. (2) If the linearized retarded propagator violates causal support in an otherwise claimed local Lorentzian EFT regime, the model or its assumed axioms must change. (3) If a proposed regular black-hole background admits any `Im omega>0` with the chosen time convention, its linear-mode stability claim fails. (4) If the emergent event data cannot reconstruct consistent metric/volume/clock information under refinement, the continuum-emergence claim fails. The third and fourth tests require a **GTOR-specific** model not yet provided here.

## Independent, reproducible null-model checks

The offline standard-library script [scripts/gtor_audit15_phase_causality_controls.py](scripts/gtor_audit15_phase_causality_controls.py) passed locally before commit (RED/GREEN-style null controls; **not** a numerical reproduction of the five papers):
- Pure qubit `psi(theta)=(cos(theta/2),sin(theta/2))`: exact `F_Q=1`. With theta-dependent nonzero scaling `exp((0.2+5i)theta)`, projective `F_Q=1` again; finite-difference Uhlmann/Bures limit is consistent.
- On a punctured-plane loop, the locally flat `A=(1/3)dphi` has `integral A=2*pi/3 = 2.094395102393` radians. Adding `dchi` with `chi=0.3 sin(2phi)+0.2 cos(3phi)` leaves it unchanged (exact part `-1.441e-16` radians). Equal-arm fringe readout gives `P_+=0.25`. A globally single-valued exact potential on a simply connected region instead has zero loop phase.
- Illustrative **Hermitian** two-mode matrix with detuning `Delta=0.2`: halving coupling from `0.01` to `0.005` reduces off-resonant shift by ratio `3.99254191` (~quadratic); at degeneracy `Delta=0, g=0.01` the eigenfrequency splitting is `0.02` (~linear). This illustrates avoided-crossing scaling, **not** the source paper's non-Hermitian QNM spectrum.

## Priority actions (no infrastructure changes)

1. **Highest:** formalize a GTOR quantum observable as a functional of a physical state, measurement/reference and experimental control. Benchmark with pure-state projective QFI, then test local-algebra fidelity if a QFT interpretation is intended. State an explicit nonzero difference from standard quantum mechanics after all gauge transformations.
2. Derive a candidate GTOR continuum **Lorentzian action, principal symbol, constraints and retarded propagator** from its own ingredients; require stable, microcausal propagation and reproducible causal metric reconstruction instead of assuming a metric.
3. Only after step 2, define neutral/charged interiors and coupled perturbations; compute geodesic extension, curvature, stress-energy and full QNM/mode stability. Maintain distinct astrophysical versus AdS/holographic observables.
4. Where a helical connection is claimed to be physical, specify topology, gauge transformations, coupling to matter, reference-clock procedure, and an explicit interference/holonomy observable. Test fringe movement against force-induced envelope movement.

**Execution:** add research and offline controls to existing unmerged GTOR draft branch only. No new repository, duplicate Linear issue, cloud provisioning, paid access, spending, merge or external notification requiring owner approval. No genuine owner-action blocker encountered.
