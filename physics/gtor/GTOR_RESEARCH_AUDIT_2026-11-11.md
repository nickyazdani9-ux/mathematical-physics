# GTOR — Research Audit 09 (11 November 2026 AEDT)

**Status:** Selective public-source review, deduplicated against Audits 01–08 (3–10 November); source texts checked 11 November. “New” means new to the GTOR ledger, not all first published today. All November 2026 arXiv works below are **preprints**, unless otherwise specified. These findings neither establish nor empirically support GTOR. No PR merge, no cloud provisioning, no spending.

## Material changes and source ledger

### A. Preferred time is part of a gravitational solution, not merely coordinates

**Antonio De Felice**, *Horizon-penetrating rotating black-hole configurations in VCDM*, submitted **9 November 2026**, arXiv:2611.11775v1, [full public HTML](https://arxiv.org/html/2611.11775v1), [abstract](https://arxiv.org/abs/2611.11775), arXiv DOI [10.48550/arXiv.2611.11775](https://doi.org/10.48550/arXiv.2611.11775), registration pending when checked.

The specified minimally modified gravity theory **VCDM** includes a preferred scalar clock/foliation, not just a Lorentzian metric. For a timelike preferred clock T, the paper defines
```
n_mu = -N partial_mu T,    N^-2 = -g^(mu nu)(partial_mu T)(partial_nu T).
K = nabla_mu n^mu.
```
On a constant-auxiliary branch, the metric obeys Einstein–Maxwell with an effective cosmological constant. The author constructs constant-mean-curvature (K=K0) preferred foliations crossing a rotating charged horizon, with hyperboloidal/expanding asymptotic ends. The *same* background four-metric permits distinct flux/shear data and therefore physically different preferred-foliation configurations within VCDM.

**Crucial distinction:** the author does **not** establish a nonsingular Kerr core or dynamically selected rotating-collapse outcome. The paper explicitly separates possible finite-proper-distance endpoints, infinite/trumpet ends, and stationary construction from evolution. Reassigning preferred clock T=t is *not* simply a passive coordinate change of T_old=t_BL. GTOR's emerging-clock claim should similarly identify whether its clock is gauge, a derived physical field, or extra degrees of freedom, and check its normalized timelike gradient at horizons.

### B. A model with a single-horizon nonsingular interior and explicit asymptotic dS end

**Daniel Areán, Suman Das, David Garcia-Fariña, Juan F. Pedraza**, *Emergent de Sitter Universes in Non-Hermitian Holography*, submitted **6 November 2026**, arXiv:2611.10121v1, [public HTML](https://arxiv.org/html/2611.10121v1), [abstract](https://arxiv.org/abs/2611.10121), arXiv DOI [10.48550/arXiv.2611.10121](https://doi.org/10.48550/arXiv.2611.10121), registration pending when checked.

This is a **planar asymptotically AdS Einstein–Maxwell–scalar** construction with a non-Hermitian boundary deformation, not an asymptotically flat isolated astrophysical black hole. The source uses nonconjugate scalar boundary data phi=(1-eta)psi and bar(phi)=(1+eta)psi; for |eta|>1 the bulk NEC can be violated. With m²=-2, quartic scalar potential and v>0, the dS attractor has
```
V(phi bar(phi)) = -2 phi bar(phi) - v(phi bar(phi))²,
H² = -1 + 1/(6v),  so H²>0 precisely for 0<v<1/6.
ds² ~ (H² tau²)^-1 [ c0² dt² + dx² - d tau² ] as tau→0-.
```
The existence of the attractor depends on the paper's phase diagram and boundary conditions; *not every* v in that interval necessarily gives the full desired solution. It has a single event horizon and a dS future boundary in the exhibited branch, and suggests boundary-correlator and complexity benchmarks. **No general proof of unitarity, stable realistic collapse, or universal singularity resolution follows.** Compare this carefully with Lan et al. (Audit 02): both exhibit a single horizon, but one has a finite-area static core and the other a dynamically expanding dS interior with non-Hermitian AdS boundary conditions. Not contradictory.

### C. Parameter-dependent gauge transforms can fake quantum sensitivity

**Wangjun Lu and Qing Yu**, *Attainable and unattainable quantum precision limits in ultrastrong light–matter coupling*, submitted **9 November 2026**, arXiv:2611.12471v1, [public HTML](https://arxiv.org/html/2611.12471v1), [abstract](https://arxiv.org/abs/2611.12471), arXiv DOI [10.48550/arXiv.2611.12471](https://doi.org/10.48550/arXiv.2611.12471), registration pending when checked.

In their exactly unitarily related family H_theta(g), the transformation itself depends on the *estimated coupling* g. Therefore an ordinary partial derivative of the transformed state gives gauge-dependent quantum Fisher information:
```
F_g(theta) = F_g(0) + 4 theta² Var(x X) - 8 theta W.
D_g = partial_g + i theta x X
```
The **covariant** derivative cancels the frame artefact and gives the same quantum Fisher information in every frame, under the paper's conventions. For their resonant harmonic model with omega=Omega=1, equation (9) reads
```
F_g(theta) = 2(1+g²)/(2+g²)² + 4(theta²-theta)/(2+g²).
```
At g=1 this gives F(0)=F(1)=4/9 and F(1/2)=1/9, a factor-four naive change although the gauge-covariant result is constant. This **does not** imply quantum measurement precision is fundamentally gauge dependent; it is an explicit warning about parameter-dependent identification of states/observables. GTOR must define and transform its connection, parameter derivative, clock calibration, and measurement apparatus consistently before claiming physical phase/normalization information. Do not confuse this with globally rephasing a state.

### D. A foundational asymptotic interior in an *alternative action* (2019)

**Christian G. Böhmer and Franco Fiorini**, *The regular black hole in four dimensional Born–Infeld gravity*, **Classical and Quantum Gravity 36, 12LT01 (2019)**, journal DOI [10.1088/1361-6382/ab1e8d](https://doi.org/10.1088/1361-6382/ab1e8d), [free original manuscript arXiv:1901.02965](https://arxiv.org/abs/1901.02965) (public full text read; publisher full text not claimed).

The paper specifies a tetrad/torsion-based modified gravitational action, rather than solving Einstein's vacuum field equations:
```
I = (16 pi G)^-1 ∫ e f(T) d⁴x,   f(T)=lambda[sqrt(1+2T/lambda)-1].
lambda=-|lambda|,  a_c=2/sqrt(|lambda|).
```
Its Kantowski–Sachs black-hole interior has an asymptotically limiting areal radius a_c>0 rather than a collapsing 2-sphere. At the limiting geometry ds²=dT²-dX²-a_c² dOmega² the paper reports Ricci scalar R=-2/a_c² and Kretschmann K=4/a_c⁴ in its signature conventions; the physical region reaches that end only at infinite proper time. The authors claim timelike/null geodesic completeness in the accessible spacetime. The inaccessible extension a<a_c still contains a **repulsive singular center** in their discussion; avoid misreporting all analytic continuations as regular.

This is a strong comparator for GTOR's proposed asymptotic coordinate map, because the asymptote follows from a **specified field equation** rather than reparameterization alone. It uses different dynamical assumptions from Bueno et al.'s quasi-topological vacuum dS-core restriction (Audit 07), so there is no direct contradiction. The new numerical question is *stability* and whether the teleparallel tetrad and physical boundary conditions are acceptable.

### E. Experimental comparator: geometric phase is inferred relationally, not an absolute ket phase

**Kui Zhao et al.**, *Hidden Coherence and Controlled Quasihole Motion in a Fermionic Laughlin State on a Superconducting Quantum Processor*, submitted **7 November 2026**, arXiv:2611.10687v1, [public HTML](https://arxiv.org/html/2611.10687v1), [abstract](https://arxiv.org/abs/2611.10687), arXiv DOI [10.48550/arXiv.2611.10687](https://doi.org/10.48550/arXiv.2611.10687), pending registration when checked.

They report a simulated three-electron filling-1/3 Laughlin state using **seven superconducting qubits**, full-state fidelity 0.662, target-sector conditional fidelity 0.886 and an *effective* geometric-phase signature from a reconstructed dominant-eigenstate trajectory, not directly a free-standing absolute U(1) phase. It is preliminary experimental research; finite-size, tomography and mixed-state reconstruction limit extrapolation. A useful GTOR benchmark would reproduce phase signatures via closed trajectories in projective space, including noise and tomography, without choosing an unobservable overall wavefunction phase.

## Small independent mathematical controls (not reproductions of the papers)

See [script](scripts/gtor_audit09_null_checks.py). It checks: (i) the quadratic QFI formula at g=1; (ii) the sign window H²(v) for the proposed dS asymptote; (iii) a *Schwarzschild Boyer–Lindquist clock* toy showing divergence of the normalized foliation's acceleration invariant while the Schwarzschild Kretschmann curvature remains finite as r→2M; (iv) the BI a_c and K endpoint algebra for |lambda|=9. These are elementary substitutions only, not a VCDM/AdS/teleparallel solution, empirical confirmation, or perturbation simulation.

## Model comparison and falsification queue

- **GR metric alone vs physical clock:** De Felice's action carries auxiliary and preferred-time fields; GTOR currently has only abstract relational update F:X→X and has not derived a Lorentzian metric or physical clock. Proposed rejection: show no normalized invariant clock observable survives allowed gauge transformations; if so, treat it as coordinate convention. Computational test: construct a finite causal/clock model, track N^-2, expansion, acceleration and geodesic completeness under different foliations and coordinate charts.
- **Interior endpoints:** Lan et al. finite-area static core; Böhmer–Fiorini teleparallel infinite-proper-time a_c end; Areán et al. non-Hermitian planar AdS-to-dS future end. Each requires different actions, topologies, energy conditions, boundary data and horizon structures. None licenses a GTOR interior prediction without an explicit action. Proposed test: solve one chosen GTOR field equation, derive the areal radius and affine length to every boundary, curvature/parallel-propagated tidal tensor, NEC, and scalar perturbation operator. Reject a claim of complete regularity upon any causal geodesic incompleteness or diverging physical curvature in the candidate.
- **Gauge/normalization:** Projective Born probabilities (Audit 08), geometric holonomy (Audit 08) and *covariant* quantum Fisher information (today) form three independent null controls. Proposed test: parameter-dependent rephasing, reference-frame changes and normalization, jointly transforming state/operator/gauge connection; reject an alleged new phase effect if Delta P or an operational Fisher metric vanishes after covariant comparison.
- **Asymptotic observable:** For a fully specified interior model, calculate scattering/quasinormal spectra with explicit boundary conditions and ask whether two geometries give distinguishable spectral shifts. The shadow-screening examples from Audit 06 show why an exterior-only null result may lack discriminating power.

**Priority:** Specify a mathematically complete small GTOR model with an action or discrete update, defined gauge equivalences and a measurement rule; then perform the invariant-clock/gauge checks *before* expensive numerical relativity. Comparisons are conditional, and no measurable departure from standard quantum theory or GR has yet been derived.

**Dedup/provenance:** Earlier audits covered Lan, Bueno, Hayward, Shao, Pang/Peng, Sontz, Powell, Wootters, Avdoshkin, Zhang–Gao, Ageev–Zueva and more; none of the five papers above appeared in Audits 01–08 when the text files were checked. Public full HTML was read for the four new November arXiv sources; Böhmer–Fiorini free arXiv full text was read. No paywalled full text was used. Journal DOI for Böhmer–Fiorini is bibliographically verified from metadata, although the publisher site blocked the full text.
