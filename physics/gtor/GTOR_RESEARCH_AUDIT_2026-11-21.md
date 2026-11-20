# GTOR — Research Audit 19 (21 November 2026 AEDT)

## Scope, source chronology, and deduplication

Selective review of public arXiv **Friday, 20 November 2026** new listings in gr-qc, quant-ph and hep-th, plus author abstracts and public HTML. **Listing and initial submission dates differ.** Of the six papers below, two were submitted 19 November, three were submitted 18 November, and one was submitted 4 November. All are unreviewed preprints; no claim that all were newly written on Friday or independently replicated. arXiv DOIs of the form 10.48550/arXiv.ID may still be marked *pending registration* on arXiv.

Compared the six arXiv IDs against the actual 17 existing audits dated **4–20 November** in GitHub branch `research/gtor-audit-20261104`. A regex-based scan found 84 distinct previously mentioned arXiv identifiers (including contextual references); **none of the six IDs was present**. This is identifier-level deduplication, not an exhaustive literature search.

Core issue: *metric regularity, characteristic causality, observational identifiability and effective-action independence are four different tests*. In quantum theory, a phase of matter is not the overall complex phase of a state vector, and distinct microscopic fermionic strategies can require a stronger operational equivalence criterion than ordinary qudit models.

## Six new-to-ledger primary preprints

### 1. Causal failure despite regular metric and hyperbolic equations — new high-priority screening test

**J. M. Romero**, *Superluminal propagation in the magnetic NED reconstruction of the extremal Dymnikova black hole*. **Submitted 19 November 2026.** Primary https://arxiv.org/abs/2611.26560 ; public text https://arxiv.org/html/2611.26560v1 ; DOI https://doi.org/10.48550/arXiv.2611.26560 .

An explicit magnetic nonlinear-electrodynamics reconstruction supports the background
`f(r)=1-2GM(1-exp(-r^3/P^3))/r`. On the magnetic branch `I=F_{mu nu}F^{mu nu}>0`, the extraordinary photon cone has, in local orthonormal coordinates,
`omega^2=k_r^2 + A(u) k_perp^2; A(u)=(3u-5)/2, u=(r/P)^3`.
The ordinary polarization remains metric-luminal. The extremal horizon satisfies `exp(u*)=1+3u*`, with `u*=1.9038136944`, `r_e/P=1.2393904597`. The extraordinary transverse speed becomes superluminal when `u>7/3`, i.e. outside `r_lum/P=(7/3)^(1/3)=1.3263524026`. At `r=1.5P`, `v_perp=1.6007811`.

Crucial distinctions: (i) its principal system is hyperbolic at finite exterior radii but **not uniformly hyperbolic as r→infinity**; (ii) it lacks the Maxwell weak-field limit; (iii) the standard DEC already fails outside the extremal horizon; (iv) positivity of the reduced radial potential was checked on a finite grid, *not proved globally*, and coupled gravitational/electromagnetic modes were not solved. The author regards the reconstructed source as not causally acceptable relative to the physical metric in the asymptotic region. This is a **specific counterexample to the inference “regular metric + local hyperbolicity ⇒ physically acceptable causal model,”** not a claim all regular-black-hole models fail.

**GTOR test:** for any GTOR matter+gravity action, derive the full principal symbol and physical characteristic cones, verify cone containment at every exterior radius (or explicitly defend a different physical causal metric), verify asymptotic uniform well-posedness, weak-field limit and all constraint equations. Curvature scalars alone will miss this failure.

### 2. A ringdown alternative that does not modify the black-hole interior

**Jahed Abedi, Niayesh Afshordi**, *Which black hole modes ring freely? Horizon regularity, the forced ringdown transient, and GW250114*. **Submitted 19 November 2026.** Primary https://arxiv.org/abs/2611.26591 ; HTML https://arxiv.org/html/2611.26591v1 ; DOI https://doi.org/10.48550/arXiv.2611.26591 .

For a separated outgoing horizon solution with `p=omega_I/kappa+i(omega_R-m Omega_H)/kappa`, bounded first derivatives require `Re(p)>=1` (apart from exceptional monomials); smooth solutions select the horizon ladder `omega_j=m Omega_H - i j kappa`, `j=1,2,...`. **This does not change the retarded QNM pole spectrum.** The paper models the actual observed mode as a forced convolution,
`y_n(u)=-i g_n integral_{-infinity}^u exp[-i omega_n(u-v)] I(v) dv`,
becoming a constant-amplitude freely damped QNM only when the source ceases or decays sufficiently rapidly.

Applied to GW250114 with particular templates and gate choices, the authors find Bayes factors roughly **2–30** favoring a fundamental-QNM + first-horizon-mode fit over fundamental + first-overtone on selected overlapping windows. Those windows are correlated, not a global significance claim; waveform injections within general relativity reproduce important fit features. The result is a **proposed interpretation** that requires comparison against alternative source and waveform models, not an independently verified detection of a novel horizon mode or quantum gravity.

**GTOR test:** any proposed high-overtone interior signature (see Audit 14) must be fit jointly with time-dependent excitation and horizon-driven transient models; only residuals surviving calibrated GR source/response models can constrain new interior dynamics.

### 3. Spinfoam effective dynamics and removable curvature corrections

**Muxin Han**, *Pure integral formula of spinfoam amplitude and curvature corrections in effective theory*. **Submitted 18 November 2026.** Primary https://arxiv.org/abs/2611.25033 ; HTML https://arxiv.org/html/2611.25033v1 ; DOI https://doi.org/10.48550/arXiv.2611.25033 .

For a fixed Lorentzian EPRL spinfoam triangulation in a large-spin and small Barbero–Immirzi-parameter regime, the paper trades internal spin sums for a continuous integral suited to stationary phase. Integrating auxiliary variables produces the Regge action plus corrections **quadratic in deficit angles** `epsilon_h`. Two sources contribute: holonomy terms in the new action, and non-Regge-area fluctuations. The paper argues the leading long-wavelength quadratic Regge sector yields a massless spin-2 mode. It also notes that an equation-of-motion-squared part of the quadratic holonomy correction can be removed by field redefinitions, whereas a Weyl-squared type contribution from non-Regge area fluctuations is a distinct issue.

**GTOR test:** build the discrete GTOR action, derive its continuum limit and physical on-shell observables, and screen candidate quadratic corrections for field-redefinition redundancy. A new connection coefficient or off-shell term alone is not a physical prediction. This is a semiclassical comparator, **not proof that GTOR has reproduced Einstein gravity**.

### 4. Same quantum-network correlations need not identify the same fermionic physics

**Fatemeh Moradi Kalarde, Luis Muñecas Tomás, Marc-Olivier Renou**, *Self-testing in Fermionic Networks*. **Submitted 18 November 2026.** Primary https://arxiv.org/abs/2611.25074 ; HTML https://arxiv.org/html/2611.25074v1 ; DOI https://doi.org/10.48550/arXiv.2611.25074 .

The authors distinguish qudit-local isometries from **fermionic parity-preserving** local isometries. Certain correlations unique in a qudit self-test admit two inequivalent fermionic implementations; their three-party rewiring network, incorporating all wiring contexts, resolves this ambiguity under their assumptions. This is standard fermionic quantum theory with local parity/superselection constraints, not evidence for absolute global wavefunction phase.

**GTOR test:** an alleged new operational quantum degree of freedom must be compared against the appropriate conventional equivalence group (global U(1) rephasing, local gauge, parity-preserving isometries and actual POVMs), not only qubit projective coordinates. A network-context effect may be entirely conventional physics.

### 5. A method for certifying Hamiltonian phase transitions, not phase-of-wavefunction measurement

**Yuanxi Liu, John Preskill, Hsin-Yuan Huang**, *Certified Discovery of Quantum Phase Diagrams*. **Submitted 18 November 2026.** Primary https://arxiv.org/abs/2611.25064 ; HTML https://arxiv.org/html/2611.25064v1 ; DOI https://doi.org/10.48550/arXiv.2611.25064 .

For smooth geometrically local Hamiltonian families, the authors propose a quantum algorithm identifying transition regions through spectral-gap witnesses, supplying certificates under **specified preparability and gapped-region assumptions**. The resource and error guarantees depend on gap resolution; the method does not certify arbitrary uncomputable physical theories. “Quantum phase” here is a many-body phase of matter, not an observable global phase of a pure state.

**GTOR test:** once a finite-size `H_GTOR(lambda)` is specified, benchmark critical parameters and gap closing against a matched standard Hamiltonian and finite-size extrapolations, rather than declaring differences from plotted eigenvector phases.

### 6. Conformal Schwarzschild regularization separates null and massive-particle probes (older submission, newly found)

**Shavkat Karshiboev et al.**, *Spinning particle dynamics around a non-singular Schwarzschild black hole in conformal gravity*. **Submitted 4 November 2026; listed on the 20 November gr-qc page.** Primary https://arxiv.org/abs/2611.24944 ; HTML https://arxiv.org/html/2611.24944v1 ; DOI https://doi.org/10.48550/arXiv.2611.24944 .

The metric is `g_tilde=S(r) g_Schwarzschild`, with `S(r)=(1+L^2/r^2)^2`. For positive `S` away from the centre, null-geodesic trajectories are conformally invariant (up to affine parametrization), while massive and spinning-particle trajectories need not be. The authors investigate Mathisson–Papapetrou–Dixon motion with Tulczyjew spin condition, finding conformal-scale-dependent bound orbits and ISCO shifts. Their curvature-finiteness and completeness statements depend on this particular conformal model; conformal invariance of null curves is not invariance of all detector or matter physics.

**Independent additional spinless benchmark:** for `M=1`, `A=S(1-2/r)`, `C=S r^2`. Thus the null impact parameter is `b^2=C/A=r^2/(1-2/r)`, giving the photon sphere at `r=3` and `b_crit=3 sqrt(3)` **independently of L**. But for spinless timelike circular orbits, differentiating the conserved squared angular momentum produces the ISCO condition
`r^5-6r^4 + L^4(4r^2-21r+30)=0`.
Numerical outer ISCO roots are `r=6.0000000` (L=0), `5.96282025` (L=1), `5.37332929` (L=2). These are our spinless controls, **not** the paper's spinning-particle computations.

**GTOR test:** compare null shadow, massive-matter orbit/ISCO and spin-induced precession simultaneously. If a proposed modification is *only a conformal rescaling*, a changed null shadow under fixed mass/conventions is not a valid novelty claim.

## Foundational, bibliographically verified controls not previously entered in the 17 audit files

1. **J. Ehlers, F. A. E. Pirani, A. Schild**, *The geometry of free fall and light propagation* (original **1972**, republished **2012** in General Relativity and Gravitation), DOI https://doi.org/10.1007/s10714-012-1353-4 ; public reprint https://bpfoundations.org/plans/ehlers2012.pdf . Causal light cones supply conformal structure; freely falling trajectories supply projective/connection structure; compatibility constrains a Lorentzian metric. It is a framework for reconstructing geometry, not dynamics derived from GTOR quantum states.
2. **Tullio Regge**, *General relativity without coordinates*, *Il Nuovo Cimento* **19**, 558–571 (**1961**), DOI https://doi.org/10.1007/BF02733251 . Original discrete-simplicial curvature/action benchmark; publisher preview/bibliography consulted, not a claim of accessible paywalled full text.
3. **S. Kamefuchi, L. O'Raifeartaigh, A. Salam**, *Change of variables and equivalence theorems in quantum field theories*, *Nuclear Physics* **28**, 529–549 (**1961**), DOI https://doi.org/10.1016/0029-5582(61)90056-6 ; bibliographic/abstract https://inspirehep.net/literature/40210 . Provides the standard on-shell equivalence warning for local field redefinitions; validity requires the theorem's conditions, not arbitrary transformations. Publisher full text was not claimed.

## Reproducible mathematical null controls

Ran the offline Python **standard-library** test script on 21 November 2026: [`gtor_audit19_causality_conformal_ringdown.py`](scripts/gtor_audit19_causality_conformal_ringdown.py).

- Magnetic NED: nontrivial root `u*=1.9038136944`; `r_DEC/P=1.1006424163 < r_e/P=1.2393904597 < r_lum/P=1.3263524026`; `v_perp(1.5P)=1.6007810594`. This confirms threshold algebra from the printed equations, **not** the paper's full electromagnetic PDE or all field equations.
- Conformal Schwarzschild: `b_crit/M=5.1961524227` at `r=3M` for all four sampled `L`, while spinless ISCO radii depend on `L` as above. This verifies the null-versus-timelike probe distinction, not the MPD spin model or global regularity.
- Forced damped-oscillator analogue with source `I(t)=exp(-kappa t)`, `kappa=0.25`, `t=100`: for `gamma=0.09`, free/driven exponential weight `8.8861105e6`; for `gamma=0.40`, `3.0590232e-7`. This shows why a rapidly damped overtone can remain *source-dominated* while a slower fundamental is free; it is a toy model, **not** a GW250114 waveform reanalysis.

All three limited control families passed locally. No GTOR-specific field equation or detector prediction exists in this audit.

## Priority tests and precise rejection criteria

1. **Causal principal-symbol gate (highest priority):** specify GTOR action; derive linearized coupled modes on the candidate metric; compute every physical characteristic speed versus radius and check (a) well-posedness, (b) containment relative to chosen physical cone, (c) uniform asymptotics, (d) weak-field limit. Reject a proposal that predicts an unaccounted superluminal physical branch or fails its own well-posedness assumptions. Reuse Romero's `v_perp^2=(3u-5)/2` as a negative-control regression.
2. **Three-probe geometry identifiability:** hold `M`, exterior boundary normalization and source/detector assumptions fixed. Compare shadow impact parameter, massive ISCO and spin-precession observables under conformal or throat modifications. For *pure conformal Schwarzschild*, a shadow-size difference is a failure of the null-control calculation.
3. **Forced-vs-free ringdown discrimination:** add explicit plunge-source convolution and horizon ladder to the null GR model before interpreting near-peak overtone fits as novel interior physics. Fit independent gate windows without treating overlapping windows as independent detections. Compare predicted transfer functions after allowing nuisance source parameters.
4. **Discrete action and gauge test:** derive a continuum metric or principal symbol from GTOR's actual degrees of freedom; compute on-shell amplitudes and remove field-redefinition-equivalent corrections. For quantum measurements, test projective and parity-preserving invariances before proposing a genuinely new phase readout.
5. **Optional certified finite-Hamiltonian benchmark:** only once `H_GTOR(lambda)` is concrete, test spectral-gap closures with conventional and GTOR Hamiltonians against the algorithm's preparability and accuracy assumptions.

## Execution and governance boundary

No full-paper replication, GTOR empirical validation, production merge, owner sign-in, cloud provisioning, spending or private data transfer. Keep draft PR #2 open for independent review. In this research subtree the five expected canonical project files `INDEX.md`, `SPEC.md`, `CONSTRAINTS.md`, `MILESTONES.md` and versioned `STATE.md` were **not found at their conventional paths**; do not claim the research subtree is fully governance-compliant or overwrite unseen templates. Preserve this audit as an append-only provenance record while the canonical-document location/templates are reconciled separately. No new owner approval request was necessary to perform this authorized public-source research and draft-branch documentation.
