# GTOR — Research Audit 11 (13 November 2026 AEDT)

**Scope:** selective primary-source review and offline mathematical checks; deduplicated against the repository's GTOR research audits dated 4–12 November 2026. Four previously unlogged arXiv preprints (submitted 7–10 November, indexed in the current arXiv lists), two previously unlogged peer-reviewed foundational papers, and one new reproducible check. No claim that these were *published on 13 November*. The latest accessible arXiv category/new list was headed *Wednesday 11 November 2026*. No new experimental evidence for GTOR. No merge, spend or cloud provisioning.

## What actually changed

### A. Observable radiative content can survive a gauge transformation that substantially changes the intermediate perturbation field

**Yizhou Lu**, *The out-state of scalar-induced gravitational waves*, submitted **10 Nov 2026**, unreviewed preprint.
- [Primary full public HTML](https://arxiv.org/html/2611.13462v1)
- [arXiv abstract](https://arxiv.org/abs/2611.13462)
- arXiv identifier DOI: [10.48550/arXiv.2611.13462](https://doi.org/10.48550/arXiv.2611.13462) (arXiv page described DOI as pending DataCite registration at retrieval).

In second-order tensor perturbation theory for scalar-induced gravitational waves on FLRW with perfect-fluid background and equation of state **0 < w < 1**, the perturbation mixes free tensor radiation and a source-maintained, gauge-dependent response. Lu defines a gauge-invariant outgoing free-radiation amplitude from the **light-cone pole/residue**, with an LSZ-like source reduction. Under his theorem's assumptions, a field change Y -> Y+Q with *no tensor-frequency component* in the late-time shift has identical on-shell residue, so the radiative out-state and its spectrum agree. The theorem treats intrinsic gauges and excludes some special resonant configurations; it is not a statement about arbitrary redefinitions of a detector response.

A key mathematical assumption from Theorem IV.1 is a late-time expansion of the shift into terms exp(i delta_j x) b_j(x) with controlled polynomial/logarithmic envelopes and **delta_j != 0**. Then the Abel-regularized pole contribution from the shift vanishes:
  epsilon * integral_{x0}^{infinity} exp[(-epsilon+i delta_j)x] b_j(x) dx -> 0.
Under the paper's further regularity hypotheses this leaves the physical out-state unchanged.

**GTOR consequence:** Do not classify a changed apparent metric perturbation, field coefficient or intermediate gauge quantity as a new signal. First extract the detector-relevant, on-shell field content in one fixed physical protocol. This is an instructive *gravity-side analogue*, not an identity, of the projective-Hilbert-space gauge tests established in earlier audits.

### B. Black-hole mode amplitudes and phases depend on the *excitation source*, not only the black-hole geometry

**Alejandro Svyatkovskyy Kholyavka, Sayak Datta, Jose Antonio León Vega and Xisco Jiménez Forteza**, *Shaping black hole resonances II: Spectroscopic landscapes and interference in black hole ringdown*, submitted **9 Nov 2026**, unreviewed preprint.
- [Primary full public HTML](https://arxiv.org/html/2611.13075v1)
- [arXiv abstract](https://arxiv.org/abs/2611.13075)
- arXiv identifier DOI: [10.48550/arXiv.2611.13075](https://doi.org/10.48550/arXiv.2611.13075) (registration pending at review).

In a linearized Schwarzschild Regge–Wheeler model, QNM excitation factors B_n depend on geometry and mode, while source-dependent overlaps T_n depend on initial data: C_n = B_n T_n. For an oscillatory real Gaussian pulse at r0 with width sigma and carrier nu,

  Psi(x,0)=exp[-(x-r0)^2/(2 sigma^2)] cos[nu (x-r0)],
  Psi_hat(omega)/(sqrt(2 pi) sigma)
    = exp(i omega r0) exp[-sigma^2 (omega^2+nu^2)/2] cosh(sigma^2 omega nu).

The hyperbolic factor is coherent interference between the +/-nu branches. Evaluated at complex QNM frequencies, it changes *both amplitude and relative phase*. The paper validates its approximations numerically within its Schwarzschild test class; reported median amplitude errors include ~4% for the fundamental mode, larger for overtones. This is **not** evidence of a regular GTOR interior.

**Important model comparison:** Huang et al. (Audit 10) studied how changes to a *Frolov metric* affect scalar mode spectra; this paper holds the *Schwarzschild background* fixed and varies the excitation profile. A measured difference in ringdown amplitudes is therefore degenerate between source physics and metric physics unless excitation assumptions are controlled.

### C. A relational quantum clock must track post-selection cost and environmental memory

**Yakir Aharonov and Avshalom C. Elitzur**, *Time Running Backwards, and Where It Stops: Superoscillations of Proper Time and the Memory of the Environment*, submitted **7 Nov 2026**, unreviewed preprint.
- [Primary full public HTML](https://arxiv.org/html/2611.12971v1)
- [arXiv abstract](https://arxiv.org/abs/2611.12971)
- arXiv identifier DOI: [10.48550/arXiv.2611.12971](https://doi.org/10.48550/arXiv.2611.12971).

The controlled-evolution model has
  U(t) = sum_n |n><n| tensor exp[-i alpha_n H t/hbar],
  alpha_w = <f|alpha_hat|i>/<f|i>.
Conditioning on a rare final control outcome can yield an *effective* clock factor alpha_w outside the range of eigenvalues alpha_n for a limited energy/time window. In the authors' sealed-cell decay model, two-branch reversal signatures fail beyond the environment's memory time; many-branch extensions require bounded spectra and rapidly falling success probabilities. The full unconditioned ensemble remains consistent with ordinary unitary dynamics and thermodynamic accounting. The proposed practical protocols and the extrapolation to gravity are model-dependent, not demonstrated temporal reversal of an isolated real-world system.

**GTOR consequence:** Explicitly declare what is measured *unconditionally* versus in a heralded post-selected subensemble; include environment bandwidth, which-branch information, and sample efficiency. This directly extends the Page–Wootters/relational-clock comparison already documented in earlier audits.

### D. Horizon formation depends on the mass definition and rotation in proposed compactness tests

**Robin Croft, Gerardo García-Moreno and Paolo Pani**, *Challenging the hoop conjecture with spinning pancakes and baguettes*, submitted **9 Nov 2026**, unreviewed preprint.
- [Primary full public HTML](https://arxiv.org/html/2611.13266v1)
- [arXiv abstract](https://arxiv.org/abs/2611.13266)
- arXiv identifier DOI: [10.48550/arXiv.2611.13266](https://doi.org/10.48550/arXiv.2611.13266) (registration pending at review).

They construct and numerically solve rotating, axisymmetric dust *initial data* (including oblate/prolate configurations and rigid/differential rotation), then diagnose marginally outer trapped surfaces. A hoop-compactness rule stays within ~14% of its nominal coefficient for the tested rotating configurations when quasilocal mass includes rotational energy. Using irreducible horizon mass instead changes the spin trend; using total ADM mass for a horizonless matter surface can give apparent deviations up to ~25%, because energy outside the matter surface is counted. These are tests of a conjecture over a specified initial-data family, **not** a universal theorem of horizon formation, regular interiors or complete collapse evolution.

**GTOR consequence:** Any claim that relational compression generates/avoids horizons needs a chosen quasilocal energy, trapped-surface definition, angular momentum and evolution rule. A coordinate or graph-size threshold cannot substitute for the geometry and constraint equations.

## Foundational additions (not duplicates)

1. **André Heslot**, *Quantum mechanics as a classical theory*, *Phys. Rev. D* **31**, 1341 (published **15 March 1985**). Primary [publisher abstract](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.31.1341); DOI [10.1103/PhysRevD.31.1341](https://doi.org/10.1103/PhysRevD.31.1341). Geometric Hamiltonian descriptions can encode standard quantum theory and constrain generalized observable algebras. **Only the primary abstract was accessible**; publisher full text required authorization.
2. **Yakir Aharonov, Jeeva Anandan, Sandu Popescu and Lev Vaidman**, *Superpositions of time evolutions of a quantum system and a quantum time-translation machine*, *Phys. Rev. Lett.* **64**, 2965 (published **18 June 1990**). Primary [publisher abstract](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.64.2965); DOI [10.1103/PhysRevLett.64.2965](https://doi.org/10.1103/PhysRevLett.64.2965). Establishes the historical controlled-superposition baseline from which the 2026 post-selected clock proposal develops. **Primary abstract only**, not paywalled body.

Previously catalogued and **not counted as new**: Kibble (1979) and Anandan–Aharonov (1990 *Geometry of quantum evolution*), as well as Wootters, Ashtekar–Schilling, Bartlett–Rudolph–Spekkens, Pancharatnam, Samuel–Bhandari, Sontz, Page–Wootters and earlier GTOR black-hole theorems. This paper-level de-duplication was checked against the existing repo audit files, not all possible literature.

## Independent offline computation and limits

[gtor_audit11_gaussian_weakvalue.py](scripts/gtor_audit11_gaussian_weakvalue.py) is a public, reproducible NumPy/SciPy check, executed in a local CPU runtime without network/cloud calls:

- For the paper's Schwarzschild l=2 fundamental mode omega=0.3737-0.0890i (M=1) and pulses (sigma,nu)=(1,0.25),(3,0.2),(5,0.4), the *analytic source Gaussian Fourier transform* agrees with independent numerical integration with relative differences **1.94e-16, 2.09e-16, 1.31e-16**.
- For fixed geometry and mode, maximizing only that source-overlap magnitude on nu in [0,1] gives **nu approximately 0 at sigma=1**, and **nu=0.37359819 at sigma=5** versus mode real frequency 0.3737. This reproduces the finite-width source effect, not the full QNM problem or the paper's Leaver computations.
- With normalized two-branch control states |i>=(1,1)/sqrt(2), |f>=(1,-0.8)/sqrt(1+0.8^2), and alpha_hat=diag(0.99,1.01), the weak value is **alpha_w=0.91**, outside the eigenvalue interval. Control-only post-selection probability is **0.01219512**. Independent global rephasings of |i> and |f> leave alpha_w invariant. This checks elementary projective algebra, not the 2026 paper's nonlinear environment dynamics or physical feasibility.
- The program asserts these numerical identities and prints PASS. **No GTOR-specific predictions were computed.**

## Distinctions and outstanding contradictions

- **Not a contradiction:** Lu's invariant *on-shell gravitational-wave out-state* and the nonzero *relative phase* of the source-dependent QNM amplitude concern different physical and gauge objects. Neither measures an isolated quantum state's absolute global phase.
- **Potential identification failure:** a measured QNM overtone/amplitude difference can reflect source shape, interior/exterior metric, or detector transfer function. GTOR must compare models at fixed source/detector, then marginalize these nuisance parameters.
- **Time interpretation conflict to settle:** a post-selected effective alpha_w outside the clock-rate spectrum does not imply a physical globally backwards time coordinate or reversible dissipative universe. For GTOR's relational-clock proposal, declare the ensemble and measurement protocol before using such models.
- **Horizon inference:** a mass threshold using total ADM energy can be misleading when the chosen compactness uses matter-surface circumference. Standardize mass definitions before comparing collapse/horizon claims.
- **Major open obstacle:** GTOR 2.0 has proposed information/relations/flow primitives but still lacks a derived continuum Lorentzian metric with gravitational field equations and a quantum measurement protocol yielding a new gauge-invariant outcome. The above are benchmarks and rejection gates.

## Next tests (order and falsifiable criteria)

1. **Quantum/clock branch, low cost:** implement a two-qubit control+clock with post-selection and explicit environment. Calculate unconditioned and conditioned detection probabilities for varying clock separation, decoherence rate and readout success. Null test: all predictions coincide with standard controlled-unitary quantum mechanics. A novel GTOR prediction must be a preregistered, nonzero difference in a physical, gauge-invariant probability, not a redefined phase or selected subset.
2. **Source/metric identifiability:** fit synthetic ringdown using the fixed-geometry Gaussian-source family and independently a perturbed metric. Evaluate phase and amplitude jointly with injection recovery. Reject any claim that an apparent metric deviation is uniquely identified if source variations alone fit within simulated noise.
3. **Emergent-gravity branch:** derive from GTOR flow a Lorentzian metric/action and solve its constraint/evolution equations, then compute trapped surfaces with a specified quasilocal mass and spin, and regular-core geodesic/tidal/stability tests. If the selected GTOR parameters cannot reproduce the Schwarzschild exterior or have incomplete causal geodesics where regularity is claimed, reject that branch.
4. **Gauge/out-state check:** on a toy sourced wave equation, apply a field redefinition with no on-shell pole, verify identical extracted outgoing amplitude; introduce an on-shell component as a controlled counterexample. Reclassify any GTOR signal that disappears in the invariant out-state as a coordinate/field-definition artifact.

No exhaustive-coverage claim. Four recent full public arXiv HTML sources inspected, two historical publisher **abstracts only**. Four preprints are non-peer-reviewed. No paywalled text claimed. No owner approval/personal sign-in blocker was encountered; no alert needed. No cloud provisioning, spending, merge or change to task schedules.
