# GTOR — Research Audit 17 (19 November 2026 AEDT)

## Provenance and scope

Selective source-first review of the **Wednesday 18 November 2026** arXiv gr-qc and quant-ph new-listings, their primary abstracts, and public arXiv HTML texts. Listing date is **not** submission date: the seven selected new-to-ledger manuscripts were originally submitted on 4, 16 or 17 November. All seven identifiers were checked absent from the 15 prior GitHub research audits (4–18 November; 69 distinct prior arXiv IDs found). All seven are **preprints**, with arXiv/DataCite DOIs sometimes marked pending registration, not peer-reviewed or independently confirmed discoveries. No claim of full paywalled access. No new GTOR field action, spacetime metric or physical effect has been independently established.

## Seven newly catalogued primary preprints

### 1. A regular one-way black-hole interior follows from a gravitational action — but probes its EFT limit

**Francesco Del Porro, Alessia Platania, Samuele Silveravalle**, *Hidden wormholes from an action principle in D=4*. Submitted **16 November 2026**; new-listed 18 November.
Primary: https://arxiv.org/abs/2611.21712 ; public full HTML https://arxiv.org/html/2611.21712v1 ; arXiv DOI https://doi.org/10.48550/arXiv.2611.21712 .

The authors solve numerically an explicit four-dimensional diffeomorphism-invariant gravity effective action
```
S = (16 pi G)^(-1) integral d^4x sqrt(-g) [R + lambda*C^3/(8 m_Pl^4) + ...]
C^3 = C_mu nu ^rho sigma C_rho sigma ^alpha beta C_alpha beta ^mu nu.
```
They find a region of static, spherical solutions with a single event horizon, an interior minimum areal radius (one-way wormhole throat), and an anisotropic Kantowski–Sachs cosmological future. For the cosmological patch,
```
ds^2 = -d tau^2 + a_parallel(tau)^2 d rho^2 + a_perp(tau)^2 d Omega_2^2,
a_perp ~ exp(H_0 tau), a_parallel ~ exp(alpha H_0 tau/2),
alpha ~ 7, H_0 ~ 1.5 lambda^(-1/4) [paper's Planck conventions].
K -> (alpha^4/4 + 2 alpha^2 + 12)*H_0^4.
```
The authors give geodesic-completeness and curvature-finiteness arguments for the exhibited geometry; the absence of an inner horizon removes the *specific* classical Cauchy-horizon mass-inflation mechanism. **Critical limit:** at the throat, null convergence is violated and the paper explicitly states this signals a breakdown of the *truncated perturbative EFT regime*. Full UV completion and nonlinear stability are not established, and their numerical scan of couplings/masses is limited.

**Independent audit flag:** their Eq. (17) defines `E = a_parallel^2 d rho/d sigma` but then prints `(d tau/d sigma)^2 = E/a_parallel^2 - epsilon`. Metric normalization gives **E^2/a_parallel^2 - epsilon** with this definition. This appears to be a notational/algebraic typo in the power of E (their asymptotic conclusion of infinite affine length still follows). Check any updated version before citing as an erratum. Near the throat, `f(r) ~ -f_T (r-r_T)` implies `r(tau) ~ r_T+f_T tau^2/4`; verified independently in the script.

**GTOR comparison:** unlike the phenomenological Simpson–Visser geometry in Audit 16, this is obtained from a specified field action, but not a controlled UV-complete proof of regularity. Require an action, explicit validity bounds and completeness tests from GTOR itself.

### 2. Wormholes with matter satisfying NEC in nonmetricity gravity, not vacuum GR

**Ismael Ayuso, Mariam Bouhmadi-López, Che-Yu Chen, Konstantinos Dialektopoulos**, *New traversable wormholes without invoking exotic matter in f(Q) from the freedom of different connections*. Submitted **4 November**, new-listed 18 November.
Primary: https://arxiv.org/abs/2611.21658 ; full HTML https://arxiv.org/html/2611.21658v1 ; DOI https://doi.org/10.48550/arXiv.2611.21658 .

The authors vary an `f(Q)` symmetric-teleparallel gravity action over the metric **and independent flat, torsionless connection**, with a prescribed Ellis–Bronnikov wormhole geometry and an anisotropic fluid. Two allowed connection families arise. For regular `f_QQ`, constant `Q` follows; for one connection a varying `Q` requires singular `f_QQ` at a finite `Q`. Specific quadratic/exponential choices permit the **matter-sector** NEC/WEC to hold. For constant Q their Eqs. (51–54) yield `rho + p_r = -2 f_Q a^2/r^4`, which is nonnegative if `f_Q <= 0` with their conventions; the connection/gravity sector supports the throat. This does not repeal the standard GR focusing result, nor establish that the full *effective* geometry obeys the Einstein-theory null-convergence condition. Some connection branches are constrained by throat regularity; same background metric does not imply identical dynamics.

**GTOR test:** calculate the full metric/connection field equations and matter NEC, effective NEC, connection regularity and propagating degrees of freedom separately; do not label an identical chosen background metric a physical novelty.

### 3. Field-theoretic quantum reference frames use relational observables, not bare global phase

**Ivan Agullo, Fabio M. Mele**, *A Proposal for Quantum Reference Frames in Quantum Field Theory and their Experimental Exploration*. Submitted **16 November**, new-listed 18 November.
Primary: https://arxiv.org/abs/2611.21740 ; public HTML https://arxiv.org/html/2611.21740v1 ; DOI https://doi.org/10.48550/arXiv.2611.21740 .

For a *massless scalar with a real-valued shift symmetry* `Phi(x) -> Phi(x)+lambda I`, localized field modes act as necessarily noisy/mixed quantum reference frames in typical field states. This is a **global shift symmetry, not U(1) wavefunction rephasing and not a gauge redundancy**. For smeared observables with integrals `bar f` and `bar f_R`, a relational field value is
```
F_{Phi_f|R}(u) = Phi_f - (bar f/bar f_R)*Phi_fR + u*(bar f/bar f_R)*I.
```
The offset term encodes measured frame outcome. Their Eq. (6) gives a Gaussian frame-resolution suppression factor `exp[-epsilon_R^2 (bar f/bar f_R)^2/2]` for relativized Weyl operators; both frame noise and correlations affect predictions. A proposal to test with quantum simulators is not a completed experimental result.

**GTOR test:** operationally define reference, POVM, resolution, physical invariant and probability. Reject phase differences that vanish upon frame transformation or originate solely from finite reference precision.

### 4. Complex interference observables require at least three outcomes in this setup

**Wenzheng Dong, Alexander Yosifov, Jinzhao Sun**, *Complex quasiprobabilities as measurable temporal interference*. Submitted **17 November**.
Primary: https://arxiv.org/abs/2611.22643 ; full HTML https://arxiv.org/html/2611.22643v1 ; DOI https://doi.org/10.48550/arXiv.2611.22643 .

For complete projective measurement families `{A_n}`, `{B_m}`, the extended Kirkwood–Dirac object is `Q_{nm,n'm'} = Tr(A_n B_m A_n' B_m')`. Their theorem: if either family has two or fewer outcomes, its imaginary part vanishes for all unitary evolutions; with three or more outcomes each, imaginary values generically occur. In a studied spin-one chain, those imaginary values distinguish dynamics that certain conventional infinite-temperature OTOCs do not. Experimental readout uses calibrated ancilla covariances and repeated randomized preparations, not an observable absolute global phase.

**GTOR test:** use a fully specified 3-level POVM/dynamics as a baseline; require any GTOR correction to alter an experimentally accessible interference statistic and survive vector rephasing. Our 3-dimensional Fourier-basis null control gives `Q = -1/18 + i sqrt(3)/18`; its imaginary part is not beyond standard QM.

### 5. Gravitational mode instability is sector-dependent in Kerr-like metrics

**Pedro Henrique Croti Siqueira, Rodrigo Dal Bosco Fontana, Maurício Richartz**, *Gravitational Mode Instabilities in Kerr-like Black Holes*. Submitted **16 November**, new-listed 18 November.
Primary: https://arxiv.org/abs/2611.21735 ; public HTML https://arxiv.org/html/2611.21735v1 ; DOI https://doi.org/10.48550/arXiv.2611.21735 .

Using continued fractions and quasinormal boundary conditions for a parametrized Kerr-like geometry, they report exponentially growing spin-weight `s=-2` modes at sufficiently high spin and deviation, while their electromagnetic modes remain damped. Some deformed configurations retain horizons even with `a/M>1`. This is **a stability analysis of a particular homogeneous Teukolsky-like operator on a phenomenological metric**, not a proof of a consistent modified-gravity action or full nonlinear instability. Their GW250114 comparison is a conditional spectrum fit, not direct detection of a modified interior.

**GTOR test:** first derive the perturbation system and constraint sector from the GTOR action; include gravity, matter, axial/polar coupling and boundary conditions. Do not infer gravitational stability from a scalar or electromagnetic test.

### 6. Horizon boundary conditions enter wave-optics lensing beyond point-mass geometry

**Alice Garoffolo, Gianmassimo Tasinato**, *Beyond the point lens: radial structure and the horizon in wave-optics gravitational lensing*. Submitted **17 November**.
Primary: https://arxiv.org/abs/2611.22618 ; public HTML https://arxiv.org/html/2611.22618v1 ; DOI https://doi.org/10.48550/arXiv.2611.22618 .

The authors calculate low-frequency **scalar-wave** lensing corrections with Newtonian dressing and radial kernels. When the exterior potential is singular at short distance, the missing input is an *inner boundary condition*; Schwarzschild horizon matching produces an absorptive contact contribution reproducing the horizon-area cross section. A nonabsorbing lens of equal mass differs. The simple thin-lens diffraction integral lacks this horizon condition. This is a controlled theoretical scalar proxy for wave optics, not a fully validated gravitational-wave measurement.

**GTOR test:** given the same exterior mass and controlled source, compare absorptive versus regular reflective/transfer boundary conditions and track the complex lens amplification; distinguish actual inner-boundary sensitivity from exterior environmental effects.

### 7. Scalarized black holes can change shadow and photon-ring indices differently

**Hao Yang**, *Shadows and Images of Black Holes with Purely Nonlinear Scalarization*. Submitted **17 November**.
Primary: https://arxiv.org/abs/2611.22586 ; public HTML https://arxiv.org/html/2611.22586v1 ; DOI https://doi.org/10.48550/arXiv.2611.22586 .

Numerical Einstein–scalar–Gauss–Bonnet black-hole families with a quartic exponential scalar coupling show, on the sampled radially stable branches, shadow-radius reductions as large as **about 21.7% at equal ADM mass**. The photon-orbit instability index can change nonmonotonically even where shadow size changes monotonically. Observed image morphology depends on emissivity and disk dynamics. This **does not contradict** the throat-blind-shadow metric of Audit 16, where the external lapse was held fixed; here the physical solutions modify exterior geometry.

**GTOR test:** compare matched ADM mass and full redshift/lapse/optical metrics, calculate independent photon-orbit indices, marginally stable disk regions and ray-traced images under common emission priors.

## Two historical foundations (new-to-ledger)

- **Marc H. Goroff and Augusto Sagnotti**, *The ultraviolet behavior of Einstein gravity* (1986), *Nuclear Physics B* **266**, 709–736; DOI https://doi.org/10.1016/0550-3213(86)90193-8 . Peer-reviewed two-loop Einstein-gravity divergence benchmark motivating the essential higher-curvature operator; historical journal metadata/abstract verified, not paywalled full text.
- **John G. Kirkwood**, *Quantum Statistics of Almost Classical Assemblies* (1933), *Physical Review* **44**, 31–37; DOI https://doi.org/10.1103/PhysRev.44.31 . Foundational phase-space quasiprobability work, not evidence that imaginary four-projector quasiprobabilities measure global wavefunction phase. APS abstract/metadata consulted, not full paper.

## Assumption-aware comparison and contradictions

| Claim family | Governing assumptions | What is established within model | What cannot be inferred |
|---|---|---|---|
| Hidden one-way throat | GR + Weyl-cubed EFT, numerically selected static solutions | An interior finite throat, asymptotic future and no inner Cauchy horizon for exhibited examples | UV-complete nonsingularity or generic nonlinear stability |
| f(Q) traversable wormhole | prescribed Ellis–Bronnikov metric, independent symmetric-teleparallel connection, anisotropic matter | choices where **matter** obeys NEC/WEC | standard Einstein-gravity NCC or absence of all geometric exoticity |
| Quantum relational frame | massless scalar shift symmetry and finite-resolution localized POVM | shift-invariant relative observables with measurable frame-noise cost | detectability of arbitrary global phase |
| Complex KD phase | >=3 outcome families, calibrated probe | nonzero imaginary *temporal interference* within conventional QM | novel GTOR dynamics |
| Parametric Kerr-like QNMs | selected background and Teukolsky-like operator | growth in gravitational sector of sampled backgrounds | universal instability of all deviations or stability inferred from EM modes |
| Wave-optics lens | specified exterior + boundary matching | inner absorption/contact response not fixed by mass alone | an observed GTOR interior signature |
| Scalarized shadow | scalar–Gauss–Bonnet exterior and emission assumptions | matched-mass shadow/index differences | contradiction with a different fixed-lapse throat geometry |

## Independent reproducible algebra (not whole-paper replications)

Ran `scripts/gtor_audit17_relational_throat_controls.py` offline with Python 3 standard library, 19 Nov AEDT. Results:
- shift-invariant scalar relational observable change **0** under a nonzero global shift; finite frame visibility `exp[-epsilon_R^2 (bar f/bar f_R)^2/2] = 0.805801078869` for the selected parameters;
- 3-outcome Fourier-basis four-projector quasiprobability `Q = -0.055555555556 + 0.096225044865 i`, unchanged by independent eigenvector rephasings to **1.55e-17**; an example two-outcome measurement gives imaginary part **0**;
- near-throat local solution `r(tau)=r_T+f_T tau^2/4` satisfies `(dr/dtau)^2=-f(r)` with max residual **8.33e-17**;
- checking conserved momentum and normalization in the hidden-wormhole asymptotic geodesic equations detects the apparent printed `E` versus `E^2` inconsistency noted above (test `E=2` makes ratio **2**).
These are self-contained null controls and a possible text correction, not complete reproductions of the papers, numerical solutions of field equations, or empirical validation of GTOR.

## Falsifiable next tests (ordered)

1. **Choose a physical GTOR quantum observable** `P_GTOR(m|preparation, reference, POVM, t)`. Run rephasing, normalization and reference-change invariance tests. Measure whether `Delta P = P_GTOR - P_standard` survives calibration, including frame-resolution uncertainty, and whether it differs from ordinary 3-level KD temporal interference. If not, the proposal is representation-only.
2. **Write an explicit GTOR gravitational action/constraint system**. Solve its field equations before imposing a regular interior; compute invariant curvature, geodesic completeness, parallel-propagated tidal stress and effective versus matter-sector energy conditions. Report EFT small-parameter bounds: higher-curvature terms becoming order-one requires a UV error budget, not merely a regular numerical metric.
3. **Do coupled linear stability** of the resulting background: gravity, scalar/gauge/connection fields, and physical constraints; compare overtone spectra under identical sources and boundary conditions. A scalar or EM sector alone is insufficient.
4. **Cross-observable geometry control:** hold ADM mass and emissivity fixed; vary the radial metric/lapse separately, measure shadow, photon-ring Lyapunov exponent and wave-optics absorptive amplification. Test whether a GTOR interior-only perturbation leaves exterior observables invariant as expected, or induces a well-defined nonzero correction.

**Status:** Existing research branch and draft PR only. No merge, deployment, cloud provisioning, spending or owner approval requested. New scientific findings are selected preprint claims; the null script verifies limited algebra only.