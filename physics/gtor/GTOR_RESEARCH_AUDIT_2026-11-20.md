# GTOR — Research Audit 18 (20 November 2026 AEDT)

## Scope, chronology, provenance and deduplication

Selective primary-source review of arXiv **Thursday 19 November 2026** new listings in gr-qc, quant-ph and hep-th, with paper-specific arXiv abstracts and public HTML. A **listing date is not a submission date**: the seven selected papers were submitted **17 or 18 November 2026**. They are seven new-to-ledger arXiv preprints, including one formal scientific Comment; none is treated as a confirmed result of GTOR. Their arXiv DOIs use the DataCite 10.48550 prefix and may be pending registration.

Read the actual 16 prior audit Markdown files dated 4–19 November on the existing research branch and extracted 76 distinct prior arXiv identifiers. **None of the seven IDs below occurred there**. This checks ID-level deduplication, not exhaustive conceptual uniqueness. The 2025 source paper challenged in item 2 is included as contextual comparison, **not** counted as a new preprint.

Scientific distinction: standard quantum mechanics identifies pure states with complex projective rays and implements measurement by operators/POVMs; a novel phase observable requires an *operational path/reference effect beyond ordinary calibrated quantum geometry*. The gravity branch must verify **all field equations on the full proposed domain**, not just a regular-looking metric or reduced ansatz. A spacetime light-cone tail is not acausal propagation.

## Seven primary new-to-ledger manuscripts

### 1. Empirical test of path-sensitive hidden gauge frames; no effect at reported precision

**Li-Yu Shen, Lukas Scarfe, Carrie Rodriguez, Alessio D'Errico, Andrea Aiello, Ebrahim Karimi**, *Experimental constraints on gauge hidden-variable models in Bell tests*. **Submitted 17 Nov 2026**.

Primary: https://arxiv.org/abs/2611.23418 ; public HTML https://arxiv.org/html/2611.23418v1 ; DOI https://doi.org/10.48550/arXiv.2611.23418 .

- Polarization-entangled photon experiments compare direct versus closed wave-plate paths calibrated to implement the **same final measurement effect**, unlike an ordinary relative Berry-phase interferometer. The local hidden-frame story is a **hypothesis**; it does not itself reproduce Bell-nonlocal quantum correlations or evade Bell's theorem.
- The phenomenological correlation model is `E_gamma(alpha,beta)=V cos[2(alpha-beta)-delta_gamma]`, with `delta_gamma=kappa*Omega_gamma (mod 2pi)`; `kappa` is not fixed by Berry geometry. Their mixed-context signed witness changes by `Delta S = 2V sin(delta_gamma/2) sin(2beta-pi/4+delta_gamma/2)`, with their setting conventions.
- Measured direct and looped correlation curves were indistinguishable at experimental precision, even for four loops with distinct enclosed solid angles. Source reports Bell violation for the direct reference, but looped contextual witnesses are *not* separate standard CHSH experiments. This does **not** supply a model-independent bound on counterfactual mismatch `p_gamma`, nor a quantified exclusion limit on all possible `kappa`.
- **GTOR implication:** the most direct existing falsification template for phase-path memory. Specify GTOR's predicted `kappa`, loop dependence, preparation and POVM calibration. If the model predicts a nonzero effect above noise but none is observed, it fails in this implementation. If it predicts only standard relative Berry phase, the experiment neither supplies novelty nor directly disproves that standard effect.

### 2. A concrete challenge to a purported exact nonsingular black hole

**Mustapha Azreg-Aïnou, Marco A. A. de Paula**, *Comment on "Exact, nonsingular black holes from a phantom DBI field as primordial dark matter"*. **Submitted 17 Nov 2026**.

Primary: https://arxiv.org/abs/2611.23597 ; full public HTML https://arxiv.org/html/2611.23597v1 ; DOI https://doi.org/10.48550/arXiv.2611.23597 .

Context/original claim: **Tausif Parvez and S. Shankaranarayanan**, *Exact, non-singular black holes from a phantom DBI Field as primordial dark matter*, public version https://arxiv.org/abs/2511.14047 (submitted **18 November 2025**, revised April 2026; now a journal article). This is **a scientific disagreement**: the Comment's conclusion is not an independently adjudicated retraction.

- The Comment accepts that the proposed metric function solves a **reduced geometric equation** but argues the complete Einstein–Dirac–Born–Infeld field equations require a real positive square-root branch `Q=sqrt(1+2X/Lambda^4)>0`.
- For `rho(r)=sqrt(r^2+a^2)`, the Comment gives `Q=-G^t_t/G^r_r`, and at the proposed horizon `G^t_t(r_h)[1+Q(r_h)]=0`. For the alleged black-hole branch with `G^t_t(r_h)<0`, this forces `Q(r_h)=-1`, conflicting with `Q>0`. It argues the valid branch terminates **outside** the proposed horizon.
- Independent limited verification with the printed lapse `f(x)=1+3 x_g[x-(1+x^2)atan(1/x)]`, `x_g=4/(3pi)`: bisection yields `x_h=0.621184459919`, and the printed `a^2 G^t_t=(x_h f'(x_h)-1)/(1+x_h^2)=-0.299784630709<0`. This confirms the numerical sign relevant to the Comment, **not** the full scalar EOM or the complete no-go claim.
- **GTOR implication:** enforce domain-wide action variation, signed branch constraints, field-equation residuals and horizon regularity. Do not infer a valid black hole just because a chosen metric has finite scalar invariants. Compare with Audit 17's Weyl-cubed effective-action-derived throat, whose EFT validity near the throat remains limited: these are different gravitational actions and challenges.

### 3. A nonlinear causal semiclassical-gravity model gives testable alternatives to QM

**Samuel Fedida, Adrian Kent**, *A relativistic gravitisation of quantum theory?*. **Submitted 18 Nov 2026**.

Primary https://arxiv.org/abs/2611.24227 ; HTML https://arxiv.org/html/2611.24227v1 ; DOI https://doi.org/10.48550/arXiv.2611.24227 .

- Authors propose classical gravitational sourcing by a **locally updated reduced quantum state**, with measurement-collapse information admitted only within the future light cone. They derive a weak-field formulation and a nonrelativistic Schrödinger–Newton limit; these are **model proposals**, not tests demonstrating nature uses this theory.
- In the authors' analysis, the nonlinear source would allow nonstandard weakly disturbing state readout and approximate cloning while their specified update rule prevents superluminal signalling. They distinguish this from the usual quantum no-signalling principle. Its consistency and empirical viability require independent scrutiny.
- **GTOR consequence:** a useful countermodel for claims of physically nonlinear quantum dynamics. Calculate GTOR predictions for a given tabletop gravity/entangled-superposition experiment and compare both to ordinary linear quantum mechanics and this causally constrained nonlinear alternative. Demand stress-energy consistency, completely specified collapse/readout rule, and no operational superluminal signalling.

### 4. Clock-correlated open-subsystem dynamics is basis-dependent, even with globally unitary evolution

**A. Boette, F. Lomoc, N. Canosa, R. Rossignoli**, *Subsystem evolution in the history state formalism*. **Submitted 18 Nov 2026**.

Primary https://arxiv.org/abs/2611.23784 ; HTML https://arxiv.org/html/2611.23784v1 ; DOI https://doi.org/10.48550/arXiv.2611.23784 .

- For a history state and subsystem `A`, the post-clock-measurement mutual information is the Holevo-like quantity `I'(A,T)=S((1/N) sum_t rho_A(t))-(1/N) sum_t S(rho_A(t))` in the uniform standard clock basis. It is nonnegative and vanishes if all conditional reduced subsystem states coincide.
- For open subsystem `A`, the quantity generally **depends on clock basis**. The maximum across projective clock measurements is related to entanglement of formation: `max I'(A,T)=E(A,BT)-E(A,B)` within the paper's pure-history assumptions. This is not a measurement of universal absolute time, nor evidence that arbitrary state-vector phase is observable.
- **GTOR test:** specify whether "emergent time" is a clock-conditioned probability, relational observable or new dynamical field. Sweep clock bases/environment coupling. Require distinct invariant predictions *after* choosing actual clock measurement and measurement budget.

### 5. Curved spacetime produces a retarded gravitational-wave tail *inside*, not outside, the light cone

**Töre Boybeyi, Lorenzo Giombi, Niko Jokela, K. Kajantie**, *Filling the interior of the light cone*. **Submitted 17 Nov 2026**.

Primary https://arxiv.org/abs/2611.23407 ; HTML https://arxiv.org/html/2611.23407v1 ; DOI https://doi.org/10.48550/arXiv.2611.23407 .

- For spatially flat FLRW expansion, authors formulate the inside-light-cone tail of the gravitational-wave retarded Green function as a **Goursat characteristic-boundary problem**, calculate it for general expansion histories, and check exact Lambda-CDM solutions against a series and approximation.
- A signal's tail records scattering off curvature/trace of the cosmic matter content. The tail is supported *within* the future cone. Thus, light-cone interior support is consistent with microcausality and does not imply faster-than-light propagation.
- **GTOR test:** reconstruct a causal effective metric and compute its retarded propagator. Track (i) direct characteristic contribution, (ii) inside-cone tail and (iii) any support outside cone. Out-of-cone support would require an explicit departure from relativistic causal dynamics, not an appeal to ordinary curvature tails.

### 6. Matched thermodynamics can hide different intrinsic horizon geometry

**Hai Huang, Juhua Chen**, *Thermodynamic Degeneracy and Distinct Horizon Geometries of Rotating Kaluza-Klein Black Holes in Independent External Electric and Magnetic Fields*. **Submitted 17 Nov 2026**.

Primary https://arxiv.org/abs/2611.23425 ; HTML https://arxiv.org/html/2611.23425v1 ; DOI https://doi.org/10.48550/arXiv.2611.23425 .

- Under a special external-field Einstein–Maxwell–dilaton (Kaluza–Klein) solution family, a **common specified reference prescription** makes two regular branches share horizon area, quasi-local charges and temperature/potentials, despite distinct intrinsic horizon metrics.
- They prove a strict ordering of positive axisymmetric horizon scalar-Laplacian eigenvalues, `0<Lambda_(n,super)<Lambda_(n,sub)`, for the regular paired branches, n>=1. The branches may differ in boundary source data, so they are **not automatically coexisting thermal states in an identical ensemble**.
- **GTOR test:** thermodynamics, horizon area and shadow size are incomplete geometry fingerprints. Compare intrinsic curvature and the actual mode spectrum under controlled external fields; specify whether an exterior perturbation couples to those horizon harmonics.

### 7. Lorentz/Poincaré holonomy gives an operational acoustic analogue, not an absolute phase

**Claudio Furtado, A. M. de M. Carvalho, L. C. Garcia de Andrade, Fernando Moraes**, *Lorentz and Poincaré holonomies of an acoustic vortex: ergoregion transition, Sagnac time, and beyond-Aharonov–Bohm phonon scattering*. **Submitted 18 Nov 2026**.

Primary https://arxiv.org/abs/2611.24298 ; HTML https://arxiv.org/html/2611.24298v1 ; DOI https://doi.org/10.48550/arXiv.2611.24298 .

- For a hydrodynamic analogue metric, the Lorentz holonomy changes from rotation outside the ergoregion to null rotation at its boundary and boost inside. The authors derive a Poincaré translational holonomy interpreted as half the acoustic Sagnac delay: `tau(r)=(Gamma/c^2)/sqrt(1-v_s^2/c^2)`.
- They report a long-range **non-topological** correction to phonon scattering: `q sigma_perp=2pi alpha-(pi^3/3)alpha^3+O(alpha^5)`, rather than standard AB `sin(2pi alpha)`, under their vortex model assumptions.
- **GTOR test:** define its connection, non-contractible loops and detector response. A nonzero holonomy can be physical and gauge invariant, but a chosen coordinate winding, an exact flat connection on a contractible loop, and an observable loop phase are not interchangeable. Acoustic analogue metrics are not direct evidence of a fundamental spacetime helix.

## Two new foundational cross-checks (peer-reviewed historical literature)

1. **Robert W. Spekkens (2005)**, *Contextuality for preparations, transformations, and unsharp measurements*, Physical Review A 71, 052108. DOI https://doi.org/10.1103/PhysRevA.71.052108 ; public manuscript https://arxiv.org/abs/quant-ph/0406166 . Generalizes *operational equivalence* and contextuality beyond sharp measurements. Do not confuse this with the separate Bartlett–Rudolph–Spekkens 2007 review already catalogued.
2. **Alexander S. Holevo (1973)**, *Bounds for the Quantity of Information Transmitted by a Quantum Communication Channel*, Problemy Peredachi Informatsii 9(3), 3–11; English translation Problems of Information Transmission 9, 177–183. Archival bibliographic/abstract https://www.mathnet.ru/eng/ppi/v9/i3/p3 . The accessible original record gives **no DOI verified here**. Underpins the Holevo information constraint and contextualizes the clock paper's mutual-information expression. This is not an extra GTOR degree of freedom.

Page–Wootters 1983 and Bartlett–Rudolph–Spekkens 2007 are already in earlier audits and deliberately not recounted as additions.

## Reproducible limited numerical controls

Ran a **Python-standard-library** script offline on 20 Nov 2026, then added it to the existing research branch at `physics/gtor/scripts/gtor_audit18_operational_controls.py`. No full Einstein/DBI PDE, FLRW Goursat problem, nonlinear-collapse quantum theory or experiment is reproduced.

| Test | Result | Interpretation |
|---|---:|---|
| Pure-state Born transition probability | 0.126416219439475 | Ordinary projective baseline |
| Under independent nonzero complex state rescalings | residual 5.55e-17 | Standard C* invariance |
| Illustrative hidden-loop model: V=.93, Omega=pi/6, kappa=.02 | delta=0.010471975512 rad | **Hypothetical parameter**, not measured |
| Direct correlation difference vs closed-form Eq. (18) | Delta E=0.00973875922811; residual <3e-16 | Algebraic consistency, not evidence of memory |
| Dephased two-time qubit-history clock information: Z vs Fourier clock | ln(2)=0.693147180560 nats vs 0 | Basis-dependent conditional information |
| Twin KK branches at J=5,Q=P=2,H=4 | (p,C)=(4,.75) and (1,3); equal E=1.767766952966 | Toy use of paper's parametric identities only |
| DBI-comment lapse x_g=4/(3pi) | x_h=0.621184459919; a² G^t_t=-0.299784630709 | Horizon and sign check from printed equation |

This test suite explicitly *does not* validate GTOR, quantify experimental exclusion significance, or establish the whole DBI Comment. It provides transparent acceptance controls for a later GTOR-specific calculation.

## Contradictions and precise separations

- **"Gauge phase always matters" versus operational null:** geometric/relative Berry holonomy is standard physics; a *hidden path-memory correction* to identical POVMs is additional physics and was not seen at the sensitivity of the Shen et al. experiment. An absolute global state-vector phase is not thereby shown detectable.
- **"Regular metric" versus "actual solution":** the DBI Comment identifies a potential physical-branch failure at a purported horizon. The original journal-published black-hole claim and the unreviewed Comment must be kept distinct pending independent field-equation adjudication. This contrasts with, but does not directly refute, the Weyl-cubed EFT model in Audit 17.
- **"Nonlinear QM" versus "causal gravity":** Fedida–Kent's update rule is an explicit alternative to compare operational predictions with; their no-superluminal-signalling conclusion is conditional on that rule. Model-specific state-readout abilities create potential falsifiers, not a validated quantum-gravity result.
- **"Emergent time" versus "clock basis":** globally unitary history evolution can yield open-subsystem clock correlation with strong basis dependence. Any GTOR clock claim must specify which operational clock is measured.
- **"Light-cone interior" versus "acausal":** ordinary curvature scattering produces support inside the future cone without violating its boundary; test outside-cone tails separately.
- **"Same black-hole thermodynamics" versus "same metric":** distinct horizon geometries can have the same selected thermodynamic readouts under a chosen common normalization, especially if boundary sources differ. Isospectrality is not guaranteed.

## Prioritized falsification and computational work

1. **Quantum phase laboratory null**: define a fixed preparation, POVMs and direct/closed analyzer implementations as in 2611.23418, and explicit GTOR `Delta S(Omega,beta)`. Compare with calibrated standard QM and finite detection precision; calculate a preregistered exclusion/power curve. If GTOR reduces to a common unitary or rephasing, `Delta S=0` and there is no novel operational prediction.
2. **Full DBI/GTOR action compliance**: before asserting regularity, vary the *entire* action and solve all independent field equations and constraint equations on an open interval spanning the horizon. Check square-root/branch positivity, horizon determinant, normalized residuals, geodesic completeness and energy conditions. Use the DBI Comment's `Q>0` sign check as a regression test. A GTOR effective action needs explicit cutoff and truncation-error budget.
3. **Relational-time and nonlinear-QM discrimination**: construct the history state, environment and readout POVM; calculate `I'(A,T)` for multiple clock bases, and any alleged GTOR quantum-state readout/causal-gravity correction. Verify causality and calibration under state rephasing and arbitrary reference changes.
4. **Gravity multimodal test**: on an action-derived Lorentzian background compute retarded GW direct/tail Green functions, inner horizon/branch constraints, curvature/tidal response, and coupled perturbation spectra. Control source preparation, boundary fields and gauge. Test whether distinct interiors predict distinct *observable* transfer functions rather than only a different coordinate metric.

**Execution boundary:** source-first, selective, with no paywall/full-text claims beyond publicly retrieved arXiv HTML, no exhaustive coverage, no empirical GTOR confirmation, no cloud changes/spending, no merge or reviewer override. Continue on draft PR #2 only. No owner sign-in or fresh authorization was required for this scoped research and documentation.
