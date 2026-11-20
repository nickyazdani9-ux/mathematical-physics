# GTOR 2.0 — Research Audit 06 (8 November 2026)

**Scope/status.** Selective review of public primary scholarly sources and a reproducible toy numerical check. **New to this GTOR ledger**, not necessarily first published today. Deduplicated against the four preceding files (4–7 November, Audits 02–05) and earlier topics: Hayward, Kerr/Ovalle, Lan et al., Peng et al., Di Filippo QTG, Corichi, Page–Wootters, Ashtekar–Schilling, Aharonov–Anandan, Berry, Fubini–Study, and the independently found flat helical-surface gauge issue. No experimental support for GTOR, merge, or cloud/spending changes.

## Delta 1 — Regular core can be optically invisible, with a local NEC price (newly revised 2 November)

**F. Tello-Ortiz, Y. Gómez-Leyton, A. Rueda, K. Bamba, M. González-Espinoza**, *Regular black holes with Minkowskian cores: energy-condition obstruction and optical screening*, arXiv:2607.10713v3, **first submitted 12 July 2026, revised 2 November 2026**, open primary [arXiv HTML v3](https://arxiv.org/html/2607.10713v3), [DOI 10.48550/arXiv.2607.10713](https://doi.org/10.48550/arXiv.2607.10713). **Preprint; full open HTML examined.** Do not present revision date as original submission or assume journal peer review.

For the authors' static asymptotically flat solution in units G=c=1,
\[
ds^2=-F(r)dt^2+F(r)^{-1}dr^2+r^2d\Omega^2,\quad
F(r)=1-\frac{2m(r)}r,\quad
m(r)=M P(5,x),\quad x=r/\ell=12\alpha r/M,
\]
where
\[
P(5,x)=1-Q(5,x),\quad
Q(5,x)=e^{-x}\sum_{j=0}^{4}\frac{x^j}{j!}.
\]
Near r=0, m(r)=O(r^5); central density/curvature vanish, unlike a de Sitter center. The two-horizon branch still has an inner horizon; a flat core does not establish complete extensions or inner-horizon stability. The rotating Gürses–Gürsey ansatz has curvature invariants bounded for the **physical domain r≥0** but the authors explicitly do **not** establish a regular continuation across the Kerr disk for r<0.

**Exact conditional energy obstruction:** in this static closure class \(p_r=-\rho\) and \(\rho+p_t=-r\rho'/2\). Since \(\rho\propto r^2e^{-r/\ell}\), \(\rho'>0\) for \(0<r<2\ell\), and tangential NEC is violated **there**, not necessarily globally. This is not a no-go result for every regular-core geometry, matter action or extension.

**Optical screening:** for the outer photon sphere near \(r=3M\), the first-order impact-parameter shift is
\[
\frac{b_c-b_{\mathrm{Schw}}}{b_{\mathrm{Schw}}}\simeq-Q(5,36\alpha),\quad b_{\mathrm{Schw}}=3\sqrt3M.
\]
At M=1, α=0.5, this predicts -8.4176098049e-5; an independent offline solve of \(2F-rF'=0\) gives \(r_{\rm ph}=2.998539012385\), exact relative shift -8.4519074169e-5. This reproduces the paper's order of agreement, **not** a measurement or GTOR prediction. **Implication:** unchanged EHT-like shadow does not distinguish sharply different interiors. [Offline code](scripts/gtor_core_screening_test.py).

## Delta 2 — Evaporation does not automatically evade quantum singularity theorems

**N. Engelhardt and I. Nagar**, *A Quantum Singularity Theorem for the Evaporating Black Hole*, **submitted 6 May 2026**, [arXiv:2605.05326](https://arxiv.org/abs/2605.05326), [DOI 10.48550/arXiv.2605.05326](https://doi.org/10.48550/arXiv.2605.05326). **Open primary arXiv-rendered text reviewed; preprint status, no independent proof verification.**

They prove future **null geodesic incompleteness** given a stably causal, past-reflecting, spatially open spacetime, an appropriately compact **robustly quantum trapped** region, plus the **Generalized Second Law and generalized-entropy strong subadditivity**. Key functional:
\[
S_{\mathrm{gen}}(a)=\frac{\mathrm{Area}(\partial a)}{4G}+S_{\rm vN}(\rho_a).
\]
Their paper explicitly labels the GSL and strong-subadditivity properties it needs as conjectures assumed in the proof. Unlike the classical Penrose theorem, no global hyperbolicity or classical NEC is needed. **Conditional result in a semiclassical domain**, neither proof of curvature divergence nor universal prohibition on new fundamental quantum gravity.

**Comparison:** Di Filippo's Audit 05 theorem assumes certain QTG equations, NEC and H' positivity, while Engelhardt–Nagar require semiclassical entropy/causality structure. Different mechanisms; GTOR cannot claim completeness solely by relaxing the **classical** NEC.

## Delta 3 — A genuinely different geometric quantum model must deform dynamics, not rename phase

**H. Heydari**, *Geometric Quantum Mechanics in a Symplectic Framework: Metric-Affine Extensions and Deformed Quantum Dynamics*, **submitted 22 March 2026**, [arXiv:2603.22354](https://arxiv.org/abs/2603.22354), [DOI 10.48550/arXiv.2603.22354](https://doi.org/10.48550/arXiv.2603.22354). **Open primary arXiv-rendered text examined, preprint not verified peer reviewed.**

Standard rays still obey \([\psi]=[\lambda\psi]\) for any complex \(\lambda\ne0\), retaining the **established** Fubini–Study/Kähler quantum kinematics. Proposed deviation is \(\omega_{\cal G}=\omega+\delta\omega[g_{\mu\nu},\Gamma^\lambda_{\mu\nu}]\), with a closed, nondegenerate symplectic two-form. For fixed Hamiltonian H, the first-order correction obeys
\[
\iota_{\delta X_H}\omega=-\iota_{X_H}\delta\omega,\qquad
\iota_{X_H}\omega=dH.
\]
This is a **model proposal**, not evidence that global phase/norm become observable. Need specified spacetime-to-ray-space pullback, coupling constants, and consistent quantum measurements. Closure \(d\delta\omega=0\) is an extra condition, not automatic from choosing a background curvature/torsion. In particular, the paper's section 6.3 step from covariant Bianchi \(D{\cal R}=0\) to ordinary exterior closure \(d{\cal R}=0\) requires scrutiny for a matrix-valued curvature form; an explicitly scalar, gauge-invariant construction is needed. Do not treat general metric-affine proposals as already validated deformations of quantum theory.

**Potential falsifier:** on \(\mathbb CP^1\), specify \(\delta\omega=\epsilon\omega\) with spatially constant \(\epsilon\). Then \(X'_H=X_H/(1+\epsilon)\) for the same H, shifting a calibrated qubit precession rate by \(1/(1+\epsilon)\). But a clock/Hamiltonian rescaling degeneracy exists unless energy and external clock are independently fixed. Require same-ray invariance under \(\psi\mapsto c\psi\); reject any GTOR shift changing under an arbitrary common phase or norm.

**Related phase comparator (abstract only):** **K. Kumar and S. Majid**, *Geometric Aharonov-Bohm phase effect around a black hole*, **submitted 27 May 2026**, [arXiv:2605.28582](https://arxiv.org/abs/2605.28582), [DOI 10.48550/arXiv.2605.28582](https://doi.org/10.48550/arXiv.2605.28582). They propose semiclassical Klein–Gordon-linked amplitude transport and a background-dependent phase around a black hole. **Only primary arXiv abstract verified**; do not attribute a measurable absolute-phase effect or validated gravity model to this source. A relative-path interferometer or closed holonomy would have to be specified.

## Delta 4 — Kasner asymptotics yield an interior variational benchmark

**Z.-X. Xiong and H. Lü**, *First Law of Black Hole Interior Dynamics*, **submitted 19 August 2026**, [arXiv:2608.18583](https://arxiv.org/abs/2608.18583), [DOI 10.48550/arXiv.2608.18583](https://doi.org/10.48550/arXiv.2608.18583). **Open primary arXiv-rendered text examined; publication/peer review not verified.**

For specified electrically charged Einstein–scalar black holes with a Kasner-like singular regime, the Iyer–Wald covariant phase-space formalism yields an interior relation
\[
\delta M_K=\mathcal T\delta S+\Phi_e\delta Q_e+\phi_K\delta\Sigma_K.
\]
The relevant anisotropic power-law interior has Kasner constraints \(P_t+2P_T=1\), \(P_\phi^2+3P_T^2-2P_T=0\), under scalar kinetic dominance. This is **not** an asymptotically regular black-hole model and does not prove any such first law for GTOR; it is a comparator for conserved charges, variational integrability, near-endpoint scaling and action derivation. Foundational method: **Iyer and Wald (1994)**, *Some properties of Noether charge and a proposal for dynamical black hole entropy*, [DOI 10.1103/PhysRevD.50.846](https://doi.org/10.1103/PhysRevD.50.846).

## Assumption conflicts and falsifiable queue

| Hypothesis/claim | Established comparator and limitation | GTOR rejection test |
|---|---|---|
| A global phase/norm or original helix winding k is physically observable | Rays quotient nonzero complex scaling; previous Audit 04 eliminated k from the induced flat metric under \(\phi=v+2\pi ku\) | Name an operational gauge-invariant observable. Numerically compare states/metrics after complex rescaling, rephasing and helical re-coordination. A nonzero difference without an extra physical structure is a gauge artifact. |
| Finite curvature at the center = nonsingular complete black hole | Tello-Ortiz v3: local static regularity, yet inner horizon and possible obstructed rotating extension; Engelhardt–Nagar: semiclassical entropy hypotheses force null incompleteness | Integrate affine null geodesics to boundary, compute parallel-propagated curvature and maximal Lorentzian extension; for evolving semiclassical GTOR, calculate \(S_{\rm gen}\) and check theorem hypotheses. |
| Small change in shadow = small change in interior geometry | Exponential screening \(Q(5,36\alpha)\) contradicts this inference | Reproduce benchmark \(r_{\rm ph},b_c\), then compare GW QNMs, echo structure, tidal response and matter-dependent observables with a declared likelihood model. |
| A new geometric phase model is physical | Heydari proposes changed \(\omega\); Aharonov–Anandan/Berry measure relative/cyclic holonomy, not arbitrary common phase | Compare independently clock-calibrated qubit dynamics for a **specified** GTOR \(\delta\omega\); require \(d\omega'=0\), nondegeneracy, gauge invariance and observable separation beyond standard QM. |
| An asymptotic Kasner endpoint is a regular throat | Xiong–Lü assume a singular, scalar-dominated power law; inner first law requires an action and integrable charges | Fit exponents, curvature and variational charges; fail any GTOR regularity/completeness claim if the endpoint is reached in finite affine length or its curvature grows without bound. |

**Top next steps:** (1) write the GTOR Lorentzian action plus metric and matter/quantum state coupling, without conflating intrinsic geometry with embedding coordinates; (2) choose and preregister a single gauge-invariant observable; (3) run the offline optical benchmark script and extend to geodesic/NEC/inner-horizon tests; (4) test entropy and generalized second law hypotheses in any evaporating candidate; (5) run an explicitly clock-calibrated \(\mathbb CP^1\) phase/dynamics comparison.

**Limits:** Sources here were checked as open primary HTML/abstracts; there is no claim of reading paywalled work or exhaustive 6–8 Nov postings. Newness refers to **newly catalogued** papers, including a 2 Nov revision; numerical checks only reproduce a published model and are not empirical GTOR validation. No merges, deployment, cloud provisioning, or financial changes.
