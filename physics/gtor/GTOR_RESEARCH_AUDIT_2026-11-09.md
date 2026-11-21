# GTOR 2.0 — Research Audit 07 (9 November 2026)

**Status.** Selective, public-primary literature review; new to the GTOR ledger, NOT necessarily published today. Compared with Audits 02–06 (4–8 Nov) and their source ledger. No experimental confirmation of GTOR, no new GTOR field equations, no merges, no cloud provisioning or spending.

## Delta 1 — 3 Nov quasi-topological core universality: a conditional *de Sitter* requirement

**P. Bueno, R. A. Hennigar, M. G. Marrone, Á. J. Murcia**, *The quasi-topological space of quasi-topological gravities*, **submitted 3 Nov 2026**, **unreviewed preprint**. Primary [arXiv:2611.05485v1](https://arxiv.org/html/2611.05485v1); arXiv DOI [10.48550/arXiv.2611.05485](https://doi.org/10.48550/arXiv.2611.05485) (registration marked pending). **Full public HTML read**, in particular equations 3–19 and core/horizon sections.

In the authors' positive-mass, static spherical **vacuum** branch of quasi-topological gravity (typically polynomial infinite towers in D>=5), with nondegenerate monotone characteristic function `h'(psi)>0`, define

```
S=(16*pi*G)^(-1)*int sqrt(-g) [R - 2*Lambda + sum a_n Z_n] d^D x
psi(r)=(1-f(r))/r^2
h(psi)=2*Mbar/r^(D-1),  f(r)=1-r^2*psi(r).
```

Their conditional theorem: any nontrivial regular positive-mass vacuum solution in the specified algebraic class must have `psi(0)>0` (de Sitter core); a Minkowski or AdS core is excluded. The proof relies on the **specific** characteristic equation, positive mass, monotonicity and vacuum branch. They show the number of Killing horizons is unbounded across appropriate choices of couplings (including arbitrarily degenerate cases), yet simple horizon **sign-changing parity** remains consistent with the Lan et al. 3 Nov finite-area-core analysis (already Audit 03). Neither establishes nonlinear collapse stability or completeness. For `h_H(psi)=psi/(1-alpha*psi)`, `f=1-2*Mbar*r^2/(r^(D-1)+2*Mbar*alpha)`, with core `f=1-r^2/alpha+...`. This is an explicit action-class example, not a generic GTOR model.

**Apparent clash resolved.** Audit 06 catalogued Tello-Ortiz et al.'s **Minkowskian core** supported by a nontrivial effective matter tensor. The new theorem assumes **vacuum** quasi-topological equations; the claims are not contradictory. Audit 05 Di Filippo's incompleteness theorem has further global/energy/trapped-surface assumptions, so regular curvature here also does not contradict it or prove completeness.

## Delta 2 — Hayward collapse can regenerate a singularity despite a static regular core

**C. Shao, J.-Q. Guo, Y. Tian, H. Zhang**, *Internal structure of Hayward black holes*, submitted **28 Nov 2025**, revised **16 June 2026**; **Phys. Rev. D 113, 124040 (2026)**. Open primary [arXiv:2511.23165v2](https://arxiv.org/html/2511.23165v2), journal DOI [10.1103/qyyr-df8v](https://doi.org/10.1103/qyyr-df8v). **Full open HTML read** (scalar-collapse action, evolution, near-critical results).

Their *Einstein gravity + nonlinear electrodynamics + massless scalar* dynamical model keeps a finite-radius inner horizon under weak scalar-field perturbations, but sufficiently strong perturbations shrink it to zero and create a spacelike singularity. Near a model-dependent threshold `p_*`, reported numerical fit:

```
r_minus ~ |p-p_*|^gamma;   gamma ≈ 0.5.
```

This is a **published numerical result for a specified model**, not a universal exponent for GTOR. It sharpens earlier Audits 02–06: scalar **linear mode stability**, tamed late-time mass inflation, and full **nonlinear** stability are different criteria. It is *not* a direct refutation of the distinct single-horizon finite-area-core construction.

**Proposed quantitative benchmark:** solve the published matter+metric PDEs for a family of Gaussian amplitudes about the cited `C_*≈0.048825647950` at their stated fixed width/position; fit log(r_minus) versus log|C-C_*| with grid refinement. Fail the claimed scaling if it is not convergent, or if the metric/action mismatch prevents comparison. GTOR itself has no verified PDEs for this test yet.

## Delta 3 — charged regularity is stricter than neutral regularity

**R. Carballo-Rubio, C. Coviello, V. Vellucci**, *Charging up regular black holes*, submitted **8 July 2026**, revised **27 Oct 2026 (v3)**, open primary [arXiv:2607.07831](https://arxiv.org/abs/2607.07831), arXiv DOI [10.48550/arXiv.2607.07831](https://doi.org/10.48550/arXiv.2607.07831). **Primary arXiv metadata and abstract verified only**; full proof not read. Unreviewed publication status.

Within their particular identically conserved deformed-Einstein-tensor construction coupled to electrodynamics, standard charged Bardeen/Hayward continuations develop singularities; the authors give modified regular options. This is not a universal statement that *all* charged Hayward-inspired theories are singular. **GTOR test:** from one explicit action, vary charge Q continuously away from zero, recompute curvature invariants and the domain of regular extension; do **not** infer charged/rotating stability from Q=0 or a stationary line element.

## Delta 4 — established quantum geometry has an operational *reference-frame* theory

Not novel papers, newly catalogued foundational control group:

- **T. W. B. Kibble (1979)**, *Geometrization of quantum mechanics*, Commun. Math. Phys. 65, 189–201. DOI [10.1007/BF01225149](https://doi.org/10.1007/BF01225149). **Publisher/metadata/abstract verified; not full publisher paper.** Symplectic Hamiltonian geometry of the manifold of pure quantum states; no implied observable arbitrary global phase.
- **W. K. Wootters (15 Jan 1981)**, *Statistical distance and Hilbert space*, Phys. Rev. D 23, 357. DOI [10.1103/PhysRevD.23.357](https://doi.org/10.1103/PhysRevD.23.357). **APS primary abstract verified.** Operational statistical distinguishability yields the ray-angle distance under the paper's measurement assumptions.
- **S. D. Bartlett, T. Rudolph, R. W. Spekkens (5 Apr 2007)**, *Reference frames, superselection rules, and quantum information*, Rev. Mod. Phys. 79, 555–609. DOI [10.1103/RevModPhys.79.555](https://doi.org/10.1103/RevModPhys.79.555); open [arXiv:quant-ph/0610030](https://arxiv.org/abs/quant-ph/0610030). **Primary publisher abstract reviewed, public text excerpt accessible.** Measurable phase relations require a physical reference and/or relational encoding, not an absolute ray phase.

Standard pure states are **rays** `[psi]=[c*psi]`, for **all** nonzero complex `c`, with projective angle `d_FS=acos(|<psi|phi>|/(||psi|| ||phi||))` in the convention used by Audits 04–06. Observable relative phase between components is not observable *global* phase. Constraint-group averaging (Audit 05, Corichi) introduces **additional** equivalences: do not mix with normalization or standard U(1) quotient.

**Operational null test:** prepare `|psi(phi)>=(|0>+e^(i phi)|1>)/sqrt(2)` and measure `|+>`. Then `P(+|phi)=(1+cos phi)/2`, with `P(+|pi/3)=.75`, `P(+|2pi/3)=.25`; multiplying the entire ket by any `c≠0` changes neither probability. This confirms ordinary relative-phase sensitivity, not GTOR. For any proposed new GTOR phase effect, state reference/measurement, Hamiltonian, gauge action and at least one predicted `Delta P ≠ 0` invariant under `psi→c psi`; absent that, it is a gauge choice.

## Asymptotic-model cross-check (not counted as a new thesis)

**Y.-B. Liang, H.-R. Li**, *Asymptotic Throat: The Geometric Inevitability of Regular Black Holes* (**13 Apr 2026**), [arXiv:2604.11396](https://arxiv.org/html/2604.11396v1), [DOI 10.48550/arXiv.2604.11396](https://doi.org/10.48550/arXiv.2604.11396). Open full HTML read, especially their demonstration that if `g_tt*g_rr=-1` and areal radius reaches finite `L`, the radial null affine length `int sqrt(-g_tt*g_rr) dr` is finite; calling `r=L` infinity without an independently diverging affine integral fails. Their **30 Sept follow-up [arXiv:2609.39186](https://arxiv.org/abs/2609.39186)** was already identified in Audit 02. This April source is context, not a new 9 Nov result. Note their separate-source stress/energy conditions and that some complete asymptotic ends can require NEC violations close to the endpoint. Check exact metric functions rather than relying on coordinate divergence.

## Independent offline check and next decisions

[Executable stdlib-only test](scripts/gtor_qtg_and_ray_check.py): for the new paper's `D=5, alpha=0.1, Mbar=1` Hayward-form vacuum branch: `r_- = 0.324919696233`, `r_+ = 1.376381920471`, `psi(0)=1/alpha=10`, and constant-core-curvature `K(0)=2D(D-1)/alpha²=4000` (chosen length units). Verify `h((1-f)/r²)=2Mbar/r⁴`, `h'>0`; qubit rephasing/norm invariance to 1e-12. These are **model mathematics**, not astrophysical data or GTOR confirmation.

**Highest-value GTOR decisions/tests:**
1. Derive a GTOR action, a signature-(−+++) metric, a matter/charge sector and admissible boundary/clock conditions; otherwise black-hole and quantum claims remain two uncoupled programmes.
2. Make the neutral/charged distinction explicit. Scan `Q=0→Q≠0` and finite-energy scalar pulses; monitor Kretschmann, parallel-propagated components, horizons, and null affine length.
3. Declare which no-go theorem hypotheses GTOR satisfies (**Bueno h'>0 vacuum class; Di Filippo focusing; Peng scalar-tensor turning point; Engelhardt–Nagar generalized entropy**) before claiming evasion. Violating an assumption is not itself a consistent replacement theory.
4. On quantum states, specify reference-assisted gauge-invariant predictions beyond Kibble/Wootters/Bartlett and the earlier Bargmann-invariant benchmark, with preregistered tolerances. Drop unsupported absolute-phase/norm observability claims if all results match ordinary rays.
5. Numerically validate the D=5 algebra baseline first; only then attempt expensive nonlinear collapse PDEs **offline**. No GCP/GKE provisioning or spend.

**Evidence accounting.** Full public HTML: Bueno et al., Shao et al., Liang–Li April. Primary abstracts only: charging paper, Kibble, Wootters, Bartlett (with limited open excerpt). Latest arXiv preprint peer review unverified. Selective search, not exhaustive. Mathematically distinct papers from Audits 02–06 are not re-labelled as new. No merge, compute resource provisioning or spending.