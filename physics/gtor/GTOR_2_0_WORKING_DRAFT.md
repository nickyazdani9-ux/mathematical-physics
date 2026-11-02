# GTOR 2.0 — Recursive Relational Flow

**Author:** Nick Navid Yazdani  
**Status:** Working research draft 0.1, 3 November 2026  
**Linear:** https://linear.app/atlas-inference/project/gtor-20-mathematical-physics-9f56de56ae30

## Abstract

This paper develops a candidate mathematical framework in which distinguishable information, relational structure, and state evolution are treated as foundational. Geometry, physical time, and observable fields are proposed as possible emergent descriptions, not as established derived results. A program of precise definitions, symbolic verification, counterexample construction, and empirical benchmarking is outlined. No claim of verified unification is made.

## Plain-language working table

| # | Concept | Shorthand | Working meaning | Evidence status |
|---|---|---|---|---|
| 1 | Information | I | Distinguishable states and configurations | Proposed primitive |
| 2 | Connections | C | Relationships among distinguishable states | Proposed primitive |
| 3 | Flow | F | Rules producing changes in the relational state | Proposed primitive |
| 4 | State | X | Complete configuration at one update | Defined notation |
| 5 | Determinism | D | Same full state, same next state | Candidate assumption |
| 6 | Reversibility | R | Complete past state is recoverable | Open physical hypothesis |
| 7 | Space | S | Emergent geometry of relations | To derive |
| 8 | Time | T | Emergent physical duration from ordered change | To derive |
| 9 | Scale | L | Resolution and refinement of structure | To investigate |
| 10 | Projection | P | Observable lower-information representation | To define |
| 11 | Dissipation | H | Subsystem energy transfer to other degrees of freedom | Established for specific coupled models, not universal |

## Minimal formal model (candidate)

Let X be a set of full relational configurations, F:X→X a single-valued update rule, and X(n+1)=F(X(n)). This is deterministic by construction. F is **not** necessarily invertible. Reversibility requires F to be one-to-one on the accessible states (or a richer dynamical structure). An initial state X(0) and a past-eternal chain are alternative assumptions and must not be conflated.

An underlying graph may encode relations without presupposing spatial distance. Graph shortest-path distance supplies a metric when the graph is connected; proving a continuous 3+1-dimensional Lorentzian limit is a separate open problem.

## Research hypotheses, not established results

- Macroscopic geometry may emerge from relational network structure.
- Physical time may emerge from ordering and clocks built from subsystems.
- Multiscale refinement may yield fractal-like structure; infinite physical refinement is not established.
- Localised excitations of a coupled structure may reproduce field or particle observations.
- Quaternions may be an effective algebra for particular rotational and coupled modes, without being assumed fundamental.
- Dissipation in a subsystem may emerge from reversible/conservative environment coupling.

## Method and verification standards

1. Discuss in plain English and assign intuitive shorthand.
2. Define mathematical objects without smuggling desired conclusions into assumptions.
3. Derive an explicit proposition and check with SymPy / the better_calculator CAS kernel.
4. Test limiting cases, units, counterexamples, and independent numerical implementations.
5. Label each result: proposed, mathematically derived (conditional on assumptions), independently computationally checked, empirically supported, contradicted, or open.
6. Compare known theories with reproducible benchmarks; claims about reality require experiment.

## First research tasks

1. Define relational state and evolution with a minimal worked example.
2. Distinguish deterministic, reversible and information-loss dynamics via explicit counterexamples.
3. Establish conditions for emergent metric and investigate manifold limits.
4. Derive a coupled system-and-bath model yielding a memory kernel; show Markovian approximation assumptions and energy accounting.
5. Investigate quaternionic and self-similar structure only after foundational definitions.
6. Identify a falsifiable prediction or quantitative computational advantage.

## Existing source material

- [GToR revised, six axioms and twenty corollaries](https://github.com/nickyazdani9-ux/mathematical-physics/blob/main/physics/wips/docs/gtor_revised_cors1-20.tex)
- [Maxwell helix symbolic experiment](https://github.com/nickyazdani9-ux/mathematical-physics/blob/main/physics/gtor/scripts/maxwell_helix_complete.py)
- [SymPy-based calculator kernel](https://github.com/nickyazdani9-ux/better_calculator/blob/main/python/engine.py)

## Limitations

A rigorous internal mathematical derivation does not independently validate the physical assumptions used. The proposed framework has not been shown to reproduce general relativity, the Standard Model, quantum measurement statistics, or the observed cosmological constant. Such results remain tasks for future research.


## Research Audit 01 — 3 November 2026

### Aim

Compare GTOR's candidate assumptions to established mathematical/physical constraints. A mismatch with an interpretation is not necessarily a flaw in the predictive formalism. The burden is to produce an explicit, quantitative alternative.

| Topic | What conventional formalism actually establishes | Open GTOR test |
|---|---|---|
| Global phase | Replacing an entire isolated quantum state by the same state times one unit-modulus complex number changes no Born probabilities; relative phases remain measurable. | Define an external reference or added interaction that gives a common phase an observable distinction. |
| Normalisation | Rescaling a nonzero state vector to unit norm picks a representative of a physical ray; it is not generally a projection that discards geometry. | State exactly what extra amplitude/radial degree of freedom is physically measured and how. |
| Black holes | Classical singularity theorems establish geodesic incompleteness under specified hypotheses, not independently an actual point of infinite matter density. | Build a complete nonsingular interior satisfying clear dynamics and boundary conditions, and recover exterior observables. |
| Information preservation | Deterministic evolution alone does not imply a reversible inverse. | Prove injectivity/bijectivity or explicitly expand state space to retain a bath/history. |
| Emergent geometry | A graph metric is definable, but Lorentzian 3+1 geometry does not follow without additional assumptions and a scaling limit. | Build a finite model, then demonstrate a continuum limit and physical clock/ruler behaviour. |
| Dissipation | Effective damping can arise after eliminating environmental degrees of freedom; static friction needs contact mechanics. | Benchmark against solvable system-plus-bath models and verify energy accounting. |

### First exact check — global phase

For any complex state vector psi, scalar a with |a|=1, and Hermitian observable O, (a psi)† O (a psi) = psi† O psi. If psi is nonzero, dividing by psi† psi makes this identity independent of the vector's overall norm. This is elementary algebra. It establishes observational equivalence for the standard observable algebra, not the metaphysical nonexistence of an absolute phase. To claim a measurable global phase, GTOR must specify an additional comparison system or operational observable that changes between these states.

### First exact check — normalisation

For a nonzero complex vector v, v / sqrt(v†v) has unit norm. The map discards the positive overall magnitude as a *coordinate convention for rays*, but a physical theory may assign meaning to magnitude via total excitation number or other degrees of freedom. These are distinct constructions: no theorem states all physical information must be lost.

### Black-hole asymptote proposal

A regularising map r = L * u/(1+u) (u>0) produces a bounded r but does not itself solve Einstein's equations, ensure curvature invariants remain finite, or establish completeness of all causal geodesics. The proposed solution must provide a metric, field equations, stress tensor or effective action, boundary/initial conditions, curvature checks, global extension and measurable predictions. Begin with a spherically symmetric toy metric, compare against Schwarzschild and published regular-black-hole models, and derive its ringdown/shadow consequences.

### Literature benchmarks

- Luminet (2025), A Primer on Spacetime Singularities I: Mathematical Framework: https://doi.org/10.3390/universe11080272
- Gregory and Nissanke (2025), Black Holes as Laboratories: Tests of General Relativity: https://doi.org/10.1007/s10714-025-03437-7
- Black Hole Spectroscopy and Tests of General Relativity with GW250114 (2026): https://doi.org/10.1103/6c61-fm1n
- Nonperturbative quantum gravity denounces singular black holes (2025): https://doi.org/10.1103/PhysRevD.111.086026
- Nemoto and Braunstein (2003), Quantum coherence in the presence of unobservable quantities: https://doi.org/10.1103/PhysRevA.68.042326

### Decision gate

Before adding a new axiom: specify one physically distinguishable observable that GTOR predicts differently from an established baseline, or one rigorously demonstrated simplification with no loss of empirically established results. Prefer the phase/normalisation test as a small first benchmark, before the full black-hole interior problem.
