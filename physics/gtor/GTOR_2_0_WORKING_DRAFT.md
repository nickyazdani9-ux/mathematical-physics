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
