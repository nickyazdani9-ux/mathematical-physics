# GTOR foundational comparison — time, units, and helical axioms
**Date:** 2026-11-14 · **Status:** preliminary source-grounded review; not a complete formal consistency proof.
**Sources:** `physics/wips/docs/gtor_revised_cors1-20.tex` (March 2026); `physics/gtor/GTOR_2_0_WORKING_DRAFT.md` (3 November 2026). Original versions remain unchanged.

## Revised conceptual intent (owner discussion, 14 Nov)
Start with explicitly chosen units of time and length. Define one spatial unit as the distance locally propagating light traverses in one temporal unit. Thus the *numerical value* of c is one in this convention; dimensional speed remains length/time until units are identified. Time readings are clock- and observer-dependent; do not infer a physical rate of time from the identity dt/dt = 1. Treat angular phase, axial displacement, radius and pitch as independent modelling quantities until a coupling is defined or derived. The previous rule of one axial unit per turn is an arbitrary model choice, not an axiom justified by the physics. Candidate variable pitch/radius or their growth-rate ratios are separate hypotheses; alpha (fine-structure constant) is a dimensionless candidate to test, not an established consequence.

## Comparison with original six-axiom helical paper
| Topic | March 2026 paper | Revised interpretation / finding |
|---|---|---|
| Time | Axiom 1: dt/dt=1; asserts this fixes c=1 | dt/dt=1 is an identity where t is a valid differentiable coordinate. It does not constrain c or establish a physical clock model. **Unjustified inference.** |
| Axial coordinate | Axiom 1: z=1+u; Axiom 5 embedding uses z=u | Different origins can be equivalent under translation, but the convention and cell gluing must be stated. |
| Radius | Axiom 2: r=theta/pi and r(u)=1+u | Axiom 4: theta=v+2*pi*k*u, so r=theta/pi would depend on v while r(u) does not. **Literal simultaneous definitions conflict for variable v**, unless the first equality is reinterpreted/restricted. |
| Expansion | Axioms 2 and 3 both require dr/dz=1 | Redundant assumption; not derived from a light-cone physical model. |
| Winding | Axiom 4: theta=v+2*pi*k*u, integer k | Coupling and integer restriction imposed. Reopen pitch and winding parameters as hypotheses. |
| Closure | Axiom 6: psi(u,v+2*pi)=psi(u,v) | Fourier integer mode labels follow for ordinary single-valued periodic functions; physical angular momentum interpretation requires more structure. |
| Geometry | Embedding produces a Euclidean induced metric, a cone after coordinate transformation | Mathematical metric computations can be valid conditionally; a Euclidean 2-manifold is not thereby a physical 3+1 Lorentzian spacetime. |
| Holonomy | A_u=2*pi*k; identifies k as topological charge | Under v'=v+2*pi*k*u, the off-diagonal term disappears. Nontrivial gauge-invariant holonomy/topological charge is **not established** by this coordinate expression alone. |
| Dynamics | Later geodesic potential substitutes L=k | Identification of conserved geodesic angular momentum L with winding k requires independent justification. |
| Quantum analogies | Wave/Dirac/Schrodinger analogies | Some are explicitly labelled analogies; do not promote them to physical derivations. |

## Comparison with GTOR 2.0 working draft
The November draft no longer lists dt/dt=1 or c=1 as foundational axioms. It instead proposes relational configurations X and deterministic update F:X→X, explicitly separating determinism from reversibility and leaving physical time and geometry as emergence hypotheses. This is a **substantive change of foundational model**, not merely different notation. The new draft also expressly declines to claim a verified physical unification. It still needs formal checks for definitions, model existence, independence, and any claimed emergent geometry/time.

## Time and units: precise statement
Choose a temporal unit T0 and length unit L0 with L0 = c*T0 for local vacuum light propagation. The speed has numerical representation c=1 L0/T0. Setting c=1 as a convention identifies the corresponding time and length scales; it does not imply that time itself changes at an absolute rate. In relativity, proper time along a timelike worldline and coordinate time need not coincide. A photon's local speed is c even in curved spacetime; coordinate speeds can vary.

## Immediate audit priorities
1. Extract the exact March axiom set and decide whether r=theta/pi is a universal equality, a restricted cross-section, or a heuristic. The literal reading conflicts with theta(v,u).
2. Remove the asserted derivation dt/dt=1 ⇒ c=1. Replace with an explicit unit convention, if intended.
3. Classify all corollaries as algebraic consequences, imported mathematics, additional assumptions, or physical interpretations; inspect first unsupported step.
4. Check coordinate/gauge invariance of claimed winding charge and distinguish integer parameter choices from topological invariants.
5. Run symbolic consistency checks and construct finite countermodels where suitable; report what is formally proved and what remains underdetermined.
6. Preserve source version hashes and do not extend dependent GTOR results as established until validation.

**Initial conclusion:** The March text contains at least one apparent literal incompatibility in its radius/angle definitions and several unsupported physical inferences. This is **not yet a full proof of inconsistency** of every possible formal interpretation. The November GTOR 2.0 draft explicitly revises the foundations rather than retaining the old time axiom.
