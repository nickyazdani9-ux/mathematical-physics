# A winding and projective selective state model

Status: proposed architecture and independently derived primitives. Novelty and financial usefulness are unestablished. All time derivatives use Leibniz notation.

## What carries forward from the repository

The Riccati manuscript defines the logarithmic derivative z=y'/y. For y''+P(t)y'+Q(t)y=0, direct differentiation gives

$$\frac{dz}{dt}=-z^2-P(t)z-Q(t).$$

This identity is exact where y is nonzero. A zero of y creates a pole of z, even when y itself is regular. A constant root of z²+Pz+Q=0 can generate an exponential solution; an instantaneous root r(t) generally cannot. Substituting y=exp(∫r dt) leaves the normalised residual r'(t), since r²+Pr+Q=0. Require that residual to be zero for an exact claim. The existing paper explicitly discusses instantaneous roots and WKB; `recover_complex` falls back to the real approximation path for its L2 case. This programme does not inherit the solver's broader display labels as proof. [S01](SOURCES.md#s01)

The helical manuscript uses theta(u,v)=v+2*pi*k*u and periodic closure in v. We reuse phase, closure and winding as representation ideas. Its gravitational or dimensional interpretations are not assumptions of a financial model. [S02](SOURCES.md#s02)

## Phase and integer memory

Maintain a lifted phase as a small principal angle and a separate signed integer count:

$$\phi_j=\theta_j+2\pi n_j,\quad 0\leq\theta_j<2\pi,\quad n_j\in\mathbb Z.$$

For a proposed increment delta, divide theta+delta by 2*pi; the quotient updates n and the remainder updates theta. This preserves several forward or backward rotations in one known model step. The unit-circle state is exp(i theta), whose modulus is one. Keeping the integer separate avoids adding a large accumulated time value to a small phase at every step.

This is a signed phase lift for engineering. It does not redefine the manuscript's positive integer winding parameter k. A path's net winding is a topological integer for a closed loop; an open trajectory's counter also depends on the chosen initial phase and branch convention. Counts from the model's own increments are bookkeeping, not independently observed market cycles. Wrapped observations cannot identify unknown complete turns without frequency/sampling assumptions. Near-zero analytic-signal amplitude also makes phase unstable. A causal filter and uncertainty model must handle these conditions.

The initial prototype retains n for audit and testing but uses principal-phase features in its readout. A learned winding-age channel is an explicit future ablation against a plain elapsed-time counter. Floor-based counter updates are not differentiable everywhere; train smooth phases and treat integer crossing events separately or justify an estimator.

## Nonlinear Riccati flow without a chart explosion

Consider dz/dt=a+bz+cz² and let z=v/u. Evolve the homogeneous state h=(u,v) with

$$\frac{dh}{dt}=M h,\qquad M=\begin{pmatrix}-b/2&-c\\a&b/2\end{pmatrix}.$$

Then (v'u-vu')/u²=a+b(v/u)+c(v/u)². Thus this two-component representation exactly reproduces the Riccati equation wherever u is nonzero. At u=0, retain h and switch charts instead of clipping z. Nonzero scalar multiples of h represent the same point in complex projective space CP1. [Related theory: S06](SOURCES.md#s06)

For constant coefficients, h(t+delta)=exp(delta*M)h(t). The induced map is a Möbius transformation. Trace M=0 implies determinant exp(delta*M)=1 analytically. Normalising h after a step changes its representative, not its projective point; it deliberately discards amplitude information, which belongs in a separate state if needed.

A bounded, gauge-invariant observation is

$$s(h)=\frac{(2\operatorname{Re}(\bar u v),\;2\operatorname{Im}(\bar u v),\;|u|^2-|v|^2)}{|u|^2+|v|^2}.$$

Its Euclidean norm is one. This avoids reading a divergent ratio near a pole. The code compares the a=c=1,b=0 flow with tan(t), tests a pole at pi/2, verifies gauge invariance and checks constant-generator composition.

Exact frozen propagation is not an exact solver for changing coefficients. If M depends on time, use time ordering or a justified numerical method. If M depends on h, freezing it at the step start is generally only first order globally. A matrix exponential still uses numerical floating-point arithmetic. Structure preservation and accuracy require distinct tests.

## Selectivity and coupled dynamics

Proposed state space: a product of phase tori, projective channels, real trend/scale coordinates and a probability simplex for regimes. State variables are latent; they need not correspond uniquely to physical market factors.

For phase channel j, one candidate is

$$\frac{d\phi_j}{dt}=\omega_j(m_t)+g_j(x_{\leq t},h_t)\sum_k K_{jk}\sin(\phi_k-\phi_j)+\eta_j(x_{\leq t}),$$

with gate g in [0,1]. Projective coefficients a,b,c are bounded functions of past/present inputs, macro state, other channels and regime weights. Trend/volatility coordinates can use dissipative drifts and stochastic noise. An observation model produces calibrated predictive distributions or risk measures, then a separate decision policy maps those distributions into exposure. Predicting a mean alone is insufficient for tail-risk control.

Mamba's input-dependent selection is useful prior art, but its particular discretisation is not required here. Selectivity does not itself imply a fully nonlinear state transition: if all matrices depend only on exogenous input, the homogeneous recurrence remains linear conditional on that input. State-dependent coefficients and cross-channel couplings produce the stronger nonlinear candidate and may remove convenient parallel-scan structure. Record that computational tradeoff. [S03–S05](SOURCES.md)

## Geometry and PDE research

Riccati/projective charts, Lie flows, differential forms, connections/holonomy and Koopman observables offer distinct tools. Koopman theory uses a linear operator on observables of nonlinear dynamics; rejecting physical linearity does not require rejecting useful linear representations. Finite invariant closures must be demonstrated. [S14](SOURCES.md#s14)

Complex analytic structure can help with poles, residues and Möbius transforms. Typical neural activations and norm-based normalisation are not holomorphic; do not invoke holomorphic theorems for the entire model. Wirtinger derivatives or real two-channel representations need explicit conventions. Banach fixed-point arguments require an invariant complete domain and a genuine contraction bound, not just decreasing training loss.

Possible PDE directions: probability-density evolution via Fokker–Planck; distributional agent belief dynamics; liquidity as a spatial field; coupled reaction/advection/diffusion hypotheses for narrative propagation. Derive state variables, coefficients, boundary conditions and observable links first. Neural operators may learn a family of such equations after a validated synthetic benchmark; they do not establish the equations from market data. [S13](SOURCES.md#s13)

Do not assume markets are Hamiltonian or volume-preserving. Dissipation, jumps, intervention and external information are central candidates. If a Hamiltonian component is isolated, test symplectic methods there; if a density evolves, test positivity and mass; if a diffusion is defined on a manifold, distinguish Itô and Stratonovich formulations and the geometry of the noise.

## Proof and test obligations

Local existence/uniqueness requires appropriate regularity; bounded circle readouts do not guarantee stability of the whole coupled system. Establish parameter constraints, stochastic non-explosion assumptions, identifiability modulo phase/gauge/permutation, state observability, gradient conditioning and training/inference complexity. Measure solver convergence against independent reference integration, not only self-consistency. Evaluate missing events, sampling aliasing, branch transitions and float32 error. None of these claims implies market predictability.
