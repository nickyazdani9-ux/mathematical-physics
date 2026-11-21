# GTOR Research Audit 20 — 22 November 2026 AEDT

## Scope, chronology and deduplication

A selective follow-up to Audits 01–19, examining public arXiv abstracts/HTML and publisher records. On Sunday 22 November, the latest arXiv new listings in gr-qc, quant-ph and hep-th were **Friday 20 November**. Initial submission dates below range from 10 to 19 November; do not misrepresent listing date as submission date. All six contemporary items are preprints; several arXiv DOI registrations are marked pending.

The full set of **18** existing audits dated 4–21 November was read through GitHub. Regex extraction found **90 distinct arXiv identifiers**. None of the six IDs below had previously been cited. Deduplication is at identifier level, not an exhaustive literature search. Previously recorded Provost–Vallée quantum metric work is reused as a mathematical baseline, not recounted as new.

**Principal finding:** several models hold photon critical-curve size fixed while changing radial geometry, ring spacing, accretion or causal structure. GTOR needs multi-channel, identifiable predictions, not geometric analogies. Likewise, network entanglement inequalities and isometric coding are conditional quantum results; neither establishes a physical 3+1 Lorentzian metric on its own.

## Six new-to-ledger papers

### 1. Fixed critical impact, altered photon-ring and disk observables

**A. A. Araújo Filho, N. Heidari, P. H. M. Barros, Carlos F. S. Pereira**, *Accretion disks and photon rings in Palatini Kalb-Ramond gravity*. **Submitted 10 November**. https://arxiv.org/abs/2611.24953 ; full https://arxiv.org/html/2611.24953v1 ; DOI https://doi.org/10.48550/arXiv.2611.24953 .

Under the chosen Palatini action, electric vacuum and test-disk regime, the physical metric is ds²=-(1-2μ/r)dt²+σ²/(1-2μ/r)dr²+r²dΩ², where σ=sqrt(1-q/2), q<2. The independent connection is Levi-Civita for an auxiliary metric. The conserved action-defined M and orbital μ differ; comparisons here hold μ fixed. Then r_ISCO=6μ, r_ph=3μ, b_crit=3sqrt(3)μ independent of q. Yet the physical geometry is conically asymptotic, R(g)=2(1-σ^-2)/r², and the image-ring offset ratio tends to exp(-π/σ). Local thin-disk flux scales as σ^-1, temperature as σ^-1/4; integrated emitted binding efficiency remains 1-sqrt(8/9)≈5.7191%, not necessarily observed escaped radiation. Under quoted universal coupling bounds realistic variations may be extremely small (flux ~1e-11; temperature ~1e-12).

**GTOR test:** hold mass, disk source and observer normalization fixed; calculate shadow impact, photon subring spacing, radial epicyclic frequency, disk emissivity and transfer response simultaneously.

### 2. Horizonless Damour–Solodukhin wormhole mimicry

**G. Alencar, V. H. U. Borralho, T. M. Crispim, Marcos V. de S. Silva**, *Damour--Solodukhin Wormholes: Matter Sources, Observational Constraints, and Accretion Dynamics*. **Submitted 10 November**. https://arxiv.org/abs/2611.24954 ; full https://arxiv.org/html/2611.24954v1 ; DOI https://doi.org/10.48550/arXiv.2611.24954 .

After normalizing time at infinity, ds²=-(1-2M/r)dt²+[1-2(1+λ²)M/r]^-1 dr²+r²dΩ². Throat r_th=2(1+λ²)M has nonzero lapse for λ>0. When λ²<1/2, r=3M is outside the throat and the shadow critical impact retains the Schwarzschild value; at equality the photon sphere meets the throat, a limiting case. The paper reports identical shadow radius for λ²≤1/2 but modified lensing, ring structure and radial accretion speed. Matter supporting the GR solution includes a phantom scalar plus nonlinear electrodynamics. With the specified isentropic barotrope, exterior sonic point requires 0<w<1/(1+4λ²).

**GTOR test:** an identical silhouette cannot establish an event horizon. Compare horizon boundary conditions, lensing delays, fluid dynamics and wave responses together.

### 3. Ultrastatic multi-throat wormholes separate local NEC and global ANEC

**F. S. N. Lobo, M. E. Rodrigues, G. I. Róis**, *Exact ultrastatic multi-throat wormholes: Geometry, energy conditions, and optical cavities*. **Submitted 13 November**. https://arxiv.org/abs/2611.24965 ; full https://arxiv.org/html/2611.24965v1 ; DOI https://doi.org/10.48550/arXiv.2611.24965 (registration pending).

For ds²=-dt²+dℓ²+R(ℓ)²dΩ² under Einstein equations, rho+p_radial=-R''/(4πR). At an extremum of R, the circular null effective-potential second derivative is -2L²R''/R³. A throat (R''>0) violates radial NEC and has an unstable photon orbit; an anti-throat (R''<0) has positive radial NEC and a stable orbit. The authors obtain N+1 throats alternating with N anti-throats. The full two-ended null integral is nevertheless negative: I_ANEC=-1/(4π)∫(R'/R)²dℓ<0 (with asymptotically vanishing boundary term). The identities depend on ultrastatic Einstein assumptions and cannot be imported unchanged into arbitrary modified gravity.

**GTOR test:** compute local energy conditions, averaged null condition, light-ring stability, curvature and perturbative evolution separately. Stable local optics do not establish a globally nonexotic geometry.

### 4. Bounce metrics and interacting perturbation validity

**Jamal Saeed, Rui Pan, Anzhong Wang**, *Strong Coupling of Cosmological Perturbations in mLQC-I: The Hybrid Approach*. **Submitted 18 November**. https://arxiv.org/abs/2611.25180 ; full https://arxiv.org/html/2611.25180v1 ; DOI https://doi.org/10.48550/arXiv.2611.25180 (registration pending).

Using a specified hybrid modified-loop-cosmology canonical prescription, the paper finds enhanced nonlinear coefficients near a pre-bounce quantum-geometric de Sitter-like phase. The particular equilateral cubic diagnostic reaches unity at K≈0.6483 m_P (instantaneous quadratic bounce state) and K≈0.64626 m_P (limited evolving state). These are not universal physical cutoffs. The paper explicitly notes a selected quartic contact cancellation does not determine the full fourth-order amplitude, and backreaction/other sectors are unresolved. No reproduction of those calculations is claimed here.

**GTOR test:** a bounce requires a controlled quadratic/cubic/quartic validity regime and backreaction estimate, not just a regular metric.

### 5. Loop quantum gravity tetrahedra to holographic tensor codes

**D. M. Akbar, F. P. Zen, J. S. Kosasih**, *Holographic Tessellations Tensor Networks {p,q} from Entangled Quantum Tetrahedron in Loop Quantum Gravity*. **Submitted 19 November**. https://arxiv.org/abs/2611.25993 ; full https://arxiv.org/html/2611.25993v1 ; DOI https://doi.org/10.48550/arXiv.2611.25993 (registration pending).

SU(2)-holonomy-dressed four-valent intertwiners define hyperbolic-tessellation network tensors. A {5,4} cell can be completed to a perfect-tensor-type isometry if its microscopic transfer map has full row rank and the kernel can be completed; finite networks need a radial orientation condition. For qualifying cuts, S_boundary=sum_links log(2j_l+1)+S_bulk. Black-hole interior models use *postselected non-isometric encoding*, while Page-curve averages introduce further statistical assumptions. A graph encoding does not automatically supply a causal spacetime metric.

**GTOR test:** check exact rank/isometry and postselection probabilities; then derive a Lorentzian principal symbol, proper clocks and local dynamics independently.

### 6. Graph-theoretic entanglement inequalities

**Masoud Gharahi**, *Entanglement Geometry Beyond Polygons: Graph-Theoretic Entanglement Inequalities*. **Submitted 18 November**. https://arxiv.org/abs/2611.25019 ; full https://arxiv.org/html/2611.25019v1 ; DOI https://doi.org/10.48550/arXiv.2611.25019 (registration pending).

From strong subadditivity and pure-state complementarity, the paper derives weighted/hypergraph entropy polygon bounds, sharp separation coefficients, equality and recovery conditions. Standard control for globally pure multipartite states: 2E(A_k)≤sum_j E(A_j). The stronger results depend on specified graph cuts, weights and relevant parity conditions. This concerns subsystem entropies; it does not establish a measurable absolute global wavefunction phase or a physical metric.

**GTOR test:** specify exactly which subsystem state/cut entropy is being mapped to a geometric variable; verify conventional inequalities before proposing a difference.

## Two new foundational peer-reviewed comparators

**Damour–Solodukhin (2007)**, *Wormholes as black hole foils*, *Phys. Rev. D* 76, 024016, published 27 July 2007. DOI https://doi.org/10.1103/PhysRevD.76.024016 ; https://arxiv.org/abs/0704.2667 . Establishes the longstanding possibility of horizonless black-hole observational mimicry within explicit assumptions; not empirical detection.

**Finster–Grotz (2012)**, *A Lorentzian Quantum Geometry*, *Adv. Theor. Math. Phys.* 16, 1197–1290. DOI https://doi.org/10.4310/ATMP.2012.v16.n4.a3 ; https://arxiv.org/abs/1107.2026 . Causal fermion systems offer a worked theoretical construction of Lorentzian tangent geometry and connections under explicit operator/regularization assumptions; not generic emergence from all graphs.

Existing foundation **not recounted as new:** Provost–Vallée (1980), *Riemannian structure on manifolds of quantum states*, DOI https://doi.org/10.1007/BF02193559 .

## Offline mathematical controls, reproducible script

Tested Python standard-library source: physics/gtor/scripts/gtor_audit20_identifiability_controls.py. **Five limited control families passed**; initial overly strict numerical differencing tolerance was corrected after diagnosing cancellation. Not paper-level replications or GTOR evidence.

1. Palatini q=-0.6,0,+0.6: σ=(1.140175425,1,0.836660027), flux ratios=(0.877058019,1,1.195228609), temperatures=(0.967736409,1,1.045593188), ring-image offset ratios=(0.063586186,0.043213918,0.023402290). Fixed b_crit=5.196152423 for μ=1.
2. Damour–Solodukhin M=1: λ=.5 gives throat 2.5 below photon sphere 3; w=.3 has exterior r_c=3.166667, while w=.55 gives non-exterior r_c=2.409091. For λ=.8, throat 3.28 exceeds photon sphere 3.
3. Independent ultrastatic *toy*, not the paper's particular R_N: R(ℓ)=sqrt(ℓ²+1+2 exp(-ℓ²)); anti-throat at ℓ=0 with radial NEC +0.026525824/stable light ring; throats at ±sqrt(ln 2) with radial NEC -0.040962410/unstable rings. Finite [-40,40] radial integral -0.085087816 matches squared term -0.081111428 plus nonzero endpoint correction -0.003976388; complete-geodesic boundary vanishes only at infinity.
4. Standard qubit projective/Fubini–Study control: g_phi,phi from ray overlap approaches sin²(theta)/4 and is unchanged by separately chosen nonzero complex rescalings.
5. One-party entanglement polygon: GHZ pure three-qubit marginal entropies (1,1,1) bits obey inequality strictly, Bell_AB⊗pure_C (1,1,0) saturates.

## Falsifiable next steps

**High priority:** two explicitly defined Lorentzian models with *the same* normalized lapse A(r)=1-2M/r but distinct radial component B(r). At fixed mass/source/observer, calculate b_crit, image offsets, radial epicyclic frequency, accretion/sonic behavior and a source-aware waveform. If a GTOR proposal changes *only* B and predicts an unsupported change in b_crit, reject that implementation as failing the standard null control. Seek a measurable residual after conventional nuisance fits.

**Quantum/relational:** specify one GTOR Hamiltonian/quantum channel and POVM correlator and derive a measurable prediction invariant under allowed projective rescalings and reference gauge changes. Confirm graph-cut inequalities and encoding rank; explicitly model postselection. A shortest-path graph without derived Lorentzian cone, proper clock or dynamical field equations does not satisfy the spacetime-emergence test.

**Nonlinear/gravity:** specify an action and coupled equations, derive the physical principal symbol and propagation cones, establish curvature/geodesic completeness and a controlled interacting perturbative expansion. Do not equate geometric regularity with hyperbolic or nonlinear viability.

Research-only additions to existing draft PR #2, no merge or independent review bypass, no cloud/spending changes. Previously noted five canonical governance files absent from expected research-subtree locations remain for separate reconciliation; do not invent templates or destroy version history. No genuine owner sign-in/approval blocker arose from this bounded work.
