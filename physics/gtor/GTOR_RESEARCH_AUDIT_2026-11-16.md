# GTOR — Research Audit 14 (16 November 2026 AEDT)

**Scope and source discipline.** Selective public-primary-source research, not exhaustive. arXiv gr-qc, quant-ph and hep-th *new* feeds were still headed Friday 13 November when checked 16 November AEDT. Submission dates are **not** listing dates. Primary preprint abstracts and publicly accessible arXiv HTML were inspected; the two foundational journal sources were inspected through publisher pages/abstracts, not inaccessible paid full text. This audit does not claim experimental confirmation of GTOR.

**Deduplication:** The 12 prior Markdown audits dated 4–15 Nov on this same draft branch were fetched (50 unique arXiv IDs in those audit texts). The six selected IDs below and both foundational DOIs were absent. No previously catalogued 2611.15795, 2611.16853, 2611.17281, 2611.17288 or similar papers are counted as new.

## Six previously unlogged primary preprints

1. **Matteo Della Rocca, Paolo Pani, Andrea P. Sanna, _Can One Hear the Absence of a Singularity?_ (submitted 27 Oct 2026).**
   - Primary https://arxiv.org/abs/2610.37028 ; arXiv DOI https://doi.org/10.48550/arXiv.2610.37028 (arXiv registration indicated pending).
   - In the paper's static, spherically symmetric, asymptotically flat regular-black-hole classes with nondegenerate horizons and suitable complex-radial analytic structure, the **test massless-scalar** highly damped quasinormal spectrum has Re(omega_n) → 0 as n→infinity for *every fixed nonzero* regularization parameter. Schwarzschild instead obeys omega_n ~ T_H ln(3) - i 2 pi T_H n (their convention, ignoring subleading constant offsets). For M=1, T_H=1/(8 pi), so the Schwarzschild real-part limit is ln(3)/(8 pi) ≈ 0.0437123941.
   - **Noncommuting limits:** fixing q>0 then n→infinity need not agree with setting q=0 first. It is a spectral singular-perturbation result, **not** a measured GW signature or a proven statement for gravitational perturbations, rotation or degenerate horizons. Extremely large n are highly damped and hard to access observationally. A vanishing asymptotic real part is not an unconditional proof of regularity, as exceptions exist in singular families.
   - **GTOR test:** if GTOR supplies a specific regular f(r), compute complex-radial monodromies / high-overtone test-field spectrum, converge in precision and n, and compare limits with q=0 reference. Do not infer a signal from low-overtone ringdown alone.

2. **Dawei Shen, Yuefeng Song, Jingbo Wan, _Weak null singularities for the Einstein–Vlasov system_ (submitted 11 Nov 2026).**
   - Primary https://arxiv.org/abs/2611.15318 ; public full text https://arxiv.org/html/2611.15318v1 ; arXiv DOI https://doi.org/10.48550/arXiv.2611.15318 (pending registration).
   - For explicitly specified characteristic data, self-gravitating massive *and* massless collisionless matter can develop a weak null singular boundary despite a continuously extendible metric and bounded, continuously extending Ricci curvature and stress-energy. In original double-null coordinates, singular data produce Christoffel symbols not locally square integrable; under added curvature assumptions, the paper proves **Lipschitz inextendibility**. This is not a universal statement about all collapse data.
   - **GTOR tension:** checking that scalar Ricci R or matter stress-energy remains finite is not sufficient to certify strong regularity or well-posed evolution. Check curvature Weyl components, C^0 versus C^{0,1}/C^2 extensions, and affine completeness separately. No contradiction with singularity theorems: they test different notions of singular behavior.

3. **Alireza Allahyari, Shinji Mukohyama, Yi Pang, _Cubic Curvature Deformations to the Kerr Black Hole Shadow_ (submitted 11 Nov 2026).**
   - Primary https://arxiv.org/abs/2611.15876 ; public full text https://arxiv.org/html/2611.15876v1 ; arXiv DOI https://doi.org/10.48550/arXiv.2611.15876 (pending registration).
   - Perturbatively corrected stationary rotating Kerr metrics with **two independent cubic-curvature EFT couplings**, ray-traced without Carter separability. The null Hamiltonian is H=(1/2)g^{mu nu}p_mu p_nu=0 with p_t=-E, p_phi=L_z conserved. Numerical shadows depend on coupling, spin and inclination; the authors recover Kerr when couplings vanish.
   - **GTOR comparison:** numerical shadows distinguish some **exterior** deformations but do not identify an interior endpoint. Under the 15 Nov audit's conditional smooth stationary vacuum Kerr uniqueness, non-Kerr exterior EFT solutions change the field equations, so there is no contradiction. Avoid interpreting every image-level difference as a unique GTOR effect.

4. **Sebastian Brezina, Martin Krššák, _Gauge aspects of teleparallel gravity and Noether's second theorem_ (submitted 12 Nov 2026).**
   - Primary https://arxiv.org/abs/2611.17147 ; public full text https://arxiv.org/html/2611.17147v1 ; arXiv DOI https://doi.org/10.48550/arXiv.2611.17147 (pending registration).
   - For Teleparallel Equivalent of General Relativity (TEGR), authors identify diffeomorphisms and local Lorentz transformations as variational gauge symmetries inducing Noether identities. They argue TEGR's matter coupling is fixed to Levi-Civita coupling (unlike some modified torsion theories). General Noether-II identity schematically: E_a R_i^a + (-1)^d d(E_a wedge S_i^a)=0, from arbitrary local gauge parameters in a differentiable action.
   - **GTOR test:** first specify action, dynamical fields, boundary conditions, gauge algebra and corresponding off-shell Noether identities. Simply relabeling a helical coordinate or adding an exact connection does not constitute a new gauge charge.

5. **Max Niedermaier, Mahdi Sedighi Jafari, _Gravitational observables from a post-Carrollian expansion_ (submitted 10 Nov 2026).**
   - Primary https://arxiv.org/abs/2611.14644 ; public full text https://arxiv.org/html/2611.14644v1 ; arXiv DOI https://doi.org/10.48550/arXiv.2611.14644 (pending registration).
   - The authors construct exact **ultralocal Dirac observables** in the ultrarelativistic/electric Carroll limit of gravity (which remains diffeomorphism invariant), then outline an expansion toward Einstein-gravity Dirac observables organized in spatial gradients. An order-by-order local approximation is **not** a proof of exactly local observables in full GR.
   - **GTOR consequence:** diffeomorphism covariance alone does not settle locality/nonlocality of observables; constraints and dynamics matter. Test candidate GTOR observables against **both** Hamiltonian and spatial-diffeomorphism constraints rather than just coordinate reparameterizations.

6. **Hyunwoo Oh, _Spectral function, OPE, and QNMs detect different singular points of black branes with unequal R-charges_ (submitted 7 Nov 2026).**
   - Primary https://arxiv.org/abs/2611.10998 ; public full text https://arxiv.org/html/2611.10998v1 ; arXiv DOI https://doi.org/10.48550/arXiv.2611.10998 (listed without pending notice).
   - In **STU-model charged AdS black branes** with q1>q2>q3>=0, analytic continuation in R=r^2 produces points R=-q_i. Only R=-q3 is the *physical* singularity in the spacetime considered; others are complex continuation artifacts beyond it. The paper's WKB analysis finds that thermal spectral functions, OPE convergence and separate QNM families respond to different complex optical times T_i.
   - **GTOR comparison:** apparent spectral interior sensitivity depends on **which** observable and analytic structure one uses. These are holographic signatures requiring STU/AdS boundary conditions, not a model-independent view into 3+1D astrophysical interiors.

## Two foundational peer-reviewed controls not previously logged

- **Roger Penrose, _Gravitational Collapse and Space-Time Singularities_, Physical Review Letters 14, 57 (published 18 Jan 1965)**. Publisher https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.14.57 ; DOI https://doi.org/10.1103/PhysRevLett.14.57 . Foundational trapped-surface/causal-geodesic-incompleteness theorem with global and energy assumptions; **not** a proof that all curvature scalars must diverge. Publisher page consulted, not presumed access to a paywalled full paper.
- **Samuel L. Braunstein, Carlton M. Caves, _Statistical distance and the geometry of quantum states_, Physical Review Letters 72, 3439 (published 30 May 1994)**. Publisher https://doi.org/10.1103/PhysRevLett.72.3439 ; DOI 10.1103/PhysRevLett.72.3439 . Establishes operationally motivated quantum statistical distinguishability geometry from optimized measurements; a standard control for Fisher information and Fubini–Study/projective reasoning. Publisher abstract consulted, not presumed access to paid full text.

## Assumptions, equations and potential contradictions

| Framework | Explicit assumptions | Established within scope | What GTOR must not infer |
|---|---|---|---|
| Projective pure-state QM + Braunstein–Caves | A nonzero state vector up to arbitrary complex scaling, standard POVM probabilities | P_psi=psi psi†/(psi†psi); p_j=Tr(P_psi E_j); pure-state F_theta=4(<d psi|d psi>-abs(<psi|d psi>)^2) for normalized representatives | Overall rephasing or normalization alone creates new observable physics |
| Brezina–Krššák TEGR | Differentiable tetrad/connection action, specified local symmetries | Off-shell Noether identities constrain redundant variables and matter couplings | Helical winding is a conserved physical charge absent an action/coupling |
| Niedermaier–Jafari Carroll | Ultralocal electric-gravity limit, Hamiltonian constraints | Exact Dirac observables **in limit**, formal spatial-gradient expansion toward GR | These are exact local 3+1D GR observables |
| Shen–Song–Wan Vlasov | Specified Einstein–Vlasov characteristic data and regularity | C^0 extension and bounded Ricci can coexist with severe connection irregularity | Finite Ricci = globally nonsingular / unique extendible dynamics |
| Della Rocca–Pani–Sanna | Test massless scalar, static spherical nonextremal regular BH, analytic monodromy hypotheses | High-damping Re omega limit differs nonuniformly from Schwarzschild | A low-n astrophysical QNM measurement will detect a regular core |
| Allahyari–Mukohyama–Pang | Cubic-curvature EFT corrections, stationary rotating exterior, perturbative control | Non-Kerr shadows in certain EFTs; reproduce Kerr at zero couplings | This establishes an interior bounce, or violates vacuum Kerr uniqueness |
| Oh STU black branes | STU AdS charged branes, high-frequency/WKB complex continuation | Different boundary observables probe distinct complex singular structures | Those observations are directly available in an ordinary telescope |

**Key reconciliation:** Regular Schwarzschild-like exteriors, modified high-overtone scalar spectra, and nearly unmodified low-overtone ringdown can coexist because limits and observables differ. Also, Ricci finiteness, continuous extendibility and geodesic completeness are not interchangeable. Neither the Carroll model nor TEGR implies that arbitrary GTOR helical winding is a physical degree of freedom.

## Independent offline verification (standard-model null controls ONLY)

Reproducible code: [scripts/gtor_audit14_controls.py](scripts/gtor_audit14_controls.py), run with Python/SymPy/NumPy.

For static spherical areal-gauge metrics ds^2=-f(r)dt^2+dr^2/f(r)+r^2dOmega^2 use
R=-f''-4f'/r+2(1-f)/r^2 and K=(f'')^2+4(f'/r)^2+4((1-f)/r^2)^2.

- For Schwarzschild f_S=1-2M/r, verified R_S=0, K_S=48M^2/r^6; Schwarzschild photon sphere r_ph=3M and critical b=3sqrt(3)M.
- For a **phenomenological Hayward control** f_H=1-2Mr^2/(r^3+2M ell^2), verified R_H(0)=12/ell^2 and K_H(0)=24/ell^4. With M=1, ell=1/2, positive roots f_H=0 at r=0.596968283237 and r=1.854637679718. A smooth center and horizons alone do **not** establish dynamical stability.
- For the projective spinor psi(theta)=(cos(theta/2),sin(theta/2)), verified pure-state QFI F_theta=1 even after rephasing by exp(i alpha theta); a naive non-covariant 4||d psi/dtheta||^2 instead equals 1+4alpha^2, or **197** at alpha=7. Arbitrary nonzero complex scaling changes the normalized projector by only 1.2e-16 in a seeded numerical example.
- Verified the analytic Schwarzschild asymptotic baseline Re(M omega)=ln(3)/(8pi)=0.043712394071. **Did not** compute the highly damped spectra of any regular BH, the full Kerr shadow or the nonlinear Einstein–Vlasov evolution. Hence the checks are baselines, not numerical verification of those papers or of GTOR.

## Next falsification-oriented research

1. **Priority A: Operationally nontrivial GTOR phase.** Define explicit normalized projector, actual Hamiltonian/control loop, measurable POVM, independent references and a nonzero predicted Delta p_j after all representative rephasings/scalings. Require F_theta computed covariantly and compare to the Braunstein–Caves information metric. Falsifier: predicted effect vanishes under gauge correction or is reproduced by an ordinary control Hamiltonian.
2. **Priority B: GTOR field equations before geometry claims.** Supply one action S_GT[g,fields] or exact discrete evolution rule, off-shell gauge identities, physical constraint algebra and Lorentzian solution. A candidate winding k must survive allowed coordinate/bundle transformations in at least one invariant observable.
3. **Priority C: Black-hole regularity hierarchy.** Integrate geodesics with affine parameters, classify C^0/C^{0,1}/C^2 extendibility, calculate Ricci/Weyl/K and physical Jacobi tidal distortion, then apply finite-energy perturbations. A bounded scalar is not a completeness certificate.
4. **Priority D: Spectrum/observables comparison.** Derive a concrete radial wave equation for GTOR's metric and test noncommuting (regularization, overtone) limits. Separately ray trace null geodesics with H=0, varying source/inclination/couplings. If no STU/AdS dual exists, do not promise boundary OPE diagnostics.

**Governance:** Commit to existing **draft** PR #2 only, without merging. No owner-approval or personal sign-in blocker encountered, no cloud provisioning, no spending, no paywalled-access claim. Sources remain selective and unreviewed contemporary preprints have not been promoted to accepted experimental results.
