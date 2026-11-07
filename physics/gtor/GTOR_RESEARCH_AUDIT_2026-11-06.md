# GTOR 2.0 — Research Audit 04 (6 November 2026)

**Status:** Selective literature review plus independent symbolic counterexample checks. Research draft, not peer reviewed or empirical validation. Compared with Audits 01–03 and the main GTOR 2.0 working draft. No cloud provisioning or spending.

## Significant new result A — a constraint on the original helical model

The earlier *GToR revised, six axioms and twenty corollaries* defines the embedding

`R(u,v)=((1+u)cos(v+2πku),(1+u)sin(v+2πku),u)`, `0≤u≤1`, `v mod 2π`.

Its induced line element is

`ds² = [2+(2πk)²(1+u)²]du² + 2(2πk)(1+u)²du dv + (1+u)²dv²`.

**Exact coordinate equivalence.** Set `φ=v+2πku`. Then

`ds²=2du²+(1+u)²dφ²`.

The transformed metric is independent of `k`. More strongly, since `v` spans a complete circle for every fixed `u`, the *embedded surface as a point set* is the same truncated cone for every `k`; changing `k` changes its coordinate/marked spiral lines, not its intrinsic geometry. This was known implicitly in the original paper's diagonalization but its consequences for the *physical winding-number* claim were not called out in Audits 01–03.

The induced metric is positive definite and has **Ricci scalar R=0** on the open cone frustum, independent of `k`. There may be global cone-deficit/endpoint questions if extended, but there is no intrinsic `k`-dependent local curvature. By Sylvester's law of inertia, a nonsingular coordinate transformation cannot convert this Riemannian 2-metric into Lorentzian signature. Calling a slope `dr/dz=1` a physical light cone needs additional causal/dynamical structure.

**Important nuance:** `k` can encode winding of a *distinguished, physically marked curve* or boundary gluing if those structures are independently specified. It is **not** established as an intrinsic topological invariant of the unmarked manifold. Any claim that `k` changes intrinsic observables must supply a physical field, coupling, boundary condition, preferred foliation, or gauge-inequivalent bundle structure.

**Gauge connection and holonomy check.** The paper defines `A=(g_uv/g_vv)du=2πk du=d(2πku)`. Because `u` is globally an interval coordinate, this one-form is exact on the cylinder and its integral along any closed loop is zero. The integral `∫_(0→1) A=2πk` is along an **open segment**, not gauge-invariant closed-loop holonomy; furthermore its exponent is `exp(i2πk)=1` for integer `k`. A nontrivial U(1) observable requires different global identification or a separate physical reference. This distinction should be compared to Berry's gauge-invariant circuit phase [Berry 1984](https://doi.org/10.1098/rspa.1984.0023), not conflated with it.

**Geodesic consistency check.** In the diagonal coordinates, a unit-speed geodesic has conserved angular momentum `L=(1+u)² dφ/ds` and `(du/ds)²=[1-L²/(1+u)²]/2`. The original manuscript *identifies* `L=k` without a derivation. At `k=L=2`, this radial speed squared is negative for every `0≤u<1`: such a unit-speed geodesic cannot traverse the whole cell under those assumptions. Also `∫_0^1 (du/ds)² du = 1/2 - k²/4` becomes negative for `k≥2`, confirming the inconsistency of that integrated kinetic interpretation. Remedy: keep geometric chart winding and conserved geodesic `L` independent, or explicitly derive a new consistent speed/energy normalization.

**Independent reproduction:** see `scripts/gtor_gauge_audit.py` added in the same draft PR. Exact SymPy simplifications check coordinate equivalence, curvature zero, open-segment integral and the `L=k=2` counterexample. These are **mathematical checks**, not statements about which physics nature chooses.

## Significant new result B — new scalar-tensor black-hole no-go preprint

**Shulang Peng, Jun Tao, Mian Zhu and Yi-Fu Cai**, *A Geometric No-Go Theorem for Regular Black Holes in Scalar-Tensor Theories*, [arXiv:2611.06460v1](https://arxiv.org/abs/2611.06460), submitted **4 November 2026** (arXiv DOI form: https://doi.org/10.48550/arXiv.2611.06460 ; registration may be pending). **New, not peer-reviewed; original open HTML reviewed**: https://arxiv.org/html/2611.06460v1.

For a **static, spherically symmetric, two-horizon**, regular black hole sourced solely by a real radial scalar, define the trapped-region Kantowski–Sachs scale ratio `q(τ)=a(τ)/b(τ)>0`, vanishing at both horizon endpoints. Continuity then forces at least one interior maximum, with `d²q/dτ²≤0`. In the paper's defined broad class of turning-point scalar-tensor theories (TPST), equations of motion yield, *at a turning point*,

`[(d²q/dτ²)/q]_(dq/dτ=0) = c_odd² / b²`,

where `c_odd²=F_T/K_T` is the odd-parity tensor propagation-speed squared. A nondegenerate hyperbolic/Lorentzian tensor cone requires positive `c_odd²`, which would instead force `d²q/dτ²>0`. Within the paper's assumptions this excludes a stable two-horizon regular black hole.

**Comparison:** the **3 Nov 2026** Lan–Yang–Khlopov–Krasnov single-horizon finite-area-core preprint was already in Audit 03: https://arxiv.org/abs/2611.04385. The new no-go argument assumes a two-horizon trapped region bounded by zeros of `q`; it **does not directly rule out** the single-horizon finite-area-core construction. Conversely, the latter's scalar-mode stability is not a proof of general nonlinear or tensor-sector stability.

**GTOR decision test:** Before proposing any scalar-tensor interior as a solution, publish `a(τ), b(τ), φ(τ)`, count horizon endpoints, calculate `q` and its turning points, and obtain the relevant odd-parity principal symbol. Either show the no-go hypotheses are absent (specify exactly which), or reject the two-horizon ansatz if it meets them.

## New comparator C — measurable quantum projective geometry beyond Berry curvature

**Alexander Avdoshkin and Fedor K. Popov**, *Beyond the Berry Phase: Extrinsic Geometry of Quantum States*, [Phys. Rev. B 107, 245136 (2023)](https://doi.org/10.1103/PhysRevB.107.245136), open [arXiv:2205.15353](https://arxiv.org/abs/2205.15353). Publication is foundational/previously uncatalogued, **not a new 2026 article**. The paper studies the gauge-invariant three-point Bargmann quantity

`B_123=Tr(P1 P2 P3)=⟨ψ1|ψ2⟩⟨ψ2|ψ3⟩⟨ψ3|ψ1⟩`, `Pj=|ψj⟩⟨ψj|`.

It is invariant under independent `|ψj⟩→exp(iαj)|ψj⟩`. Its infinitesimal expansion contains the quantum metric and Berry curvature; higher-order terms contain additional **gauge-invariant extrinsic geometry** not fixed by metric/curvature alone.

**GTOR implication:** richer geometric information can exist *within* projective Hilbert-space physics; one need not treat an overall phase as measurable to access it. A GTOR phase model should predict an experimentally accessible change in a `B_123`-type interference observable *after* accounting for relative reference frames. Failure: all gauge-invariant predictions match the ordinary projective model despite a new absolute-phase variable.

Foundational context: [Berry, 1984](https://doi.org/10.1098/rspa.1984.0023), geometric phases for adiabatic closed circuits. Already catalogued in Audit 02: Fubini–Study and global-ray invariance; not counted as novel findings.

## New comparator D — operational causal-set horizon test

**Astrid Eichhorn, Pedro Gamito and Nawder Stokes**, *Towards black-hole horizons and geodesic focusing in causal sets*, submitted **7 May 2026**, https://arxiv.org/abs/2605.06813 , arXiv DOI https://doi.org/10.48550/arXiv.2605.06813. Open preprint, not independently verified peer-reviewed. They develop discrete null-geodesic proxies called ladders and fuzzy ladders and recover a sign change of a discrete expansion around a horizon in a **1+1-dimensional toy black-hole causal set**.

**GTOR implication:** this gives a more specific discretization benchmark than generic shortest-path distance. In candidate emergent relational networks: specify an order relation, construct a null proxy/expansion, test sign changes across a known horizon, and check convergence with sampling density. A single 1+1 toy horizon **does not establish** a 3+1 dynamical Einstein limit.

## New optional observational comparator, different asymptotics

**Heng-Yu Chen, Yasuaki Hikida and Yasutaka Koga**, *Quasinormal modes, complex null geodesics and singularity resolution in regular AdS black holes*, submitted **22 Oct 2026**, https://arxiv.org/abs/2610.30915 , arXiv DOI https://doi.org/10.48550/arXiv.2610.30915. Their WKB study of **AdS** regular black holes links changes in complex-geodesic structure to smoothed features of dual-CFT boundary correlators. A promising **model-specific mathematical observable**, not an observed astrophysical signal and not interchangeable with asymptotically flat Kerr GW ringdown.

## Reproducible next steps / pre-registered rejection criteria

1. **Priority 1 — helical gauge audit:** keep `k`, geodesic angular momentum `L` and phase-mode number `ℓ` independent. Search for any unambiguous `k`-dependent intrinsic observable. If none exists on the present unmarked manifold, reclassify `k` as a *chosen helical framing*, not a physical geometric charge. Distinguish local Gaussian curvature, global cone deficit and marked-curve winding.
2. **Priority 2 — quantum discrimination:** compute three-point `B_123` and reference-assisted interference probabilities for the smallest qubit-plus-reference toy model, with no-signalling checks. Specify numerical tolerance and a measurable deviation from the standard model before claiming a new degree of freedom.
3. **Priority 3 — interior dynamics:** construct a **Lorentzian** GTOR metric/action, rather than extrapolating the Riemannian cone. Find scalar invariants, maximal geodesic extension, matter/energy conditions, horizons and tensor/scalar principal symbols; run the new turning-point no-go test where relevant.
4. **Priority 4 — emergent causality:** test causal-set ladder expansion and 1+1 horizon detection against a known toy manifold, with density scaling and failure checks, before attempting 3+1 recovery.
5. **Priority 5 — observational separation:** keep AdS/CFT boundary correlator tests distinct from asymptotically flat gravitational-wave ringdown tests.

## Evidence and constraints

Sources reviewed: the full open arXiv HTML of Peng et al. and Chen et al. where available, arXiv abstracts for Avdoshkin–Popov/Eichhorn et al., the original GTOR TeX, and the publisher abstract of Berry. This is a **selective** survey; not every paper was read in full and no paywalled access is claimed. The gauge findings and geodesic counterexample are direct algebra based on the supplied manuscript, not previously published experimental results. No measured confirmation of GTOR. No paid services, GCP resources, or expenses changed.
