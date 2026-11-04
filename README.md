# mathematical-physics

**Nick Navid Yazdani** — Independent Researcher, Melbourne, Australia

Manuscripts, verification scripts, and computational tools from an ongoing research programme in differential geometry, ODE theory, and geometric physics.

All papers are provisional drafts. Corrections, challenges, and independent verification are actively invited.

---

## Repository Structure

```
mathematical-physics/
├── mathematics/
│   ├── contraction_integral/     # Integration via level-set contraction
│   ├── riccati_odes/             # Special functions paper (EJDE submission)
│   ├── riccati_complete.py       # Full verification script
│   └── wips/                     # Works in progress
├── physics/
│   ├── gtor/                     # Helical manifold / GToR manuscript
│   └── wips/                     # Works in progress
├── engineering/
│   └── riccati/       	          # Riccati ODE solver package
└── LICENSE
```

---

## Mathematics

### The Contraction Integral
An alternative geometric framework for integration based on level-set contraction and the layer-cake decomposition. Central tool: inverse duality — for a strictly monotone continuous *f* on [*a*, *b*],

∫f + ∫f⁻¹ = b·f(b) − a·f(a).

No Riemann sums appear anywhere.

### What Is So Special About "Special Functions"?
The expanded March manuscript opens with the question behind “try an exponential.” Four familiar functions introduce fractional growth. A spring, Euler's logarithmic clock and an outgoing Bessel wave then carry the same calculation to a winding coordinate on a cone. The paper derives the Schwarzian and Langer quarter from elementary differentiation. Its Lean companion checks 40 public theorems, including both endpoint limits of the explicit half-order Bessel wave, recovery up to scale, and the cone's change of phase at a fixed spectral parameter.

[Revised paper (PDF)](mathematics/riccati_odes/riccati_paper.pdf) · [LaTeX source](mathematics/riccati_odes/riccati_paper.tex) · [Lean proofs](mathematics/riccati_odes/RiccatiZoo.lean) · [Runnable CAS examples](https://github.com/nickyazdani9-ux/better_calculator/blob/6e75789b946429b259de33fb6c93be74dfc2b8a8/docs/ODE.md)

The March 2026 draft was submitted to the *Electronic Journal of Differential Equations*. To reproduce the revision with TeX Live and the pinned Lean toolchain, run the following commands from the repository root.

~~~sh
cd mathematics/riccati_odes
latexmk -pdf -halt-on-error riccati_paper.tex
lake exe cache get
lake build
~~~

### Original computational exploration

The March [SciPy and SymPy demonstration](mathematics/riccati_complete.py) accompanies the original draft. For the corrected Bessel branch, Euler potential and coupled exponents, use the November paper and the proof above.

---

## Physics

### Complex Differential Geometry on the Helical Manifold (GToR)
Seven axioms define a helical 2-manifold embedded on an expanding torus. From these axioms: an induced metric with *g*ᵘᵘ = ½; a gauge connection whose holonomy identifies the winding number *k* as topological charge; integer-quantised angular momentum; a per-mode radial ODE (Euler's equation on a cone); inter-cell transfer matrices with det *T* = ½; the closed-form trace tr *T*ₗ = (3/2)cosh(ℓ√2 ln 2); intrinsic flatness (*K* = 0); and a dimensionless ratio 1/(14π²) matching the fine structure constant to 0.83%, with the gap exactly filled by the leading toroidal curvature correction.

One manifold. One parameter. One torus. Contraction integrals throughout. Leibniz notation only.

---

## Engineering

### Riccati ODE computation

[Better Calculator’s differential-equation component](https://github.com/nickyazdani9-ux/better_calculator/pull/16) returns exact bases or explicit Riccati reductions. The [original research prototype](engineering/riccati/) explores symbolic and numerical fixed-point candidates.

---

## Key Results

| Result | Location | Status |
|---|---|---|
| Contraction integrals via inverse duality | mathematics/ | Verified (SymPy) |
| Linear ODE and projective Riccati equation | mathematics/riccati_odes/ | Local equivalence and interval recovery checked in Lean |
| One outgoing half-order Bessel wave connects both endpoint regimes | mathematics/riccati_odes/ | Original ODE and both limits checked in Lean |
| Half-Schwarzian of log *x* is 1/(4*x*²) | mathematics/riccati_odes/ | Checked in Lean |
| Power exponents on the coupled cone satisfy r² = 2ℓ² | mathematics/riccati_odes/ | Checked against the printed ODE in Lean |
| det *T* = ½ (exact, from Wronskian) | physics/ | Proved analytically |
| tr *T*ₗ = (3/2)cosh(ℓ√2 ln 2) | physics/ | Proved analytically |
| *k*-independence of *T*ₗ | physics/ | Proved (gauge diagonalisation + integrality) |
| 1/(14π²) ≈ α to 0.83% | physics/ | Observed; toroidal correction matches α_CODATA |
| *K* = 0 everywhere | physics/ | Proved (developable surface) |
| Mass derived: *m* = π*k*²/2 | physics/ | Proved (volumetric triple integral) |

---

## Dependencies

- Python 3.10+
- SymPy
- SciPy / NumPy
- LaTeX (TeX Live) for manuscript compilation

## Citation

```
N. N. Yazdani, "What Is So Special About 'Special Functions'?"
Expanded author-review edition, November 2026.

N. N. Yazdani, "Complex Differential Geometry on the Helical
Manifold," provisional draft, March 2026.

N. N. Yazdani, "The Contraction Integral: Integration via
Level-Set Contraction," provisional draft, March 2026.
```

## Acknowledgments

This work was developed with AI cognitive partners: Anthropic's Claude, xAI's Grok, and OpenAI's ChatGPT. The mathematics is mine; the conversation that shaped it was ours.

## License

All manuscripts © 2026 Nick Navid Yazdani. Code is released under the MIT License.

## Contact

nickyazdani9@gmail.com
