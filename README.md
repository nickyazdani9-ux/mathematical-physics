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
The substitution *z* = *y*′/*y* converts any second-order linear ODE into a Riccati flow on ℂ. The fixed points of this flow are the solutions. Every classical special function (Bessel, Airy, Hermite, Legendre) is a smooth trajectory connecting two fixed-point regimes whose autonomising coordinates are globally incompatible. The Langer ¼ correction is exactly ½{ln *x*, *x*} — the Schwarzian derivative of the logarithmic coordinate change. A new result shows that off-diagonal metric coupling eliminates the two-regime incompatibility entirely.

**Submitted to the *Electronic Journal of Differential Equations* (EJDE), March 2026.**

### riccati_complete.py
Self-contained verification of every claim in the special functions paper. Covers constant-coefficient, harmonic oscillator, Euler, Bessel, and Airy equations; Schwarzian derivative computation; Hankel function flow on ℂ; and the coupled-coordinate dissolution of Level 2 (Theorem 12.1). Requires SciPy and SymPy.

---

## Physics

### Complex Differential Geometry on the Helical Manifold (GToR)
Seven axioms define a helical 2-manifold embedded on an expanding torus. From these axioms: an induced metric with *g*ᵘᵘ = ½; a gauge connection whose holonomy identifies the winding number *k* as topological charge; integer-quantised angular momentum; a per-mode radial ODE (Euler's equation on a cone); inter-cell transfer matrices with det *T* = ½; the closed-form trace tr *T*ₗ = (3/2)cosh(ℓ√2 ln 2); intrinsic flatness (*K* = 0); and a dimensionless ratio 1/(14π²) matching the fine structure constant to 0.83%, with the gap exactly filled by the leading toroidal curvature correction.

One manifold. One parameter. One torus. Contraction integrals throughout. Leibniz notation only.

---

## Engineering

### Riccati ODE Solver (`riccati/`)
A Python package implementing the Riccati flow approach to solving second-order linear ODEs. Three modes: symbolic (real field), numerical (complex field integration), and coupled-manifold. Classifies equations by autonomy level, extracts fixed points, and recovers solutions via quadrature.

---

## Key Results

| Result | Location | Status |
|---|---|---|
| Contraction integrals via inverse duality | mathematics/ | Verified (SymPy) |
| Special functions = Riccati flow transitions | mathematics/ | Verified; submitted to EJDE |
| Langer ¼ = ½{ln *x*, *x*} | mathematics/ | Verified (SymPy) |
| Coupled coordinates dissolve Level 2 | mathematics/ + physics/ | Verified (symbolic + numerical) |
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
N. N. Yazdani, "What Is So Special About 'Special Functions'?
Riccati Flows, the Schwarzian Derivative, and the Unification
of Classical Solutions," submitted to EJDE, March 2026.

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
