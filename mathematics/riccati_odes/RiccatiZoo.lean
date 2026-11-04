import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Ring

/-!
# Riccati coordinates for scalar second-order homogeneous linear ODEs

A formal companion to Nick Navid Yazdani, *What Is So Special About “Special
Functions”?* (March 2026), §§3–9 and 12. The independent variable is real.
Solutions and coefficients in the Riccati and family theorems may be complex.

Reproduce in a mathlib4 checkout at tag `v4.26.0` (Lean 4.26.0):
`lake exe cache get`, then `lake env lean -DwarningAsError=true RiccatiZoo.lean`.
For the dependency audit, append `#print axioms RiccatiZoo.riccati_iff`.
All family statements are equations for solutions, independent of the choice
of boundary conditions. Named-function asymptotics require additional analysis.
-/

namespace RiccatiZoo

open Filter Topology

noncomputable section

/-- The affine coordinate on the solution direction `(y, y′)`. -/
def logD (y : ℝ → ℂ) (x : ℝ) : ℂ := deriv y x / y x

/-- The pointwise hypotheses for using the `y ≠ 0` chart. -/
structure RegularAt (y : ℝ → ℂ) (x : ℝ) : Prop where
  first : DifferentiableAt ℝ y x
  second : DifferentiableAt ℝ (deriv y) x
  nonzero : y x ≠ 0

variable {y : ℝ → ℂ} {x : ℝ}

/-- Theorem 3.1 before setting either residual to zero. -/
theorem riccati_residual (r : RegularAt y x) (P Q : ℂ) :
    deriv (logD y) x + (logD y x)^2 + P * logD y x + Q =
      (deriv (deriv y) x + P * deriv y x + Q * y x) / y x := by
  unfold logD
  rw [(r.second.hasDerivAt.fun_div r.first.hasDerivAt r.nonzero).deriv]
  field_simp [r.nonzero]
  ring

/-- The same substitution works in both directions at every regular point. -/
theorem riccati_iff (r : RegularAt y x) (P Q : ℂ) :
    deriv (deriv y) x + P * deriv y x + Q * y x = 0 ↔
      deriv (logD y) x = -(logD y x)^2 - P * logD y x - Q := by
  have h := riccati_residual r P Q
  constructor
  · intro heq
    rw [heq, zero_div] at h
    linear_combination h
  · intro heq
    have hz : (deriv (deriv y) x + P * deriv y x + Q * y x) / y x = 0 := by
      rw [← h, heq]
      ring
    exact (div_eq_zero_iff.mp hz).resolve_right r.nonzero

/-- Arbitrary leading coefficient `a`, evaluated at the chosen point. -/
theorem leading_coeff_iff (r : RegularAt y x) {a : ℂ} (ha : a ≠ 0) (b c : ℂ) :
    a * deriv (deriv y) x + b * deriv y x + c * y x = 0 ↔
      deriv (logD y) x = -(logD y x)^2 - (b/a) * logD y x - c/a := by
  rw [← riccati_iff r (b/a) (c/a)]
  have h : deriv (deriv y) x + (b/a) * deriv y x + (c/a) * y x =
      (a * deriv (deriv y) x + b * deriv y x + c * y x)/a := by
    field_simp [ha]
  rw [h, div_eq_zero_iff]
  simp [ha]

/-! ## The equations from the paper, in their original independent variable -/

/-- Constant coefficients, including the paper's example `P = -3, Q = 2`. -/
theorem constant_coeff_iff (r : RegularAt y x) (p q : ℂ) :
    deriv (deriv y) x + p * deriv y x + q * y x = 0 ↔
      deriv (logD y) x = -(logD y x)^2 - p * logD y x - q := riccati_iff r p q

theorem oscillator_iff (r : RegularAt y x) (ω : ℂ) :
    deriv (deriv y) x + ω^2 * y x = 0 ↔
      deriv (logD y) x = -(logD y x)^2 - ω^2 := by
  simpa using riccati_iff r 0 (ω^2)

theorem shifted_euler_iff (r : RegularAt y x) (ℓ : ℂ) (ht : 1 + (x : ℂ) ≠ 0) :
    (1 + (x : ℂ))^2 * deriv (deriv y) x + (1 + (x : ℂ)) * deriv y x +
      (-(2 * ℓ^2)) * y x = 0 ↔
    deriv (logD y) x = -(logD y x)^2 - ((1+(x:ℂ))/(1+(x:ℂ))^2) * logD y x -
      (-(2 * ℓ^2))/(1+(x:ℂ))^2 :=
  leading_coeff_iff r (pow_ne_zero 2 ht) _ _

theorem bessel_iff (r : RegularAt y x) (ν : ℂ) (hx : (x : ℂ) ≠ 0) :
    (x : ℂ)^2 * deriv (deriv y) x + x * deriv y x + ((x : ℂ)^2 - ν^2) * y x = 0 ↔
    deriv (logD y) x = -(logD y x)^2 - logD y x / x - 1 + ν^2/(x : ℂ)^2 := by
  rw [leading_coeff_iff r (pow_ne_zero 2 hx)]
  have h : -(logD y x)^2 - ((x:ℂ)/(x:ℂ)^2) * logD y x - ((x:ℂ)^2-ν^2)/(x:ℂ)^2 =
      -(logD y x)^2 - logD y x / x - 1 + ν^2/(x:ℂ)^2 := by
    field_simp [hx]
    ring
  rw [h]

theorem airy_iff (r : RegularAt y x) :
    deriv (deriv y) x - x * y x = 0 ↔
      deriv (logD y) x = (x : ℂ) - (logD y x)^2 := by
  simpa [sub_eq_add_neg, add_comm, neg_mul] using riccati_iff r 0 (-(x : ℂ))

/-- Physicists' Hermite equation for `Hₙ`, before the Gaussian change of unknown. -/
theorem hermite_iff (r : RegularAt y x) (n : ℂ) :
    deriv (deriv y) x + (-2 * (x : ℂ)) * deriv y x + (2*n) * y x = 0 ↔
    deriv (logD y) x = -(logD y x)^2 - (-2 * (x : ℂ)) * logD y x - 2*n :=
  riccati_iff r _ _

theorem legendre_iff (r : RegularAt y x) (ℓ : ℂ) (hx : 1 - (x : ℂ)^2 ≠ 0) :
    (1 - (x : ℂ)^2) * deriv (deriv y) x + (-2 * (x : ℂ)) * deriv y x +
      (ℓ*(ℓ+1)) * y x = 0 ↔
    deriv (logD y) x = -(logD y x)^2 - ((-2 * (x : ℂ))/(1-(x:ℂ)^2)) * logD y x -
      (ℓ*(ℓ+1))/(1-(x:ℂ)^2) := leading_coeff_iff r hx _ _

/-! ## Recovery, with the integral in Theorem 3.2 -/

/-- The normalized solution has value one at the base point. -/
def recover (z : ℝ → ℂ) (base t : ℝ) : ℂ := Complex.exp (∫ u in base..t, z u)

theorem recover_base (z : ℝ → ℂ) (base : ℝ) : recover z base base = 1 := by simp [recover]

theorem recover_ne_zero (z : ℝ → ℂ) (base t : ℝ) : recover z base t ≠ 0 :=
  Complex.exp_ne_zero _

/-- Recovery on any open interval. The slope may have poles outside that interval. -/
theorem recover_hasDerivAt_on {z : ℝ → ℂ} {I : Set ℝ} (ho : IsOpen I) (hc : I.OrdConnected)
    (hz : ContinuousOn z I) {base t : ℝ} (hb : base ∈ I) (ht : t ∈ I) :
    HasDerivAt (recover z base) (recover z base t * z t) t := by
  exact (intervalIntegral.integral_hasDerivAt_right
    (hz.mono (hc.uIcc_subset hb ht)).intervalIntegrable
    (ContinuousOn.stronglyMeasurableAtFilter ho hz t ht)
    ((hz t ht).continuousAt (ho.mem_nhds ht))).cexp

/-- The Riccati solution on an open interval yields a solution of the linear ODE there. -/
theorem recover_solves_on {z : ℝ → ℂ} {I : Set ℝ} (ho : IsOpen I) (hc : I.OrdConnected)
    (hz : ContinuousOn z I) {base x : ℝ} (hb : base ∈ I) (hx : x ∈ I) (P Q : ℂ)
    (hr : HasDerivAt z (-(z x)^2 - P * z x - Q) x) :
    deriv (deriv (recover z base)) x + P * deriv (recover z base) x + Q * recover z base x = 0 := by
  have hI : ∀ᶠ t in 𝓝 x, t ∈ I := ho.mem_nhds hx
  have hfirst : deriv (recover z base) =ᶠ[𝓝 x] fun t => recover z base t * z t :=
    hI.mono fun t ht => (recover_hasDerivAt_on ho hc hz hb ht).deriv
  have hd := recover_hasDerivAt_on ho hc hz hb hx
  rw [hfirst.deriv_eq, hfirst.self_of_nhds, (hd.fun_mul hr).deriv]
  ring

/-- The one missing multiplicative constant is exactly the initial value `y base`.
This proves recovery for a given solution on the whole chosen interval. -/
theorem reconstruct_on {y z : ℝ → ℂ} {I : Set ℝ} (ho : IsOpen I) (hc : I.OrdConnected)
    (hz : ContinuousOn z I) {base : ℝ} (hb : base ∈ I)
    (hy : ∀ t ∈ I, HasDerivAt y (y t * z t) t) :
    ∀ t ∈ I, y t = y base * recover z base t := by
  let g := recover z base
  have hratio (t : ℝ) (ht : t ∈ I) : HasDerivAt (fun v => y v / g v) 0 t := by
    convert (hy t ht).fun_div (recover_hasDerivAt_on ho hc hz hb ht) (recover_ne_zero z base t) using 1
    change 0 = (y t * z t * g t - y t * (g t * z t)) / (g t)^2
    ring
  have heq : I.EqOn (fun t => y t / g t) (fun _ => y base) := by
    apply ho.eqOn_of_deriv_eq hc.isPreconnected
    · exact fun t ht => (hratio t ht).differentiableAt.differentiableWithinAt
    · exact differentiableOn_const _
    · intro t ht
      rw [(hratio t ht).deriv, (hasDerivAt_const t (y base)).deriv]
    · exact hb
    · simp [g, recover_base]
  intro t ht
  exact (div_eq_iff (recover_ne_zero z base t)).mp (heq ht)

theorem recover_hasDerivAt {z : ℝ → ℂ} (hz : Continuous z) (base t : ℝ) :
    HasDerivAt (recover z base) (recover z base t * z t) t := by
  exact (intervalIntegral.integral_hasDerivAt_right (hz.intervalIntegrable base t)
    hz.aestronglyMeasurable.stronglyMeasurableAtFilter hz.continuousAt).cexp

theorem recover_logD {z : ℝ → ℂ} (hz : Continuous z) (base t : ℝ) :
    logD (recover z base) t = z t := by
  unfold logD
  rw [(recover_hasDerivAt hz base t).deriv]
  field_simp [recover_ne_zero]

/-- Theorem 4.2. A constant slope gives an exact exponential mode. -/
theorem fixed_exponential (k P Q : ℂ) (hk : k^2 + P*k + Q = 0) (t : ℝ) :
    let y := fun s : ℝ => Complex.exp (k * (s : ℂ))
    deriv (deriv y) t + P * deriv y t + Q * y t = 0 := by
  let y := fun s : ℝ => Complex.exp (k * (s : ℂ))
  have hd (s : ℝ) : HasDerivAt y (y s * k) s := by
    simpa [y] using (((hasDerivAt_id s).ofReal_comp.const_mul k).cexp)
  have he : deriv y = fun s => y s * k := funext fun s => (hd s).deriv
  change deriv (deriv y) t + P * deriv y t + Q * y t = 0
  rw [he, ((hd t).mul_const k).deriv]
  linear_combination y t * hk

/-! ## The second projective chart and the logarithmic Bessel coordinate -/

/-- Section 6. At a simple zero of `y`, the reciprocal chart `y / y′` is regular. -/
theorem reciprocal_chart (hy : DifferentiableAt ℝ y x)
    (hdy : DifferentiableAt ℝ (deriv y) x) (h₁ : deriv y x ≠ 0) (P Q : ℂ)
    (hode : deriv (deriv y) x + P * deriv y x + Q * y x = 0) :
    HasDerivAt (fun t => y t / deriv y t)
      (1 + P * (y x / deriv y x) + Q * (y x / deriv y x)^2) x := by
  have h₂ : deriv (deriv y) x = -P * deriv y x - Q * y x := by linear_combination hode
  convert hy.hasDerivAt.fun_div hdy.hasDerivAt h₁ using 1
  rw [h₂]
  field_simp [h₁]
  ring

/-- In `x = exp s`, `w(s) = x z(x)` obeys `dw/ds = c - w² - k x²`.
The Bessel equation has `c = ν², k = 1`. The Euler equation, after the
translation `x = 1+u` of §5.3, has `c = 2ℓ², k = 0`. The frozen limiting
Bessel vector field as `x → 0+` thus has equilibrium equation `w² = ν²`. -/
theorem euler_bessel_log_flow {z : ℝ → ℂ} (s : ℝ) (c k : ℂ)
    (hz : HasDerivAt z (-(z (Real.exp s))^2 - z (Real.exp s) / (Real.exp s : ℂ) - k +
      c / (Real.exp s : ℂ)^2) (Real.exp s)) :
    HasDerivAt (fun t => (Real.exp t : ℂ) * z (Real.exp t))
      (c - ((Real.exp s : ℂ) * z (Real.exp s))^2 - k * (Real.exp s : ℂ)^2) s := by
  have he := Real.hasDerivAt_exp s
  have hx : (Real.exp s : ℂ) ≠ 0 := by exact_mod_cast Real.exp_ne_zero s
  convert he.ofReal_comp.fun_mul (hz.scomp s he) using 1
  simp only [Function.comp_apply, Complex.real_smul]
  field_simp [hx]
  ring

/-! ## Normal form and coordinate autonomy -/

/-- Riccati form of Theorem 7.1. The shift `w = z + P/2` removes the linear
term and gives the effective potential `Q - P′/2 - P²/4`. -/
theorem normal_form_slope {z P : ℝ → ℂ} (P₁ Q : ℂ)
    (hz : HasDerivAt z (-(z x)^2 - P x * z x - Q) x) (hp : HasDerivAt P P₁ x) :
    HasDerivAt (fun t => z t + P t/2)
      (-(z x + P x/2)^2 - (Q - P₁/2 - (P x)^2/4)) x := by
  convert hz.add (hp.div_const 2) using 1
  ring

/-- The Schwarzian in terms of the first, second and third derivatives. -/
def schwarzian (d e j : ℝ) : ℝ := j/d - (3/2 : ℝ) * (e/d)^2

/-- Theorems 7.2 and 8.2 use the potential in the new independent variable. -/
def transformedPotential (q d e j : ℝ) : ℝ := (q - schwarzian d e j / 2) / d^2

theorem autonomy_iff (q c d e j : ℝ) (hd : d ≠ 0) :
    transformedPotential q d e j = c ↔ q = c * d^2 + schwarzian d e j / 2 := by
  unfold transformedPotential
  rw [div_eq_iff (pow_ne_zero 2 hd)]
  constructor <;> intro h <;> linear_combination h

/-- The amplitude multiplying `H(s(x))` in the Liouville pullback. -/
def weight (d : ℝ → ℝ) (t : ℝ) : ℝ := d t ^ (-(1/2) : ℝ)

private theorem weight_deriv {d : ℝ → ℝ} {v t : ℝ} (hd : HasDerivAt d v t) (hn : d t ≠ 0) :
    HasDerivAt (weight d) (weight d t * (-(v / d t) / 2)) t := by
  unfold weight
  convert hd.rpow_const (p := (-(1/2) : ℝ)) (Or.inl hn) using 1
  rw [Real.rpow_sub_one hn]
  ring

/-- Theorem 7.2 as an identity of actual second derivatives. `d,e,j` give
`s′,s″,s‴`, and `V,A` give `H′,H″`. The neighborhood hypotheses allow open
coordinate patches, and `d x > 0` selects the real, orientation-preserving chart. -/
theorem schwarzian_pullback {s d e H V : ℝ → ℝ} {j A q x : ℝ}
    (hs : ∀ᶠ t in 𝓝 x, HasDerivAt s (d t) t)
    (hd : ∀ᶠ t in 𝓝 x, HasDerivAt d (e t) t) (he : HasDerivAt e j x)
    (hpos : 0 < d x) (hH : ∀ᶠ v in 𝓝 (s x), HasDerivAt H (V v) v)
    (hV : HasDerivAt V A (s x)) :
    let h := fun t => weight d t * H (s t)
    deriv (deriv h) x + q * h x = weight d x * (d x)^2 *
      (A + transformedPotential q (d x) (e x) j * H (s x)) := by
  let g := fun t => -(e t / d t) / 2
  let h := fun t => weight d t * H (s t)
  let v := fun t => weight d t * (g t * H (s t) + V (s t) * d t)
  have hsx := hs.self_of_nhds
  have hdx := hd.self_of_nhds
  have hHx := hH.self_of_nhds
  have h₀ : d x ≠ 0 := ne_of_gt hpos
  have hdn : ∀ᶠ t in 𝓝 x, d t ≠ 0 := hdx.continuousAt.eventually_ne h₀
  have hHs : ∀ᶠ t in 𝓝 x, HasDerivAt H (V (s t)) (s t) := hsx.continuousAt hH
  have hfirst : deriv h =ᶠ[𝓝 x] v := by
    filter_upwards [hs, hd, hdn, hHs] with t hst hdt hnt hHt
    have hr := ((weight_deriv hdt hnt).fun_mul (hHt.comp t hst)).deriv
    change deriv h t = v t
    calc
      deriv h t = _ := hr
      _ = v t := by simp only [v, g, Function.comp_apply]; ring
  have hg : HasDerivAt g (-((j * d x - e x * e x)/(d x)^2)/2) x :=
    ((he.fun_div hdx h₀).neg.div_const 2)
  have dv := (weight_deriv hdx h₀).fun_mul
    ((hg.fun_mul (hHx.comp x hsx)).add ((hV.comp x hsx).fun_mul hdx))
  change deriv (deriv h) x + q * h x = _
  rw [hfirst.deriv_eq]
  have dv' : deriv v x = _ := dv.deriv
  rw [dv']
  simp only [h, g, Function.comp_apply, Pi.add_apply, transformedPotential, schwarzian]
  field_simp [h₀]
  ring

/-- The coordinate `s(u) = log(1+u)` and its first three derivatives, followed
by the exact half-Schwarzian (the Langer term). -/
theorem log_langer (u : ℝ) (hu : 0 < 1 + u) :
    HasDerivAt (fun t : ℝ => Real.log (1+t)) (1/(1+u)) u ∧
    HasDerivAt (fun t : ℝ => 1/(1+t)) (-1/(1+u)^2) u ∧
    HasDerivAt (fun t : ℝ => -1/(1+t)^2) (2/(1+u)^3) u ∧
    schwarzian (1/(1+u)) (-1/(1+u)^2) (2/(1+u)^3) / 2 = 1/(4*(1+u)^2) := by
  have h₀ : 1+u ≠ 0 := ne_of_gt hu
  have ht : HasDerivAt (fun t : ℝ => 1+t) 1 u := (hasDerivAt_id u).const_add 1
  refine ⟨?_, ?_, ?_, ?_⟩
  · simpa [one_div] using (Real.hasDerivAt_log h₀).comp u ht
  · convert (hasDerivAt_const u (1:ℝ)).fun_div ht h₀ using 1
    ring
  · convert (hasDerivAt_const u (-1:ℝ)).fun_div (ht.fun_pow 2) (pow_ne_zero 2 h₀) using 1
    field_simp [h₀]
    ring
  · unfold schwarzian
    field_simp [h₀]
    ring

/-- For §5.3 set `t = 1+u`. In the paper's convention `h″ + q h = 0`,
the unshifted effective potential in §7 has the sign and factor shown below.
The logarithmic coordinate then makes it the constant `-2ℓ²`. -/
theorem euler_effective_potential (t ℓ : ℝ) (ht : t ≠ 0) :
    let P := 1 / t
    let P₁ := -1 / t^2
    let Q := -(2 * ℓ^2) / t^2
    let q := Q - P₁/2 - P^2/4
    q = (1/4 - 2 * ℓ^2) / t^2 ∧
      transformedPotential q (1/t) (-1/t^2) (2/t^3) = -(2 * ℓ^2) := by
  dsimp only [transformedPotential, schwarzian]
  constructor <;> field_simp [ht] <;> ring

/-- Theorem 7.1 on `y = μ h` itself. The gauge is the exponential of the
integral of `-P/2`. The slope and curvature of `h` are `v` and `a`. -/
theorem normal_form_reduction {P h v : ℝ → ℂ} {a P₁ Q : ℂ} {x : ℝ} (base : ℝ)
    (hP : Continuous P) (hP₁ : HasDerivAt P P₁ x)
    (hh : ∀ᶠ t in 𝓝 x, HasDerivAt h (v t) t) (hv : HasDerivAt v a x) :
    let μ := recover (fun t => -P t / 2) base
    let y := fun t => μ t * h t
    deriv (deriv y) x + P x * deriv y x + Q * y x =
      μ x * (a + (Q - P₁/2 - (P x)^2/4) * h x) := by
  let g := fun t => -P t / 2
  let μ := recover g base
  let y := fun t => μ t * h t
  have hg : Continuous g := hP.neg.div_const 2
  have hm (t : ℝ) : HasDerivAt μ (μ t * g t) t := recover_hasDerivAt hg base t
  have hfirst : deriv y =ᶠ[𝓝 x] fun t => μ t * g t * h t + μ t * v t :=
    hh.mono fun t ht => ((hm t).fun_mul ht).deriv
  have hd := (((hm x).fun_mul (hP₁.neg.div_const 2)).fun_mul hh.self_of_nhds).add
    ((hm x).fun_mul hv)
  change deriv (deriv y) x + P x * deriv y x + Q * y x = _
  rw [hfirst.deriv_eq, hfirst.self_of_nhds]
  have hd' : deriv (fun t => μ t * g t * h t + μ t * v t) x = _ := hd.deriv
  rw [hd']
  dsimp only [y, μ, g, Pi.add_apply, Pi.neg_apply, Pi.div_apply, Pi.ofNat_apply]
  ring

/-! ## Coupled coordinates (§12): a frozen root and a solution slope

Write `t = 1+u` and `κ = 2πiℓm`. The coefficients printed in §12.2 then
have the following short forms. The distinction between the polynomial with
coefficients frozen at `t` and the differential equation matters in §12.5.
-/

def coupledP (t κ : ℂ) : ℂ := 1/t - 2*κ

def coupledQ (t κ ℓ : ℂ) : ℂ := κ^2 - κ/t - 2*ℓ^2/t^2

def coupledSlope (κ r : ℂ) (u : ℝ) : ℂ := κ + r / (1 + (u : ℂ))

theorem coupled_frozen_polynomial (t κ r ℓ : ℂ) (ht : t ≠ 0) :
    (κ + r/t)^2 + coupledP t κ * (κ + r/t) + coupledQ t κ ℓ =
    (r^2 + r - 2*ℓ^2)/t^2 := by
  unfold coupledP coupledQ
  field_simp [ht]
  ring

/-- Theorem 12.1. Both real roots `r` of the frozen quadratic on `ℓ=m` have
imaginary part `2πm²` in the `z` coordinate. -/
theorem coupled_frozen_root (u m r : ℝ) (hu : 0 < 1+u) (hr : r^2 + r = 2*m^2) :
    let t : ℂ := 1 + (u : ℂ)
    let κ : ℂ := ((2 * Real.pi * m^2 : ℝ) : ℂ) * Complex.I
    let z := coupledSlope κ (r : ℂ) u
    z^2 + coupledP t κ * z + coupledQ t κ m = 0 ∧ z.im = 2*Real.pi*m^2 := by
  have h₀ : 1 + (u : ℂ) ≠ 0 := by exact_mod_cast (ne_of_gt hu)
  have hc : (r : ℂ)^2 + r = 2*(m : ℂ)^2 := by exact_mod_cast hr
  dsimp only [coupledSlope]
  constructor
  · rw [coupled_frozen_polynomial _ _ _ _ h₀, hc, sub_self, zero_div]
  · rw [show (1:ℂ)+(u:ℂ) = ((1+u:ℝ):ℂ) by norm_cast]
    simp only [Complex.add_im, Complex.mul_I_im, Complex.ofReal_re, Complex.div_ofReal_im,
      Complex.ofReal_im, zero_div, add_zero]

theorem coupled_slope_deriv (κ r : ℂ) {u : ℝ} (ht : 1 + (u : ℂ) ≠ 0) :
    HasDerivAt (coupledSlope κ r) (-r/(1+(u:ℂ))^2) u := by
  have hd : HasDerivAt (fun t : ℝ => 1+(t:ℂ)) (1:ℂ) u := by
    simpa using (hasDerivAt_id u).ofReal_comp.const_add (1:ℂ)
  convert (((hasDerivAt_const u r).fun_div hd ht).const_add κ) using 1
  ring

/-- The derivative of `r/t` removes the extra `+r` in the frozen polynomial. -/
theorem coupled_flow_residual (κ r ℓ : ℂ) {u : ℝ} (ht : 1+(u:ℂ) ≠ 0) :
    let t := 1+(u:ℂ)
    let z := coupledSlope κ r
    deriv z u + (z u)^2 + coupledP t κ * z u + coupledQ t κ ℓ = (r^2 - 2*ℓ^2)/t^2 := by
  dsimp only
  rw [(coupled_slope_deriv κ r ht).deriv]
  unfold coupledSlope coupledP coupledQ
  field_simp [ht]
  ring

/-- `exp(κu + r log(1+u))` is the unambiguous positive-base power times phase. -/
def coupledMode (κ r : ℂ) (u : ℝ) : ℂ :=
  Complex.exp (κ * (u : ℂ) + r * (Real.log (1+u) : ℂ))

private theorem coupled_mode_deriv (κ r : ℂ) {u : ℝ} (hu : 0 < 1+u) :
    HasDerivAt (coupledMode κ r) (coupledMode κ r u * coupledSlope κ r u) u := by
  have hi : HasDerivAt (fun t : ℝ => (t:ℂ)) 1 u := by simpa using (hasDerivAt_id u).ofReal_comp
  have hl := (log_langer u hu).1.ofReal_comp
  convert ((hi.const_mul κ).add (hl.const_mul r)).cexp using 1
  simp [coupledMode, coupledSlope, div_eq_mul_inv]

/-- Exact correction to §12.5. The elementary mode solves the printed ODE
precisely when `r² = 2ℓ²`. In particular the exponents on resonance are
`±√2 m`. The frozen polynomial in Theorem 12.1 has `r² + r = 2m²`. -/
theorem coupled_mode_solves_iff (κ r ℓ : ℂ) {u : ℝ} (hu : 0 < 1+u) :
    let t := 1+(u:ℂ)
    let y := coupledMode κ r
    deriv (deriv y) u + coupledP t κ * deriv y u + coupledQ t κ ℓ * y u = 0 ↔
      r^2 = 2*ℓ^2 := by
  let y := coupledMode κ r
  let z := coupledSlope κ r
  have ht : 1+(u:ℂ) ≠ 0 := by exact_mod_cast (ne_of_gt hu)
  have hy : HasDerivAt y (y u * z u) u := coupled_mode_deriv κ r hu
  have hn : ∀ᶠ v : ℝ in 𝓝 u, 0 < 1+v :=
    (isOpen_lt continuous_const (continuous_const.add continuous_id)).mem_nhds hu
  have hy' : deriv y =ᶠ[𝓝 u] fun v => y v * z v :=
    hn.mono fun v hv => (coupled_mode_deriv κ r hv).deriv
  have hz := coupled_slope_deriv κ r ht
  have hsecond : deriv (deriv y) u = (y u * z u) * z u + y u * (-r/(1+(u:ℂ))^2) :=
    hy'.deriv_eq.trans (hy.fun_mul hz).deriv
  have hres := coupled_flow_residual κ r ℓ ht
  dsimp only at hres
  rw [hz.deriv] at hres
  have h : deriv (deriv y) u + coupledP (1+(u:ℂ)) κ * deriv y u +
      coupledQ (1+(u:ℂ)) κ ℓ * y u = y u * ((r^2 - 2*ℓ^2)/(1+(u:ℂ))^2) := by
    rw [hsecond, hy.deriv, ← hres]
    dsimp only [z]
    ring
  change deriv (deriv y) u + _ * deriv y u + _ * y u = 0 ↔ _
  rw [h, mul_eq_zero, div_eq_zero_iff]
  have hyn : y u ≠ 0 := Complex.exp_ne_zero _
  simp [hyn, pow_ne_zero 2 ht, sub_eq_zero]

end
end RiccatiZoo
