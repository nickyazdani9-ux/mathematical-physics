"""
Riccati Fixed-Point Structure: Flat Plane vs Helical Manifold
=============================================================

This script verifies the central claim:

  Special functions arise from treating coupled coordinates
  as independent. The Bessel function is the scar left by
  separating r and theta on the flat plane. On the GToR helix,
  r = theta/pi — they were never separated — so no scar forms.

Specifically, we show:

  1. On the flat plane, the Riccati fixed points have TWO regimes
     (power law near origin, plane wave at infinity) — Level 2.

  2. On the helical manifold (original frame, coupling visible),
     the Riccati fixed points have ONE regime: power-law real part
     with CONSTANT imaginary part — Level 1.

  3. The off-diagonal coupling g^{uv} = -pi*k is what pins the
     imaginary part constant.

All derivatives in Leibniz notation. All integrals contraction
integrals where applicable.

Nick Navid Yazdani — March 2026
"""

import sympy as sp

# ============================================================
# UTILITIES
# ============================================================

def p_mac(title, expr=None, break_char="="):
    print()
    print("|-- " + title + " --|")
    print(break_char * 60)
    if expr is not None:
        sp.pretty_print(expr)
    print(break_char * 60)


# ============================================================
# SYMBOLS
# ============================================================

u = sp.Symbol('u', positive=True)
r = sp.Symbol('r', positive=True)
k = sp.Symbol('k', integer=True, positive=True)
n = sp.Symbol('n', integer=True, positive=True)
ell = sp.Symbol('ell', integer=True)
z_sym = sp.Symbol('z')
pi = sp.pi
I = sp.I


# ============================================================
# PART 1: FLAT PLANE — BESSEL
# ============================================================

print("\n" + "=" * 60)
print("  PART 1: FLAT PLANE (r, theta independent)")
print("=" * 60)

# Helmholtz: nabla^2 psi + k_wave^2 psi = 0
# on ds^2 = dr^2 + r^2 d theta^2
# Separate psi = R(r) exp(i n theta)
# Bessel equation:
#   R'' + (1/r) R' + (1 - n^2/r^2) R = 0
# (setting k_wave = 1 for simplicity)

P_bessel = 1/r
Q_bessel = 1 - n**2/r**2

p_mac("Bessel equation in standard form: f'' + P*f' + Q*f = 0")
print(f"  P(r) = 1/r")
print(f"  Q(r) = 1 - n^2/r^2")

# Riccati: dz/dr = -z^2 - P*z - Q
p_mac("Bessel Riccati: dz/dr = -z^2 - (1/r)*z - 1 + n^2/r^2")

# Fixed points: z^2 + P*z + Q = 0
fp_bessel = z_sym**2 + P_bessel*z_sym + Q_bessel
roots_bessel = sp.solve(fp_bessel, z_sym)

p_mac("Bessel fixed points")
for i, root in enumerate(roots_bessel):
    root_simplified = sp.simplify(root)
    print(f"  z*_{i+1} = {root_simplified}")

# Asymptotic analysis
print("\n  --- Asymptotic behaviour ---")
for i, root in enumerate(roots_bessel):
    near_origin = sp.limit(root * r, r, 0)
    at_infinity = sp.limit(root, r, sp.oo)
    print(f"  z*_{i+1} near origin (r -> 0): z ~ {near_origin}/r")
    print(f"  z*_{i+1} at infinity (r -> oo): z -> {at_infinity}")
    print()

print("  VERDICT: Two regimes. Power law (origin) vs plane wave (infinity).")
print("  Incompatible autonomising coordinates. Level 2. Special function.")


# ============================================================
# PART 2: GToR HELICAL MANIFOLD — ORIGINAL (u,v) FRAME
# ============================================================

print("\n" + "=" * 60)
print("  PART 2: GToR HELIX (r = theta/pi, coupling visible)")
print("=" * 60)

# Metric from Axiom 5 embedding:
#   g_uu = 4 pi^2 k^2 (1+u)^2 + 2
#   g_uv = 2 pi k (1+u)^2
#   g_vv = (1+u)^2

w = 1 + u  # shorthand

E_met = 4*pi**2*k**2*w**2 + 2
F_met = 2*pi*k*w**2
G_met = w**2
det_g = sp.simplify(E_met*G_met - F_met**2)
sqrt_g = sp.sqrt(2)*w

p_mac("Induced metric from Axiom 5")
print(f"  g_uu = 4 pi^2 k^2 (1+u)^2 + 2")
print(f"  g_uv = 2 pi k (1+u)^2")
print(f"  g_vv = (1+u)^2")
print(f"  det(g) = {det_g}")
print(f"  sqrt(g) = sqrt(2)*(1+u)")

# Inverse metric
guu_inv = sp.Rational(1, 2)
guv_inv = -pi*k
gvv_inv = sp.simplify(E_met / det_g)

p_mac("Inverse metric")
print(f"  g^uu = 1/2")
print(f"  g^uv = -pi*k")
print(f"  g^vv = {gvv_inv}")

# Full Laplace-Beltrami with psi = f(u) exp(i ell v)
# Build term by term

f_fn = sp.Function('f')
fp = f_fn(u).diff(u)
fpp = f_fn(u).diff(u, 2)

# Bracket 1: d/du [sqrt_g * (g^uu * f' + g^uv * i*ell * f)]
bracket1 = sqrt_g * (guu_inv * fp + guv_inv * I * ell * f_fn(u))
d_bracket1 = sp.diff(bracket1, u)

# Bracket 2: i*ell * sqrt_g * (g^uv * f' + g^vv * i*ell * f)
bracket2 = I * ell * sqrt_g * (guv_inv * fp + gvv_inv * I * ell * f_fn(u))

# LB = (d_bracket1 + bracket2) / sqrt_g = 0
LB = sp.expand((d_bracket1 + bracket2) / sqrt_g)

# Multiply by (1+u)^2 to clear denominators
LB_cleared = sp.expand(LB * w**2)

# Extract coefficients
coeff_fpp = sp.simplify(LB_cleared.coeff(fpp))
rem1 = sp.expand(LB_cleared - coeff_fpp * fpp)
coeff_fp = sp.simplify(rem1.coeff(fp))
rem2 = sp.simplify(rem1 - coeff_fp * fp)
coeff_f = sp.simplify(rem2 / f_fn(u))

p_mac("ODE after multiplying by (1+u)^2")
print(f"  f'' coeff: {coeff_fpp}")
print(f"  f'  coeff: {coeff_fp}")
print(f"  f   coeff: {coeff_f}")

# Standard form coefficients
P_gtor = sp.simplify(coeff_fp / coeff_fpp)
Q_gtor = sp.simplify(coeff_f / coeff_fpp)

p_mac("Standard form: f'' + P(u)*f' + Q(u)*f = 0")
print(f"  P(u) = {P_gtor}")
print(f"  Q(u) = {Q_gtor}")

# Riccati fixed points: z^2 + P*z + Q = 0
fp_gtor = z_sym**2 + P_gtor*z_sym + Q_gtor
roots_gtor = sp.solve(fp_gtor, z_sym)

p_mac("GToR Riccati fixed points (general ell)")
for i, root in enumerate(roots_gtor):
    root_s = sp.simplify(root)
    print(f"  z*_{i+1} = {root_s}")

# On mass shell: ell = k
p_mac("ON MASS SHELL: ell = k")
roots_onshell = [sp.simplify(r.subs(ell, k)) for r in roots_gtor]

for i, root in enumerate(roots_onshell):
    re_part = sp.simplify(sp.re(root))
    im_part = sp.simplify(sp.im(root))
    print(f"  z*_{i+1}:")
    print(f"    Re(z*) = {re_part}")
    print(f"    Im(z*) = {im_part}")
    print(f"    Im depends on u? {im_part.has(u)}")
    print()


# ============================================================
# PART 3: DIRECT COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("  PART 3: STRUCTURAL COMPARISON")
print("=" * 60)

print("""
  BESSEL (flat plane, r and theta independent):
    Re(z*): ~ n/r near origin, -> 0 at infinity
    Im(z*): ~ 0 near origin,   -> +/- 1 at infinity
    Imaginary part CHANGES CHARACTER.
    Two autonomising coordinates needed.
    Level 2. Special function.

  GToR (helix, r = theta/pi, coupling via g^uv = -pi*k):
    Re(z*): c/(1+u)  — power law, single regime
    Im(z*): 2*pi*k^2 — CONSTANT, independent of u
    Imaginary part NEVER CHANGES.
    One autonomising coordinate suffices.
    Level 1. Elementary.

  WHY:
    On the flat plane, oscillation must EMERGE at large r because
    the angular coordinate theta carries no information to the
    radial equation after separation.

    On the helix, r = theta/pi. The angular phase is ALWAYS
    present at every u via the gauge connection g^uv = -pi*k.
    There is no 'infinity regime' to transition to — the plane
    wave is already baked in everywhere.

    The Bessel function is the scar left by cutting r from theta.
    The helix never made the cut. No scar.
""")


# ============================================================
# PART 4: VERIFY CONE FRAME EQUIVALENCE
# ============================================================

print("=" * 60)
print("  PART 4: CONE FRAME CROSS-CHECK")
print("=" * 60)

# In v' = v + 2*pi*k*u coordinates:
# ds^2 = 2 du^2 + (1+u)^2 dv'^2
# Wave eqn gives Euler: (1+u)^2 F'' + (1+u) F' - 2*ell^2 F = 0
# Standard form: F'' + 1/(1+u) F' - 2*ell^2/(1+u)^2 F = 0

P_cone = 1/w
Q_cone = -2*ell**2/w**2

fp_cone = z_sym**2 + P_cone*z_sym + Q_cone
roots_cone = sp.solve(fp_cone, z_sym)

p_mac("Cone frame Riccati fixed points")
for i, root in enumerate(roots_cone):
    root_s = sp.simplify(root)
    print(f"  z*_{i+1} = {root_s}")
    re_part = sp.simplify(sp.re(root_s))
    im_part = sp.simplify(sp.im(root_s))
    print(f"    Re = {re_part}")
    print(f"    Im = {im_part}")
    print()

print("  Both fixed points are REAL and proportional to 1/(1+u).")
print("  Globally autonomous in s = ln(1+u). Level 1.")
print("  Consistent with original frame analysis.")
print()


# ============================================================
# PART 5: VERIFY SOLUTIONS
# ============================================================

print("=" * 60)
print("  PART 5: SOLUTION VERIFICATION")
print("=" * 60)

# Euler ODE: (1+u)^2 F'' + (1+u) F' - 2*ell^2 F = 0
# Solutions: F = (1+u)^{+/- ell*sqrt(2)}

for sign in [1, -1]:
    r_val = sign * ell * sp.sqrt(2)
    F_test = w**r_val
    residual = sp.simplify(
        w**2 * sp.diff(F_test, u, 2)
        + w * sp.diff(F_test, u)
        - 2*ell**2 * F_test
    )
    label = "+" if sign == 1 else "-"
    print(f"  F = (1+u)^({label}ell*sqrt(2)):  residual = {residual}")

print()


# ============================================================
# PART 6: SCHWARZIAN — WHY THE CONE FRAME IS FREE
# ============================================================

print("=" * 60)
print("  PART 6: SCHWARZIAN OF THE AUTONOMISING COORDINATE")
print("=" * 60)

# s = ln(1+u) autonomises the Euler equation
# Schwarzian {s, u} = s'''/s' - (3/2)(s''/s')^2

s_func = sp.ln(1 + u)
s1 = sp.diff(s_func, u)
s2 = sp.diff(s_func, u, 2)
s3 = sp.diff(s_func, u, 3)

schwarzian = sp.simplify(s3/s1 - sp.Rational(3, 2) * (s2/s1)**2)

p_mac("Schwarzian {ln(1+u), u}")
print(f"  {{s, u}} = {schwarzian}")
print(f"  1/2 * {{s, u}} = {sp.simplify(schwarzian/2)}")
print(f"  This is the Langer 1/4 correction: 1/(4*(1+u)^2)")
print()
print("  On the cone, s = ln(1+u) autonomises GLOBALLY.")
print("  Cost: 1/(4*(1+u)^2) — absorbed into the effective potential.")
print("  No second coordinate needed. No seam. No special function.")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print("""
  The helical manifold produces a Riccati flow that is
  globally Level 1 in BOTH frames:

    Cone frame:     Real fixed points, c/(1+u).
                    Autonomous in s = ln(1+u).

    Original frame: Complex fixed points.
                    Re: c/(1+u) (power law, one regime).
                    Im: 2*pi*k^2 (constant, no regime change).

  Bessel requires TWO incompatible coordinates because
  r and theta are independent on the flat plane.

  GToR requires ONE because r = theta/pi — the coupling
  g^uv = -pi*k injects the oscillatory phase at every point.

  Special functions are the cost of pretending coupled
  coordinates are independent.

  The helix pays no cost. The zoo has one animal.
  On the helix, it has no shadow.
""")
