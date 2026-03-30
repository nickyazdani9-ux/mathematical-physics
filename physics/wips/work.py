import sympy as sp

R, X, Y, Z, u,v,p, Thet, theta, omega = sp.symbols('R X Y Z u v p Theta theta omega', positive=True, real=True)
k = sp.Symbol('k', positive=True, real=True, integer=True)

func_theta = sp.Function(Thet)(u, v)

func_X = sp.Eq(X, (u + 1) * sp.cos(v + 2 * sp.pi * k * u),eval=False)
func_Y = sp.Eq(Y, (u + 1) * sp.sin(v + 2 * sp.pi * k * u),eval=False)
func_Z = sp.Eq(Z, u, eval=False)

func_R = sp.Eq(R, (func_X.rhs, func_Y.rhs, func_Z.rhs), eval=False)

dR_du_X = sp.simplify(sp.expand(func_R.rhs[0].diff(u, 1)))
dR_du_Y = sp.simplify(sp.expand(func_R.rhs[1].diff(u, 1)))
dR_du_Z = sp.simplify(sp.expand(func_R.rhs[2].diff(u, 1)))

print('=' * 60)
print('Partial of R w.r.t u')
dR_du = sp.Eq(sp.Symbol('dR/du'), (dR_du_X, dR_du_Y, dR_du_Z), eval=False)
print(dR_du)
print('=' * 60)
print()

dR_dv_X = sp.simplify(func_R.rhs[0].diff(v, 1))
dR_dv_Y = sp.simplify(func_R.rhs[1].diff(v, 1))
dR_dv_Z = sp.simplify(func_R.rhs[2].diff(v, 1))

print('=' * 60)
print('Partial of R w.r.t v')
dR_dv = sp.Eq(sp.Symbol('dR/dv'), (dR_dv_X, dR_dv_Y, dR_dv_Z), eval=False)
print(dR_dv)
print('=' * 60)
print()

Ru_vec = sp.Matrix([sp.collect(dR_du.rhs[0], k), sp.collect(dR_du.rhs[1], k), sp.collect(dR_du.rhs[2], k)])

Rv_vec = sp.Matrix([sp.collect(dR_dv.rhs[0],k), sp.collect(dR_dv.rhs[1],k), sp.collect(dR_dv.rhs[2], k)])

print(Ru_vec)
print()
print("=" * 60)
print()
print(Rv_vec)
print("="*60)

eqn_E = sp.Symbol('E_M')

E = Ru_vec.dot(Ru_vec.T)
E = sp.trigsimp(E)
E_xpr = sp.Eq(eqn_E, E)
print(E_xpr.lhs, ' = ', E_xpr.rhs) 

eqn_F = sp.Symbol('F')

F = Ru_vec.dot(Rv_vec.T)
F = sp.trigsimp(sp.collect(sp.collect(F, sp.pi), k))
F_xpr = sp.Eq(eqn_F, F)
print(F_xpr.lhs, ' = ', F_xpr.rhs) 

eqn_G = sp.Symbol('G')

G = Rv_vec.dot(Rv_vec.T)
G_xpr = sp.Eq(eqn_G, G)
print(G_xpr.lhs, ' = ', sp.trigsimp(G_xpr.rhs))

induced_metric_det = E_xpr.rhs * G_xpr.rhs - F_xpr.rhs**2

print("Det = ", sp.sqrt(sp.factor(sp.simplify(induced_metric_det))))

g_matrix = sp.Matrix([[E_xpr.rhs, F_xpr.rhs], [F_xpr.rhs, G_xpr.rhs]])
g_inv = g_matrix.inv()

# --- Corollary 2 --- #

# Gauge connection: the off-diagonal metric term 
# divided by the v-v term. This tells you how much
# phase (v-direction) gets dragged per step in u.
# It's like asking: "if I walk in u, how much does v 
# get pulled along for free?"
A_u = sp.simplify(F / G)
print(f"A_u = F/G = {A_u}")

# Holonomy: total phase accumulated over one cell.
# Integrate the connection over u = 0 to 1.
# This is the total "free rotation" the helix 
# forces on you as you traverse one cell.
Phi_cell = sp.integrate(A_u, (u, 0, 1))
print(f"Holonomy = {Phi_cell}")

# ------------------- #

coords = [u, v]
Gamma = {}

for i in range(2):
    for j in range(2):
        for l in range(2):
            val = sp.Rational(1,2) * sum(
                g_inv[i, m] * (
                    sp.diff(g_matrix[m, j], coords[l]) +
                    sp.diff(g_matrix[m, l], coords[j]) -
                    sp.diff(g_matrix[j, l], coords[m])
                ) for m in range(2)
            )
            Gamma[(i,j,l)] = (sp.trigsimp(sp.simplify(sp.factor(sp.cancel(sp.numer(val)))))) /  (sp.trigsimp(sp.simplify(sp.factor(sp.cancel(sp.denom(val))))))

labels = ['u', 'v']

for (i, j, l), val in Gamma.items():
    print(f'Gamma_{labels[i]}_{labels[j]}_{labels[l]}: ', sp.simplify(val))

print()

from sympy import pi, Rational, simplify

# Test point
u0 = Rational(1, 2)
k0 = 1
test = {u: u0, k: k0}

# Paper's Corollary 3 expressions:
paper = {
    'Gamma^u_uu': -2*pi**2*k0*(u0+1),
    'Gamma^u_uv': -pi*k0*(u0+1),
    'Gamma^u_vv': -u0/2 - Rational(1,2),
    'Gamma^v_uu': 4*pi*k0*(pi**2*k0**2*u0**2 + 2*pi**2*k0**2*u0 + pi**2*k0**2 + 1)/(u0+1),
    'Gamma^v_uv': (2*pi**2*k0**2*u0**2 + 4*pi**2*k0**2*u0 + 2*pi**2*k0**2 + 1)/(u0+1),
    'Gamma^v_vv': pi*k0*(u0+1),
}

computed = {
    'Gamma^u_uu': Gamma[(0,0,0)].subs(test),
    'Gamma^u_uv': Gamma[(0,0,1)].subs(test),
    'Gamma^u_vv': Gamma[(0,1,1)].subs(test),
    'Gamma^v_uu': Gamma[(1,0,0)].subs(test),
    'Gamma^v_uv': Gamma[(1,0,1)].subs(test),
    'Gamma^v_vv': Gamma[(1,1,1)].subs(test),
}

for name in paper:
    diff = simplify(paper[name] - computed[name])
    print(f'{name}: paper={paper[name].evalf():.8f}')
    print(f'{name}: code={computed[name].evalf():.8f}')
    print()

v_prime = sp.Symbol('v_prime')
# Cone metric after gauge diag
E_cone = sp.Integer(2)
G_cone = (1 + u)**2

g_cone = sp.Matrix([[E_cone, 0], [0, G_cone]])
g_cone_inv = g_cone.inv()

coords_cone = [u, v]

Gamma_cone = {}
for i in range(2):
    for j in range(2):
        for l in range(2):
            val = sp.Rational(1,2) * sum(
                g_cone_inv[i, m] * (
                    sp.diff(g_cone[m, j], coords_cone[l]) +
                    sp.diff(g_cone[m, l], coords_cone[j]) -
                    sp.diff(g_cone[j, l], coords_cone[m])
                ) for m in range(2)
            )
            Gamma_cone[(i,j,l)] = sp.simplify(val)

# Print nonzero
for key, val in Gamma_cone.items():
    if val != 0:
        print(f'Gamma_cone{key} = {val}')

# Build geodesic ODEs
s = sp.Symbol('s')
u_s = sp.Function('u')(s)
vp_s = sp.Function('vp')(s)
funcs = [u_s, vp_s]
coords_map = {u: u_s, v_prime: vp_s}

for i in range(2):
    accel = funcs[i].diff(s, 2)
    conn = sum(
        Gamma_cone[(i,j,l)].subs(coords_map) * funcs[j].diff(s) * funcs[l].diff(s)
        for j in range(2) for l in range(2)
    )
    print(f'\nGeodesic {i}: {sp.simplify(accel + conn)} = 0')

# Use angular momentum conservation (manual step here because dsolve choked)
L = sp.Symbol('L', positive=True)
ode_u = sp.Eq(u_s.diff(s, 2), L**2 / (2*(1 + u_s)**3))
print('\nReduced ODE:')
print(ode_u)

p = sp.Function('p')(u)
ode_reduced = sp.Eq(p * p.diff(u), L**2 / (2*(1+u)**3))
sol = sp.dsolve(ode_reduced)
sol_simped = sp.simplify(sol[0].rhs)
print('\nSolved ODE (General):')
print(sol[0].lhs,"=", sol_simped)

C1 = sp.Symbol('C1')
p_squared = sp.expand(sol_simped**2)
print('\np^2 (raw) =', p_squared)

# Extract C1 from the normalization constraint
norm_constraint = sp.Eq(2*p_squared + L**2/(1+u)**2, 1)
C1_val = sp.solve(norm_constraint, C1)
print('\nC1 from normalization =', C1_val)

# Substitute back
p_squared_final = sp.simplify(p_squared.subs(C1, C1_val[0]))

# Verify against expected: 1/2 - L^2/(2*(1+u)^2)
expected = sp.Rational(1,2) - L**2 / (2*(1+u)**2)
p_squared_final = sp.expand(sp.numer((p_squared_final))) / sp.factor((sp.denom(p_squared_final)))

print("ODE solution (not neat): ", p_squared_final)
print()

print("Manually factor quadratic in numerator:")
simped_ode_sol =sp.Eq(sol[0].lhs, ((-L**2 + (u + 1)**2)/(2*(u + 1)**2)))
print(simped_ode_sol.lhs, "=", simped_ode_sol.rhs)

diff = sp.simplify(p_squared_final - expected)
print('\nDifference from expected:', diff)

# Corollary 4
# Metric determinant: det(g) = E*G - F^2
det_g = sp.simplify(E*G - F**2)
sqrt_g = sp.simplify(sp.sqrt(det_g))
print(f"det(g) = {det_g}")
print(f"sqrt(g) = {sqrt_g}")

# Cell action: integrate sqrt(g) over the full cell
# v goes 0 to 2pi (one trip around the circle)
# u goes 0 to 1 (one cell length)
# This is just the surface area — how much "stuff" 
# one cell of helix contains geometrically.
S = sp.integrate(sqrt_g, (u, 0, 1), (v, 0, 2*sp.pi))
S_simplified = sp.simplify(S)
print(f"Cell action S = {S_simplified}")
print(f"Expected: 3*sqrt(2)*pi = {3*sp.sqrt(2)*sp.pi}")
print(f"Match: {sp.simplify(S_simplified - 3*sp.sqrt(2)*sp.pi) == 0}")

# Cor. 5
print()
#Phase energy density: this is the geometric potential
# that each mode feels. It's k^2 over 2(1+u)^2 — 
# stronger at the start of the cell (u=0), weaker at 
# the end (u=1) because the radius is growing.
V = sp.Symbol('V', real=True) 

# Extracting potential as T + V = E so rearranging we get V = E - T

V_eq = sp.Eq(V, sp.Rational(1,2) - simped_ode_sol.rhs)

# Integrate over one cell to get total phase energy
E_phase = sp.integrate(V_eq.rhs.subs({L: k}), (u, 0, 1))
E_phase_simplified = sp.simplify(E_phase)
print(f"Phase energy = {E_phase_simplified}")
print(f"Expected: k^2/4 = {k**2/4}")
print(f"Match: {sp.simplify(E_phase_simplified - k**2/4) == 0}")
print()

# --- Corollary 6: Cell-Independence of the Local Metric --- #
# Question: if cell n has base radius a = n+1 instead of 1,
# which quantities depend on a and which don't?

a = sp.Symbol('a', positive=True)

# Re-do the embedding with general base radius a
# (cell 0 has a=1, cell n has a=n+1)
theta_a = v + 2*sp.pi*k*u
X_a = (a + u) * sp.cos(theta_a)
Y_a = (a + u) * sp.sin(theta_a)
Z_a = u

Ru_a = sp.Matrix([sp.diff(X_a, u), sp.diff(Y_a, u), sp.diff(Z_a, u)])
Rv_a = sp.Matrix([sp.diff(X_a, v), sp.diff(Y_a, v), sp.diff(Z_a, v)])

E_a = sp.trigsimp(Ru_a.dot(Ru_a))
F_a = sp.trigsimp(Ru_a.dot(Rv_a))
G_a = sp.trigsimp(Rv_a.dot(Rv_a))

print("E(a) =", E_a)
print("F(a) =", F_a)
print("G(a) =", G_a)

# These depend on a. But check what the physics cares about:

# Gauge connection
A_u_a = sp.simplify(F_a / G_a)
print(f"\nA_u(a) = {A_u_a}")
print(f"Depends on a? {a in A_u_a.free_symbols}")

# g^uu — the thing that fixed c=1
det_g_a = sp.simplify(E_a * G_a - F_a**2)
g_uu_inv_a = sp.simplify(G_a / det_g_a)
print(f"\ng^uu(a) = {g_uu_inv_a}")
print(f"Depends on a? {a in g_uu_inv_a.free_symbols}")

# Ratio r(1)/r(0) — the cell expansion factor
ratio = (a + 1) / a
print(f"\nr(1)/r(0) = (a+1)/a = {ratio}")
print("This DOES depend on a — only equals 2 when a=1")

# But: gauge diag metric ds^2 = 2du^2 + (a+u)^2 dv'^2
# The wave equation on this gives Euler ODE with 
# indicial roots r = ±ℓ*sqrt(2), independent of a.
# The transfer matrix trace however uses w(1)/w(0) = (a+1)/a
# which only gives ln(2) when a=1.
print("\n--- Summary ---")
print("Cell-independent: A_u, g^uu, indicial roots")
print("Cell-dependent:   sqrt(g), r(1)/r(0), transfer matrix")
print("Periodicity requires: each cell resets to a=1")
print()

# Cor 7.

# --- Corollary 7: Fourier Decomposition --- #
# Axiom 6 requires psi(u, v + 2*pi) = psi(u, v).
# Test: e^{i*ell*v} satisfies this iff ell is integer.

ell = sp.Symbol('ell')

# Phase picked up by shifting v -> v + 2*pi
phase_shift = sp.exp(sp.I * ell * (v + 2*sp.pi)) / sp.exp(sp.I * ell * v)
phase_shift_simplified = sp.simplify(phase_shift)
print(f"Phase shift for v -> v+2pi: {phase_shift_simplified}")
print(f"This equals 1 iff ell is integer: e^(2*pi*i*ell) = 1")

# Check integer cases
for test_ell in [0, 1, 2, -1, -3]:
    val = sp.exp(2 * sp.pi * sp.I * test_ell)
    print(f"ell={test_ell}: e^(2*pi*i*{test_ell}) = {sp.simplify(val)}")

print()
print("Non integer check: ")
# Check non-integer case
print(f"ell=1/2: e^(2*pi*i*1/2) = {sp.simplify(sp.exp(2*sp.pi*sp.I*sp.Rational(1,2)))}")
print()

#corr 8.

# --- Corollary 8: Per-Mode Radial ODE --- #
#
# IMPORTANT: The Laplace-Beltrami operator is IMPORTED, not derived.
#
# What we have derived from Axioms 1-6:
#   - The embedding (Axiom 5)
#   - The induced metric g_ij (Corollary 1)  -> g_matrix
#   - The inverse metric g^ij (Corollary 1)  -> g_inv
#   - sqrt(g) = sqrt(2)*(1+u) (Corollary 1)  -> sqrt_g
#
# What we are borrowing from standard differential geometry:
#   - The Laplace-Beltrami operator: the unique second-order
#     differential operator on a Riemannian manifold that is
#     coordinate-invariant, self-adjoint, and reduces to the
#     ordinary Laplacian on flat space.
#   - The equation Box_M(psi) = 0, i.e. free propagation.
#
# Formula:
#   Box_M(psi) = (1/sqrt_g) * d_i( sqrt_g * g^ij * d_j(psi) )
#
# g^ij  corrects for non-orthogonal coordinate directions
# sqrt_g corrects for varying patch area across the surface

f = sp.Function('f')(u)
psi = f * sp.exp(sp.I * ell * v)

# Build Laplace-Beltrami term by term using g_inv and sqrt_g
# already computed in Corollaries 1 and 4
laplacian = 0
for i in range(2):
    for j in range(2):
        inner = sqrt_g * g_inv[i, j] * sp.diff(psi, coords[j])
        laplacian += sp.diff(inner, coords[i])
laplacian = laplacian / sqrt_g

# Factor out exp(i*ell*v) to get the ODE for f(u) alone
radial_ode = sp.simplify(laplacian / sp.exp(sp.I * ell * v))
radial_ode = sp.simplify(sp.collect(sp.expand(radial_ode), [f.diff(u,2), f.diff(u), f]))
print("Radial ODE (Box_M psi = 0, after dividing by e^{i*ell*v}):")
print(radial_ode, "= 0")

# Now do the SAME thing on the gauge-diagonalised cone
# ds^2 = 2 du^2 + (1+u)^2 dv'^2
# g_cone and g_cone_inv already defined

F_cone = sp.Function('F_c')(u)
psi_cone = F_cone * sp.exp(sp.I * ell * v_prime)
sqrt_g_cone = sp.sqrt(2) * (1 + u)  # same sqrt(det) on the cone

coords_cone_lb = [u, v_prime]

laplacian_cone = 0
for i in range(2):
    for j in range(2):
        inner = sqrt_g_cone * g_cone_inv[i, j] * sp.diff(psi_cone, coords_cone_lb[j])
        laplacian_cone += sp.diff(inner, coords_cone_lb[i])
laplacian_cone = laplacian_cone / sqrt_g_cone

# Factor out exp(i*ell*v')
radial_ode_cone = sp.simplify(laplacian_cone / sp.exp(sp.I * ell * v_prime))
radial_ode_cone = sp.collect(sp.expand(radial_ode_cone), [F_cone.diff(u,2), F_cone.diff(u), F_cone, u])
print()
print("Radial ODE on the cone:")
print(radial_ode_cone, "= 0")
print()
# Multiply through by 2*(1+u)^2 to get Euler's equation form
euler_form = sp.expand(radial_ode_cone * 2 * (1+u)**2)
euler_form = sp.collect(euler_form, [F_cone.diff(u,2), F_cone.diff(u), F_cone, u])
print("\nMultiplied by 2*(1+u)^2 (Euler form):")
print(euler_form, "= 0")
print()
# SymPy struggles with Euler ODEs directly.
# We know the solution: substitute w = 1+u, try F = w^r.
# Then: w^2 * r*(r-1)*w^{r-2} + w * r*w^{r-1} - 2*ell^2 * w^r = 0
# Divide by w^r: r^2 - r + r - 2*ell^2 = 0
# So: r^2 = 2*ell^2, giving r = ±ell*sqrt(2)

w = sp.Symbol('w', positive=True)
r = sp.Symbol('r')

# Indicial equation: plug F = w^r into w^2*F'' + w*F' - 2*ell^2*F = 0
indicial = r*(r-1) + r - 2*ell**2
print("Indicial equation:", indicial, "= 0")
print("Simplified:", sp.simplify(indicial), "= 0")

roots = sp.solve(indicial, r)
print("Roots:", roots)

# So the two solutions are:
# F_1(u) = (1+u)^{ell*sqrt(2)}
# F_2(u) = (1+u)^{-ell*sqrt(2)}

# Verify: plug each back into the ODE
for root in roots:
    F_test = (1 + u)**root
    lhs = (1+u)**2 * sp.diff(F_test, u, 2) + (1+u)*sp.diff(F_test, u) - 2*ell**2*F_test
    print(f"\nr = {root}: residual = {sp.simplify(lhs)}")

# --- Corollary 9: The Mass Shell --- #
# Each mode e^{i*ell*v} picks up phase 2*pi*ell per cell
# from its angular momentum. The gauge connection (Cor 2)
# contributes 2*pi*k. The relative phase mismatch per cell:

delta_phi = 2 * sp.pi * (ell - k)
print(f"Phase mismatch per cell: delta_phi = {delta_phi}")

# On-shell: ell = k -> delta_phi = 0 -> coherent propagation
print(f"On-shell (ell=k): delta_phi = {delta_phi.subs(ell, k)}")

# Off-shell: ell != k -> phase rotates by 2*pi*(ell-k) each cell
# Over N cells: total phase = N * 2*pi*(ell-k)
# For N >> 1, destructive interference kills these modes
N = sp.Symbol('N', positive=True, integer=True)
print(f"Off-shell accumulated phase over N cells: {sp.simplify(N * delta_phi)}")
print(f"Sum of e^(i*N*delta_phi) for large N -> 0 (destructive interference)")# --- Corollary 9: The Mass Shell --- #
print()
# --- Corollary 10: KG, Dirac, Schrödinger --- #
#
# Starting point: the Euler ODE (derived in Cor 8):
#   (1+u)^2 F'' + (1+u)F' - 2*ell^2 * F = 0
#
# STEP 1: KLEIN-GORDON
# The Euler ODE IS the KG equation on this manifold. Done.
print("=== Klein-Gordon ===")
print("(1+u)^2 F'' + (1+u)F' - 2*ell^2 F = 0")
print("This is Box_M(psi) = 0 after separation. Full wave equation.")
print()

# STEP 2: DIRAC via operator factorisation (algebraic, no ansatz)
# Define the Euler operator D = (1+u)*d/du
# Then (1+u)^2 F'' + (1+u)F' = D(D(F)) = D^2 F
# because D(F) = (1+u)F', D(D(F)) = (1+u)*d/du[(1+u)F']
#            = (1+u)[(1+u)F'' + F'] = (1+u)^2 F'' + (1+u)F'
#
# So the Euler ODE is: D^2 F - 2*ell^2 F = 0
# This factors: (D - ell*sqrt(2))(D + ell*sqrt(2)) F = 0

# Verify the factorisation computationally:
F_c = sp.Function('F_c')(u)

# The Euler operator D = (1+u)*d/du
def D(expr):
    return (1 + u) * sp.diff(expr, u)

# Full second-order operator: D^2 - 2*ell^2
full_op = D(D(F_c)) - 2*ell**2 * F_c

# Compare to the Euler ODE
euler_ode = (1+u)**2 * F_c.diff(u,2) + (1+u)*F_c.diff(u) - 2*ell**2*F_c
print("=== Operator factorisation ===")
print("D^2 F - 2*ell^2*F  vs  Euler ODE:")
print("Difference:", sp.simplify(full_op - euler_ode))
print()

# Now factor: (D - r)(D + r) where r = ell*sqrt(2)
r_val = ell * sp.sqrt(2)

# (D - r)(D + r)F = D(D(F) + r*F) - r*(D(F) + r*F)
inner = D(F_c) + r_val * F_c          # (D + r)F
outer = D(inner) - r_val * inner       # (D - r)(D + r)F

factored_diff = sp.simplify(outer - full_op)
print("(D - ell*sqrt(2))(D + ell*sqrt(2))F  vs  D^2 - 2*ell^2:")
print("Difference:", factored_diff)
print()

# Each factor is a FIRST-ORDER equation:
# (D + ell*sqrt(2))F = 0  =>  (1+u)F' + ell*sqrt(2)*F = 0
# (D - ell*sqrt(2))F = 0  =>  (1+u)F' - ell*sqrt(2)*F = 0
#
# These are the two Dirac-like equations. The cross-coupling
# g^{uv} = -pi*k from the original metric is what produces
# the off-diagonal structure when written as a 2-component system.

print("=== Dirac: first-order factors ===")
print("Factor 1: (1+u)F' + ell*sqrt(2)*F = 0")
print("Factor 2: (1+u)F' - ell*sqrt(2)*F = 0")
print()

# Verify each factor gives the known solutions
for sign, label in [(+1, "Factor 1 (r = -ell*sqrt(2))"), 
                     (-1, "Factor 2 (r = +ell*sqrt(2))")]:
    sol_test = (1 + u)**(sign * (-1) * ell * sp.sqrt(2))
    residual = (1+u)*sp.diff(sol_test, u) + sign*ell*sp.sqrt(2)*sol_test
    print(f"{label}: residual = {sp.simplify(residual)}")

print()

# STEP 3: SCHRÖDINGER
# Divide the Euler ODE by -2*(1+u)^2:
#   -1/2 F'' - F'/(2*(1+u)) + ell^2/(1+u)^2 * F = 0
#
# The Schrödinger form emerges by absorbing the F' term.
# Standard reduction to normal form (not an ansatz — change
# of dependent variable to eliminate first derivatives):
#   F = (1+u)^alpha * h(u), choose alpha to kill F' term.
#
# Coefficient of h' in transformed ODE: (2*alpha + 1)*(1+u)
# Set 2*alpha + 1 = 0  =>  alpha = -1/2

alpha = sp.Rational(-1, 2)
h = sp.Function('h')(u)
F_sub = (1 + u)**alpha * h

# Substitute into Euler ODE
euler_with_h = (1+u)**2 * sp.diff(F_sub, u, 2) + (1+u)*sp.diff(F_sub, u) - 2*ell**2*F_sub
euler_with_h = sp.expand(euler_with_h)

# Factor out (1+u)^alpha
euler_h = sp.simplify(euler_with_h / (1+u)**alpha)
euler_h = sp.collect(sp.expand(euler_h), [h.diff(u,2), h.diff(u), h])
print("=== Schrödinger: reduction to normal form ===")
print("Substitution F = (1+u)^{-1/2} * h(u)")
print("Transformed ODE (divided by (1+u)^{-1/2}):")
print(euler_h, "= 0")

# Should have NO h' term. Divide by -2(1+u)^2 to get:
# -1/2 h'' + V_eff(u) * h = 0
schrod_form = sp.simplify(euler_h / (-2*(1+u)**2))
schrod_form = sp.collect(sp.expand(schrod_form), [h.diff(u,2), h])
print()
print("Divided by -2*(1+u)^2:")
print(schrod_form, "= 0")
print()
print("WHERE DOES i COME FROM?")
print()
# The cone frame hides it. Go back to the original frame.
# The brutal ODE from Cor 8 has complex coefficients because
# g^{uv} = -pi*k, and d/dv acting on e^{i*ell*v} gives i*ell.

# Let's extract the i*f' term from the full original-frame ODE.
# Recall: psi = f(u) * exp(i*ell*v)
# The cross-term in the Laplacian is:
#   (1/sqrt_g) * d_u(sqrt_g * g^{uv} * d_v(psi)) 
#   + (1/sqrt_g) * d_v(sqrt_g * g^{uv} * d_u(psi))

g_uv_val = sp.simplify(g_inv[0, 1])
print(f"g^uv = {g_uv_val}")

# d_v(psi) = i*ell*f(u)*exp(i*ell*v)
# So the cross terms contribute i*ell*g^{uv} * f'(u) type terms.
# That's i * (-pi*k) * ell * f'(u) = -i*pi*k*ell*f'(u)

# On the mass shell (ell = k):
cross_coupling = g_uv_val * sp.I * k  # g^{uv} * i * ell with ell=k
print(f"Cross coupling on mass shell: {sp.simplify(cross_coupling)} * f'(u)")
print(f"This is the i*f' term — the Schrödinger first-order-in-u structure.")
print()

# Now: isolate this in the full original-frame ODE.
# Take the brutal ODE from Cor 8, set ell = k,
# and collect terms by derivative order.

radial_ode_onshell = radial_ode.subs(ell, k)
radial_ode_onshell = sp.collect(sp.expand(radial_ode_onshell), 
                                [f.diff(u,2), f.diff(u), f])
print("Full ODE on mass shell (ell = k):")
print(radial_ode_onshell, "= 0")
print()

# Extract coefficient of f'(u) — should contain i
coeff_f_prime = radial_ode_onshell.coeff(f.diff(u))
coeff_f_double_prime = radial_ode_onshell.coeff(f.diff(u, 2))
coeff_f = radial_ode_onshell.coeff(f)

print(f"Coefficient of f'':  {sp.simplify(coeff_f_double_prime)}")
print(f"Coefficient of f':   {sp.simplify(coeff_f_prime)}")
print(f"Coefficient of f:    {sp.simplify(coeff_f)}")
print()

# The f' coefficient has an imaginary part from g^{uv}.
# Rearrange: move the i*f' term to the left, everything else right.
# That's the Schrödinger form: i*(something)*f' = -1/2*f'' + V*f
print("The i in Schrödinger comes from g^{uv} = -pi*k")
print("coupling d/du to d/dv through the off-diagonal metric.")
print("The cone frame hides it; the original frame exposes it.")
print()

print("Clean up the coefficients from the mass-shell ODE: ")
# Coefficient of f'': already clean = 1/2
# Coefficient of f': factor out from the numerator

f_prime_coeff = coeff_f_prime
f_prime_num = sp.factor(sp.numer(sp.together(f_prime_coeff)))
f_prime_den = sp.factor(sp.denom(sp.together(f_prime_coeff)))
print("f' coefficient numerator:", f_prime_num)
print("f' coefficient denominator:", f_prime_den)

# Split into real and imaginary parts
f_prime_re = sp.re(f_prime_coeff).rewrite(sp.cos).simplify()
f_prime_im = sp.im(f_prime_coeff).rewrite(sp.cos).simplify()
print(f"f' coeff = {sp.simplify(f_prime_re)} + i*({sp.simplify(f_prime_im)})")

# The numerator: -4*I*pi*k^2*(u+1) + 1
# So: coeff of f' = [1 - 4*I*pi*k^2*(1+u)] / [2*(1+u)]
#                  = 1/(2*(1+u)) - 2*I*pi*k^2
# Real part: 1/(2*(1+u))     <- geometric damping
# Imag part: -2*pi*k^2       <- this is the i*f' Schrödinger term!

print()
print("Split form:")
print(f"  Real part: 1/(2*(1+u))  — geometric damping from expanding radius")
print(f"  Imag part: -2*pi*k^2    — the Schrödinger i*f' term")
print()

# Same treatment for f coefficient
f_coeff = coeff_f
f_coeff_together = sp.together(f_coeff)
f_coeff_num = sp.expand(sp.numer(f_coeff_together))
f_coeff_den = sp.factor(sp.denom(f_coeff_together))

# Factor (1+u) out of the numerator where possible
print("f coefficient numerator (expanded):", f_coeff_num)
print("f coefficient denominator:", f_coeff_den)

f_re = sp.simplify(sp.re(f_coeff))
f_im = sp.simplify(sp.im(f_coeff))
print(f"\nf coeff real part: {f_re}")
print(f"f coeff imag part: {f_im}")
print()

# Final clean Schrödinger form:
# 1/2 * f'' + [1/(2*(1+u)) - 2*i*pi*k^2] * f' + [V_real + i*V_imag]*f = 0
# Rearrange: multiply through by -2, isolate i terms on the left:
# i*(4*pi*k^2*f' + ...) = f'' + f'/(1+u) + ...
print("=== CLEAN FORM ===")
print("1/2 f'' + [1/(2(1+u)) - 2i*pi*k^2] f' + V(u)*f = 0")
print()
print("The imaginary coefficient -2*pi*k^2 on f' is CONSTANT.")
print("This is the Schrödinger i*df/du structure, with coupling strength 2*pi*k^2.")

t, x, h, m = sp.symbols('t x h m')
func = sp.Function('f')(x, t)
V_func = sp.Function('V')(x, t)

print("Let m = 1/(2(1 + u)) and let k = h")

eqn = sp.Eq((sp.I * h * sp.Derivative(func, t, 1)) * func, (-sp.Rational(1, 2) * (1/m) *  sp.Derivative(func, x, 2) * h ** 2 + V_func) * func, eval=False)
print("COMMON SCHRODINGER FORM: ")
print(eqn.lhs, "=", eqn.rhs)
