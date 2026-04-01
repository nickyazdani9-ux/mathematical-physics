import sympy as sp
from sympy import Matrix, simplify, sqrt

# === HELICAL MANIFOLD: CONICAL GAUGE COORDINATES (START OF SCRIPT) ===
# We use the gauge-diagonalized frame (v' = v + 2πk u) from Corollary 5.
# This gives the clean cone metric ds² = 2 du² + (1+u)² dv'².
# All downstream steps will pull directly from this definition.

u, k = sp.symbols('u k', real=True, positive=True)
# k is the integer winding number (kept symbolic here)

# Metric components in (u, v') coordinates
g_uu_metric = sp.sympify(2)          # from Axiom 1 + gauge diagonalization
g_vv_metric = (1 + u)**2
g_uv_metric = sp.sympify(0)

# Metric tensor (fed forward)
g_cone = Matrix([[g_uu_metric, g_uv_metric],
                 [g_uv_metric, g_vv_metric]])

# Determinant (computed, not typed)
det_g = simplify(g_cone.det())

# Square root of determinant (for later Laplace-Beltrami)
sqrt_g = sqrt(det_g)

# Inverse metric (computed directly from g_cone)
g_inv = g_cone.inv()
guu = simplify(g_inv[0,0])
gvv = simplify(g_inv[1,1])
guv = simplify(g_inv[0,1])

# Print symbolic results for verification only
print("Conical metric g_cone:")
sp.pprint(g_cone)
print("\ndet(g) =")
sp.pprint(det_g)
print("\nsqrt(g) =")
sp.pprint(sqrt_g)
print("\nInverse metric components:")
print("guu =", guu)
print("gvv =", gvv)
print("guv =", guv)
