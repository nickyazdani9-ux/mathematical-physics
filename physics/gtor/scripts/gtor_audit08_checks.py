"""GTOR Research Audit 08 (2026-11-10): offline sanity checks.

No network, external dependencies, cloud provisioning, or resource spending.
This checks standard projective Born invariance and a simple covariance model;
it does NOT simulate a 4D causal set or validate GTOR physics.
"""
import cmath
import math
import random

R = random.Random(20261110)


def born_overlap(a, b):
    dot = sum(x.conjugate() * y for x, y in zip(a, b))
    return abs(dot) ** 2 / (
        sum(abs(x) ** 2 for x in a) * sum(abs(y) ** 2 for y in b)
    )


maxerr = 0.0
maxscale = 0.0
for _ in range(2500):
    a = [complex(R.gauss(0, 1), R.gauss(0, 1)) for _ in range(3)]
    b = [complex(R.gauss(0, 1), R.gauss(0, 1)) for _ in range(3)]
    p = born_overlap(a, b)
    d_fs = math.acos(min(1, max(0, math.sqrt(p))))
    maxerr = max(maxerr, abs(p - math.cos(d_fs) ** 2))

    sc_a = R.uniform(0.1, 5.0) * cmath.exp(1j * R.uniform(-math.pi, math.pi))
    sc_b = R.uniform(0.1, 5.0) * cmath.exp(1j * R.uniform(-math.pi, math.pi))
    maxscale = max(
        maxscale,
        abs(p - born_overlap([sc_a * x for x in a], [sc_b * y for y in b])),
    )

# Two independent groups of m and n points in one dimension with independent
# displacement epsilon ~ Uniform(-delta, delta). With s > 2*delta,
# each pairwise separation is s + epsilon_b - epsilon_a. All m*n distances
# are correlated due to repeated use of the same points.
m = n = 8
s = 10.0
delta = 0.3
N = 25000
means = []
for _ in range(N):
    ea = [R.uniform(-delta, delta) for _ in range(m)]
    eb = [R.uniform(-delta, delta) for _ in range(n)]
    means.append(s + sum(eb) / n - sum(ea) / m)
mu = sum(means) / N
sigma = math.sqrt(sum((t - mu) ** 2 for t in means) / (N - 1))
sigma_theory = math.sqrt((1 / m + 1 / n) * delta**2 / 3)
sigma_wrong = math.sqrt(2 * delta**2 / (3 * m * n))

print("max |Born-cos²(FS)| residual", maxerr)
print("max independently rescaled Born residual", maxscale)
print("correlated mean-distance observed std", sigma)
print("covariance-aware prediction", sigma_theory)
print("wrong independent-pairs prediction", sigma_wrong)
print("theoretical std inflation factor", sigma_theory / sigma_wrong)

assert maxerr < 1e-12
assert maxscale < 1e-12
assert abs(sigma / sigma_theory - 1) < 0.04
