#!/usr/bin/env python3
"""GTOR Audit 05: dependency-free toy quantum-projection and Kerr root checks.

This tests algebraic consequences of explicitly stated finite-dimensional
and stationary ansatz assumptions. It does NOT confirm any quantum-gravity
model, time-dependent collapse, stability, or GTOR's physical predictions.
"""
from __future__ import annotations

import cmath
import math


def fs_distance(a: tuple[complex, ...], b: tuple[complex, ...]) -> float:
    """Fubini–Study angle d=acos(|<a,b>|/(||a|| ||b||)) for nonzero vectors."""
    inner = sum(x.conjugate() * y for x, y in zip(a, b))
    norm2a = sum(abs(x) ** 2 for x in a)
    norm2b = sum(abs(y) ** 2 for y in b)
    if min(norm2a, norm2b) <= 0:
        raise ValueError("Ray representative cannot be zero")
    return math.acos(min(1.0, abs(inner) / math.sqrt(norm2a * norm2b)))


def normalized_projection(a: tuple[complex, complex, complex]) -> tuple[complex, ...]:
    """Project onto the physical kernel span(e0,e1) and normalize."""
    projected = (a[0], a[1], 0j)
    norm = math.sqrt(sum(abs(x) ** 2 for x in projected))
    if norm == 0:
        raise ValueError("Seed has zero physical projection")
    return tuple(x / norm for x in projected)


def kerr_stationary_profile(n: int, mass: float, spin: float):
    """Ovalle N=1 stationary m_n(r) model with Kerr exterior."""
    if n <= 2 or mass <= 0 or abs(spin) >= mass:
        raise ValueError("Requires n>2 and nonextremal |a|<M")
    h = mass + math.sqrt(mass * mass - spin * spin)

    def m(r: float) -> float:
        x = r / h
        return mass * ((n + 1) * x**3 - 3 * x**(n + 1)) / (n - 2)

    def dm_dr(r: float) -> float:
        x = r / h
        return mass * 3 * (n + 1) * (x**2 - x**n) / ((n - 2) * h)

    def delta(r: float) -> float:
        return r * r - 2 * r * m(r) + spin * spin

    assert abs(m(h) - mass) < 1e-12
    assert abs(dm_dr(h)) < 1e-12
    assert abs(delta(h)) < 1e-12

    # Detect a positive -> negative crossing strictly before outer horizon.
    a, b = None, None
    prev = 0.0
    for i in range(1, 2000):
        r = h * i / 2000
        if delta(prev) > 0 and delta(r) < 0:
            a, b = prev, r
            break
        prev = r
    if a is None or b is None:
        raise RuntimeError("Failed to bracket inner root")
    for _ in range(100):
        mid = 0.5 * (a + b)
        if delta(mid) > 0:
            a = mid
        else:
            b = mid
    return h, 0.5 * (a + b)


def main() -> None:
    s = (1 + 0j, 0j, 0j)
    t = (1 / math.sqrt(2) + 0j, 0j, 1 / math.sqrt(2) + 0j)
    base = fs_distance(s, t)
    physical = fs_distance(normalized_projection(s), normalized_projection(t))
    phase_scale = 3 * cmath.exp(1.1j)
    transformed = tuple(phase_scale * x for x in t)
    assert abs(base - math.pi / 4) < 1e-12
    assert abs(physical) < 1e-12
    assert abs(fs_distance(s, transformed) - base) < 1e-12
    print(f"FS(seeds)={base:.12f}; FS(projected)={physical:.12f}")
    for n in (3, 4, 8, 20):
        h, hc = kerr_stationary_profile(n, mass=1.0, spin=0.7)
        print(f"n={n:2d}: h={h:.10f}, inner_horizon={hc:.10f}, gap={h-hc:.10f}")
    print("PASS: limited toy consistency tests; no physical GTOR validation")


if __name__ == "__main__":
    main()
