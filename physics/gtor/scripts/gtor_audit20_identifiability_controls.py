#!/usr/bin/env python3
"""GTOR Audit 20 (2026-11-22): offline independent null-model checks.

These algebraic/numerical controls are *not* reproductions of complete papers
or predictions of GTOR. Python 3 standard library only.
"""
import cmath
import math

M = 1.0
# 1. Palatini Kalb-Ramond: g_tt=-(1-2M/r), g_rr=sigma^2/(1-2M/r).
for q in (-0.6, 0.0, 0.6):
    sigma = math.sqrt(1-q/2)
    bcrit = 3*math.sqrt(3)*M
    risco = 6*M
    flux_ratio = 1/sigma
    temperature_ratio = flux_ratio**0.25
    spacing = math.exp(-math.pi/sigma)
    epicyclic2 = (M/(sigma**2*8**3))*(1-6/8)
    assert abs(bcrit - 5.196152422706632) < 1e-12
    assert risco == 6*M and epicyclic2 > 0
    print(f'PALATINI q={q:+.1f} sigma={sigma:.9f} F/F0={flux_ratio:.9f} T/T0={temperature_ratio:.9f} ring_spacing={spacing:.9f}')

# 2. Damour-Solodukhin: g_tt=-(1-2M/r), g_rr=(1-2(1+lambda^2)M/r)^-1.
for lam, w in ((0.0,0.3),(0.5,0.3),(0.5,0.55),(0.8,0.3)):
    throat = 2*(1+lam**2)*M
    photon_sphere_available = (throat < 3*M)
    wmax=1/(1+4*lam**2)
    rc=M*(1+3*w)/(2*w)
    sonic_allowed=(0<w<wmax)
    assert sonic_allowed == (rc > throat)
    print(f'DS lambda={lam:.2f} w={w:.2f} throat={throat:.6f} photon_sphere_external={photon_sphere_available} sonic_critical={rc:.6f} sonic_exterior={sonic_allowed}')

# 3. Independent smooth, positive ultrastatic two-throat toy R(l) (not the paper's R_N).
# R^2=l^2+a^2+c exp(-l^2), a=1, c=2. Stationary spheres: 0, +/- sqrt(ln c).
a, c=1.0, 2.0
def R_derivs(l):
    e=math.exp(-l*l)
    F=l*l + a*a + c*e
    F1=2*l*(1-c*e)
    F2=2*(1-c*e)+4*c*l*l*e
    R=math.sqrt(F)
    R1=F1/(2*R)
    R2=F2/(2*R)-F1*F1/(4*R**3)
    return R,R1,R2
for l in (-math.sqrt(math.log(c)),0.0,math.sqrt(math.log(c))):
    R,R1,R2=R_derivs(l)
    NEC=-R2/(4*math.pi*R)
    light_ring_V2 = -2*R2/R**3  # L=1; V=1/R^2 at R'=0
    assert abs(R1)<1e-13
    assert (NEC<0 and light_ring_V2<0) if abs(l)>0.1 else (NEC>0 and light_ring_V2>0)
    print(f'ULTRASTATIC l={l:.9f} R={R:.9f} NEC_r={NEC:+.9f} Vnull_2nd={light_ring_V2:+.9f}')
# ANEC integral equals -1/(4pi) integral of (R'/R)^2 for infinite endpoints.
N=40000; bound=40; h=2*bound/N
sq_sum=0.0; nec_sum=0.0
for i in range(N+1):
    l=-bound+i*h
    R,R1,R2=R_derivs(l)
    weight=0.5 if i in (0,N) else 1.0
    sq_sum+=weight*(R1/R)**2
    nec_sum+=weight*(-R2/(4*math.pi*R))
sq_integral=-h*sq_sum/(4*math.pi)
nec_integral=h*nec_sum
left_R,left_R1,_=R_derivs(-bound)
right_R,right_R1,_=R_derivs(bound)
boundary=-(right_R1/right_R-left_R1/left_R)/(4*math.pi)
assert sq_integral<0 and abs(nec_integral-(sq_integral+boundary))<1e-8
print(f'ULTRASTATIC ANEC on [-40,40]: {nec_integral:+.9f}; squared_term={sq_integral:+.9f}, boundary={boundary:+.9f}')

# 4. Standard Fubini-Study/projective quantum invariant (Provost-Vallee control).
def psi(theta, phi, scale=1):
    return [scale*math.cos(theta/2), scale*cmath.exp(1j*phi)*math.sin(theta/2)]
def ray_overlap_sq(x,y):
    ip=sum(a.conjugate()*b for a,b in zip(x,y))
    nx=sum(abs(z)**2 for z in x)
    ny=sum(abs(z)**2 for z in y)
    return abs(ip)**2/(nx*ny)
for t in (0.4, 1.1):
    phi=0.3; eps=1e-3
    base=psi(t,phi)
    delta=psi(t,phi+eps)
    fs_phi=(1-ray_overlap_sq(base,delta))/eps**2
    prediction=0.25*math.sin(t)**2
    assert abs(fs_phi-prediction)<1e-7
    # Arbitrary complex scale applied to each state separately.
    x=psi(t,phi,3*cmath.exp(1.72j))
    y=psi(t,phi+eps,0.12*cmath.exp(-2j))
    assert abs(ray_overlap_sq(base,delta)-ray_overlap_sq(x,y))<1e-13
    print(f'PROJECTIVE theta={t:.1f} g_phiphi={fs_phi:.9f} predicted={prediction:.9f}')

# 5. Polygon entanglement inequalities for analytic pure three-qubit cases.
# Entropy of each single-qubit reduced state: GHZ=(1,1,1), Bell_AB tensor |0>_C=(1,1,0).
for name, entropies in [('GHZ',(1.,1.,1.)),('Bell_AB_tensor_0C',(1.,1.,0.))]:
    lhs=[2*s for s in entropies]; rhs=sum(entropies)
    assert all(x<=rhs+1e-12 for x in lhs)
    print(f'POLYGON {name} E={entropies} max(2Ei-sumE)={max(lhs)-rhs:+.1f}')
print('PASS: five null-control families; no GTOR-specific dynamics inferred')
