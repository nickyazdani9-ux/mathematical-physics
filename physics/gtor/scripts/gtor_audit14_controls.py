"""GTOR Audit 14 (2026-11-16 AEDT): standard-theory null controls.

No GTOR action, quantum correction, or astrophysical measurement is assumed.
Run: python gtor_audit14_controls.py
Requires: sympy, numpy
"""
import numpy as np
import sympy as s

r, M, ell = s.symbols('r M ell', positive=True, finite=True)
fS = 1-2*M/r
fH = 1-2*M*r**2/(r**3+2*M*ell**2)

def curvature(f):
    # Areal radius, static spherical metric, ds²=-f dt²+dr²/f+r²dΩ²
    ricci = -s.diff(f,r,2) - 4*s.diff(f,r)/r + 2*(1-f)/r**2
    kretschmann = s.diff(f,r,2)**2 + 4*(s.diff(f,r)/r)**2 + 4*((1-f)/r**2)**2
    return s.factor(ricci), s.factor(kretschmann)

Rs,Ks = curvature(fS)
Rh,Kh = curvature(fH)
assert s.simplify(Rs)==0
assert s.simplify(Ks - 48*M**2/r**6)==0
assert s.simplify(s.limit(Rh,r,0) - 12/ell**2)==0
assert s.simplify(s.limit(Kh,r,0) - 24/ell**4)==0

# M=1, ell=1/2; nonextremal Hayward horizons solve r³-2r²+1/2=0.
roots = sorted([float(s.re(z)) for z in s.nroots(r**3 - 2*r**2 + s.Rational(1,2)) if abs(float(s.im(z))) < 1e-11 and float(s.re(z))>0])
assert len(roots)==2

# Schwarzschild photon sphere d(f/r²)/dr=0; critical b=r/sqrt(f).
photon = s.solve(s.diff(fS/r**2,r), r)
assert photon == [3*M]
impact = s.simplify((3*M)/s.sqrt(fS.subs(r,3*M)))
assert s.simplify(impact - 3*s.sqrt(3)*M)==0

# QFI of a qubit psi_theta=(cos(theta/2),sin(theta/2)) and its
# parameter-dependent global rephasing exp(i alpha theta).
theta, alpha = s.symbols('theta alpha', real=True)
psi = s.Matrix([s.cos(theta/2), s.sin(theta/2)])

def inner(u,v):
    return (s.conjugate(u).T * v)[0]

def qfi(state):
    st = state / s.sqrt(s.simplify(inner(state,state)))
    d = st.diff(theta)
    return s.trigsimp(4*(inner(d,d)-inner(d,st)*inner(st,d)), method="fu")

F_base=qfi(psi)
F_gauge=qfi(s.exp(s.I*alpha*theta)*psi)
assert s.simplify(F_base-1)==0
assert s.simplify(F_gauge-1)==0
naive = s.trigsimp(4*inner((s.exp(s.I*alpha*theta)*psi).diff(theta),(s.exp(s.I*alpha*theta)*psi).diff(theta)), method="fu")
assert s.simplify(naive-(1+4*alpha**2))==0

# Normalized projector is unaffected by nonzero complex overall rescaling.
z=2+3j
v=np.array([0.6,0.8j],dtype=complex)
P=lambda w:np.outer(w,w.conj())/np.vdot(w,w)
assert np.linalg.norm(P(v)-P(z*v))<1e-14

print('Schwarzschild: R=0, K=48 M^2/r^6 [symbolic PASS]')
print('Hayward center: R(0)=12/ell^2, K(0)=24/ell^4 [symbolic PASS]')
print('Hayward M=1 ell=0.5 horizons:',*[f'{x:.12f}' for x in roots])
print('Schwarzschild photon sphere r=3M, b=3sqrt(3)M [symbolic PASS]')
print('Projective QFI F_theta=1, independent of alpha [symbolic PASS]')
print('Naive 4||dpsi/dtheta||^2 for alpha=7:', naive.subs(alpha,7))
print('Projector scaling residual:', np.linalg.norm(P(v)-P(z*v)))
print('Schwarzschild large-n Re(M omega)=ln(3)/(8pi) =',float(s.log(3)/(8*s.pi)))
print('NOTE: This script checks baseline algebra only; it does not compute QNMs, Kerr shadows, or Vlasov evolution.')
