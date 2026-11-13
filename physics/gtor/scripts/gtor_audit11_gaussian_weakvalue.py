"""Offline reproducible checks for GTOR Research Audit 11 (13 Nov 2026).
Checks only source Gaussian spectrum and two-state weak value algebra.
Requires numpy and scipy. No web/GPU/cloud/services.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

def gaussian_spectrum(sigma, nu, omega):
    # arXiv:2611.13075 Eq. (6), r0=0, amplitude=1
    return np.sqrt(2*np.pi)*sigma*np.exp(-sigma**2*(omega**2+nu**2)/2)*np.cosh(sigma**2*omega*nu)

def direct_integral(sigma, nu, omega):
    def f(x):
        return np.exp(-x*x/(2*sigma*sigma))*np.cos(nu*x)*np.exp(1j*omega*x)
    cutoff=14*sigma+2*sigma*sigma*abs(omega.imag)
    re=quad(lambda x: f(x).real,-cutoff,cutoff,epsabs=2e-12)[0]
    im=quad(lambda x: f(x).imag,-cutoff,cutoff,epsabs=2e-12)[0]
    return re+1j*im

omega=0.3737-0.0890j # dimensionless Schwarzschild l=2 n=0, paper convention M=1
for sigma, nu in [(1,0.25),(3,0.2),(5,0.4)]:
    a=gaussian_spectrum(sigma,nu,omega)
    n=direct_integral(sigma,nu,omega)
    err=abs(a-n)/abs(a)
    assert err<1e-10
    print(f"sigma={sigma}, nu={nu}: relative Gaussian Fourier error {err:.3g}")

for sigma in [1,5]:
    res=minimize_scalar(lambda nu:-abs(gaussian_spectrum(sigma,nu,omega)),
                        bounds=(0,1),method="bounded",options={"xatol":1e-11})
    print(f"sigma={sigma}: overlap maximum nu={res.x:.8f} (Re omega={omega.real})")

# arXiv:2611.12971 Eq. (4): alpha_w=<f|alpha|i>/<f|i>
i=np.array([1,1],dtype=complex)/np.sqrt(2)
f=np.array([1,-0.8],dtype=complex)/np.sqrt(1+0.8**2)
alpha=np.array([0.99,1.01])
w=np.vdot(f,alpha*i)/np.vdot(f,i)
p=abs(np.vdot(f,i))**2
assert np.isclose(w,0.91) and w.real<min(alpha)
for a,b in [(0.2,0.7),(1.3,-0.4),(-0.11,2.9)]:
    iw=i*np.exp(1j*a)
    fw=f*np.exp(1j*b)
    assert np.isclose(np.vdot(fw,alpha*iw)/np.vdot(fw,iw),w)
print(f"alpha eigenvalues={alpha}; alpha_w={w.real:.6f}; control-only postselection={p:.8f}")
print("PASS (not a test of black-hole dynamics, environmental decoherence, or GTOR predictions)")
