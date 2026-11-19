#!/usr/bin/env python3
"""GTOR Audit 18: independent algebraic null controls, not full paper reproductions."""
import math
import cmath


def born_probability(psi, phi):
    dot = sum(a.conjugate()*b for a,b in zip(psi,phi))
    na = sum(abs(a)**2 for a in psi)
    nb = sum(abs(b)**2 for b in phi)
    return abs(dot)**2/(na*nb)


def test_projective():
    p=[1+0j, 0.75-0.25j]
    q=[0.25+0.1j, -0.4+0.3j]
    base=born_probability(p,q)
    changed=born_probability([(-3+2j)*z for z in p], [(0.2-7j)*z for z in q])
    assert abs(changed-base)<1e-14
    return base, abs(changed-base)


def test_loop_witness():
    # Eqs (13), (18) in arXiv:2611.23418. Hypothetical model, not a data fit.
    V=0.93
    Omega=math.pi/6
    kappa=0.02
    delta=kappa*Omega
    beta=3*math.pi/8
    E0=V*math.cos(2*(-math.pi/4-(beta+math.pi/8)))
    Eg=V*math.cos(2*(-math.pi/4-(beta+math.pi/8))-delta)
    direct=Eg-E0
    formula=2*V*math.sin(delta/2)*math.sin(2*beta-math.pi/4+delta/2)
    assert abs(direct-formula)<1e-13,(direct,formula)
    return delta,direct,formula


def test_clock():
    # Eq (31b), arXiv:2611.23784; classical dephased history with
    # conditional orthogonal qubit states and two mutually unbiased clock bases.
    def H2(x):
        return 0 if x in (0,1) else -x*math.log(x)-(1-x)*math.log(1-x)
    I_z=H2(0.5)-0.5*H2(0)-0.5*H2(0)
    I_x=H2(0.5)-0.5*H2(0.5)-0.5*H2(0.5)
    assert abs(I_z-math.log(2))<1e-15 and abs(I_x)<1e-15
    return I_z,I_x


def test_horizon_branches():
    # Eqs (12)-(15), arXiv:2611.23425; sample satisfies J>L>0, H>L.
    J,Q,P,H=5.,2.,2.,4.
    L=math.sqrt(J*J-(Q*P)**2)
    assert J>L>0 and H>L
    energies=[]
    for s in (1,-1):
        p=(J+s*L)/2
        C=2*L/(J+s*L)
        m=math.sqrt((H*H+L*L)/(2*C*H))
        energies.append((p,C,m*math.sqrt(C)))
    E_pred=math.sqrt((H*H+L*L)/(2*H))
    assert abs(energies[0][2]-energies[1][2])<1e-14
    assert abs(energies[0][2]-E_pred)<1e-14
    return energies,E_pred


def test_dbi_horizon():
    # Printed f(x) and horizon Einstein-tensor check, arXiv:2611.23597.
    # Not a reproduction of the whole DBI field equation calculation.
    xg=4/(3*math.pi)
    def f(x):
        return 1+3*xg*(x-(1+x*x)*math.atan(1/x))
    def derivative(x):
        return 6*xg*(1-x*math.atan(1/x))
    lo,hi=0.001,5.
    assert f(lo)<0<f(hi)
    for _ in range(90):
        mid=(lo+hi)/2
        if f(mid)>0: hi=mid
        else: lo=mid
    xh=(lo+hi)/2
    Gtt=(xh*derivative(xh)-1)/(1+xh*xh)
    assert abs(f(xh))<1e-12 and Gtt<0
    return xh,Gtt


if __name__=='__main__':
    print('Projective Born baseline and C* residual:',test_projective())
    print('Hypothetical loop phase, Delta E, Eq (18):',test_loop_witness())
    print('History-clock I_z and I_x (nats):',test_clock())
    print('KK twin branches [(p,C,E)] and common E:',test_horizon_branches())
    print('DBI-comment horizon x_h and a^2 Gt_t:',test_dbi_horizon())
    print('PASS: five algebraic null/control tests')
