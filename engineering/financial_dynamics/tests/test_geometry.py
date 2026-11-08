import unittest
import numpy as np
from numpy.testing import assert_allclose
from engineering.financial_dynamics.geometry import PhaseState, projective_step, sphere_readout, selective_step


class GeometryTests(unittest.TestCase):
    def test_multi_turn_and_reverse(self):
        s = PhaseState(0.2, 3).advance(8*np.pi + 0.3)
        self.assertEqual(s.turns, 7)
        self.assertAlmostEqual(s.phase, 0.5)
        s = s.advance(-10*np.pi-1.0)
        self.assertEqual(s.turns, 1)
        self.assertAlmostEqual(s.phase, 2*np.pi-0.5)

    def test_rotation_norm(self):
        s = PhaseState(0.0, 0)
        for _ in range(1000):
            s = s.advance(0.125)
        self.assertAlmostEqual(abs(s.unit()), 1.0)
        self.assertAlmostEqual(s.phase + 2*np.pi*s.turns, 125.0)

    def test_phase_rejects_invalid(self):
        for bad in [np.nan, np.inf, -0.1, 2*np.pi]:
            with self.assertRaises(ValueError):
                PhaseState(bad, 0)
        with self.assertRaises(ValueError):
            PhaseState(0., 0).advance(np.inf)

    def test_riccati_tangent(self):
        h = projective_step([1., 0.], a=1., b=0., c=1., dt=0.3)
        self.assertAlmostEqual((h[1]/h[0]).real, np.tan(0.3), places=12)

    def test_pole_has_regular_homogeneous_state(self):
        h = projective_step([1., 0.], a=1., b=0., c=1., dt=np.pi/2)
        self.assertTrue(np.isfinite(h).all())
        self.assertLess(abs(h[0]), 1e-12)
        self.assertAlmostEqual(np.linalg.norm(sphere_readout(h)), 1.0)

    def test_complex_scale_invariance(self):
        h = np.array([1+2j, 2-1j])
        assert_allclose(sphere_readout(h), sphere_readout((-3+7j)*h), atol=1e-14)

    def test_frozen_flow_semigroup(self):
        args = dict(a=.2+.1j, b=-.4, c=.7j)
        a = projective_step(projective_step([1, .2j],dt=.3,**args),dt=.4,**args)
        b = projective_step([1, .2j],dt=.7,**args)
        assert_allclose(sphere_readout(a), sphere_readout(b), atol=1e-13)

    def test_zero_and_nonfinite_state_rejected(self):
        for h in [[0,0], [1,np.nan], [1,2,3]]:
            with self.assertRaises(ValueError):
                projective_step(h,0,0,0,.1)
        with self.assertRaises(ValueError):
            projective_step([1,0],0,0,0,-1)

    def test_selectivity_responds_to_input(self):
        a = selective_step(PhaseState(0.,0),[1,0],0.,.1)
        b = selective_step(PhaseState(0.,0),[1,0],1.,.1)
        self.assertNotAlmostEqual(a[0].phase, b[0].phase)
        self.assertGreater(np.linalg.norm(a[2]-b[2]), 1e-5)


if __name__ == '__main__':
    unittest.main()
