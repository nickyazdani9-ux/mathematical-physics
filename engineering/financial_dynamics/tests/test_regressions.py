import tempfile
import unittest
from pathlib import Path
import numpy as np
from engineering.financial_dynamics.backtest import simulate
from engineering.financial_dynamics.geometry import projective_step,sphere_readout
from engineering.financial_dynamics.invoices import prepare_invoice,render_draft
from engineering.financial_dynamics.lifecycle import ExperimentLease,LeaseStore,run_leased
from scipy.integrate import solve_ivp


class RegressionTests(unittest.TestCase):
    def test_cost_increase_reduces_static_roundtrip(self):
        low=simulate([100,110],[1],fee_bps=1,slippage_bps=1)['equity'][-1]
        high=simulate([100,110],[1],fee_bps=100,slippage_bps=100)['equity'][-1]
        self.assertLess(high,low)

    def test_riccati_against_independent_numerical_solver(self):
        a,b,c=.2,-.3,.1
        ref=solve_ivp(lambda t,z:a+b*z+c*z*z,[0,1],[.3],rtol=1e-11,atol=1e-13).y[0,-1]
        h=projective_step([1,.3],a,b,c,1.)
        self.assertAlmostEqual((h[1]/h[0]).real,ref,places=10)

    def test_nonconstant_equilibrium_has_nonzero_residual(self):
        # Airy y''-t*y=0, instantaneous root r=sqrt(t): residual/y=r'.
        t=4.
        r=np.sqrt(t)
        residual=1/(2*np.sqrt(t))+r*r-t
        self.assertAlmostEqual(residual,.25)
        self.assertNotEqual(residual,0.)

    def test_variable_generator_converges_under_refinement(self):
        ref=solve_ivp(lambda t,z:np.sin(t)-.2*z+.1*z*z,[0,1],[.2],rtol=1e-12,atol=1e-13).y[0,-1]
        errors=[]
        for count in [10,20,40]:
            h=[1,.2]
            for k in range(count):
                h=projective_step(h,np.sin(k/count),-.2,.1,1/count)
            errors.append(abs((h[1]/h[0]).real-ref))
        self.assertLess(errors[1],errors[0]*.6)
        self.assertLess(errors[2],errors[1]*.6)

    def test_invoice_html_escapes_text_and_has_buyer(self):
        draft=prepare_invoice('x','<script>bad</script>','51824753556',True,[{'description':'<img>','quantity':1,'unit_price':'1000','taxable':True}],buyer='Fixture buyer',issue_date='2026-10-27')
        rendered=render_draft(draft)
        self.assertNotIn('<script>',rendered)
        self.assertIn('Fixture buyer',rendered)
        self.assertIn('2026-10-27',rendered)
        self.assertIn('DRAFT',rendered)

    def test_invoice_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            prepare_invoice('x','Example',None,False,[{'description':'Service','quantity':1,'unit_price':'NaN','taxable':False}])

    def test_invalid_provider_identifier_is_quarantined(self):
        class InvalidProvider:
            def create(self,*args): return {'unexpected':'shape'}
            def destroy(self,*args): raise AssertionError('must not destroy guessed resource')
        with tempfile.TemporaryDirectory() as temp:
            store=LeaseStore(Path(temp)/'lease.db')
            try:
                out=run_leased(store,InvalidProvider(),ExperimentLease('x',100,10,5),10)
                self.assertEqual(out['state'],'ALLOCATION_UNKNOWN')
            finally:
                store.close()


if __name__=='__main__':
    unittest.main()
