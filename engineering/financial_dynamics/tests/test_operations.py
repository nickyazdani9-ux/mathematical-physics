import tempfile
import unittest
from pathlib import Path
from engineering.financial_dynamics.lifecycle import ExperimentLease, LeaseStore, run_leased
from engineering.financial_dynamics.invoices import prepare_invoice


class FakeProvider:
    def __init__(self, fail_run=False, fail_destroy=False, fail_create=False):
        self.fail_run,self.fail_destroy,self.fail_create=fail_run,fail_destroy,fail_create
        self.created=self.destroyed=0

    def create(self, run_id, deadline):
        self.created+=1
        if self.fail_create:
            raise TimeoutError('unknown create outcome')
        return 'fake-instance-1'

    def run(self, instance_id, timeout_seconds):
        if self.fail_run:
            raise TimeoutError('worker timed out')
        return {'artifact_digest':'fixture-digest'}

    def destroy(self, instance_id):
        self.destroyed+=1
        if self.fail_destroy:
            raise OSError('destroy failed')

    def is_destroyed(self, instance_id):
        return not self.fail_destroy


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=LeaseStore(Path(self.tmp.name)/'leases.sqlite')
        self.lease=ExperimentLease('run-1',300,20,10)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_zero_budget_denies_before_create(self):
        p=FakeProvider()
        with self.assertRaises(PermissionError):
            run_leased(self.db,p,self.lease,allowance_cents=0)
        self.assertEqual(p.created,0)

    def test_success_also_destroys(self):
        p=FakeProvider()
        out=run_leased(self.db,p,self.lease,allowance_cents=20)
        self.assertEqual(p.destroyed,1)
        self.assertEqual(out['state'],'DESTROYED')

    def test_timeout_destroys(self):
        p=FakeProvider(fail_run=True)
        out=run_leased(self.db,p,self.lease,allowance_cents=20)
        self.assertEqual(p.destroyed,1)
        self.assertEqual(out['outcome'],'failed')
        self.assertEqual(out['state'],'DESTROYED')

    def test_destroy_failure_retains_reservation(self):
        out=run_leased(self.db,FakeProvider(fail_destroy=True),self.lease,allowance_cents=20)
        self.assertEqual(out['state'],'CLEANUP_REQUIRED')
        self.assertEqual(self.db.reserved(),20)

    def test_unknown_create_outcome_is_quarantined(self):
        p=FakeProvider(fail_create=True)
        out=run_leased(self.db,p,self.lease,allowance_cents=20)
        self.assertEqual(out['state'],'ALLOCATION_UNKNOWN')
        self.assertEqual(self.db.reserved(),20)
        with self.assertRaises(ValueError):
            run_leased(self.db,p,self.lease,allowance_cents=100)
        self.assertEqual(p.created,1)

    def test_budget_reservations_survive_restart(self):
        run_leased(self.db,FakeProvider(fail_destroy=True),self.lease,allowance_cents=20)
        self.db.close()
        self.db=LeaseStore(Path(self.tmp.name)/'leases.sqlite')
        with self.assertRaises(PermissionError):
            run_leased(self.db,FakeProvider(),ExperimentLease('run-2',300,20,10),allowance_cents=20)

    def test_duplicate_id_never_reruns(self):
        p=FakeProvider()
        run_leased(self.db,p,self.lease,allowance_cents=20)
        with self.assertRaises(ValueError):
            run_leased(self.db,p,self.lease,allowance_cents=40)
        self.assertEqual(p.created,1)

    def test_bad_caps_rejected(self):
        for duration,cap,quote in [(0,20,10),(301,20,10),(300,0,0),(300,10,20)]:
            with self.assertRaises(ValueError):
                ExperimentLease('x',duration,cap,quote)

    def test_nonregistered_invoice(self):
        out=prepare_invoice('EXAMPLE-1','Example seller',None,False,[{'description':'Research service','quantity':2,'unit_price':'10.00','taxable':False}])
        self.assertEqual(out['title'],'Invoice')
        self.assertEqual(out['gst'],'0.00')
        self.assertEqual(out['total'],'20.00')

    def test_unknown_gst_status_blocks(self):
        with self.assertRaises(ValueError):
            prepare_invoice('1','Example',None,None,[])

    def test_taxable_line_without_registration_blocks(self):
        with self.assertRaises(ValueError):
            prepare_invoice('1','Example',None,False,[{'description':'Service','quantity':1,'unit_price':'10','taxable':True}])

    def test_registered_needs_valid_abn(self):
        with self.assertRaises(ValueError):
            prepare_invoice('1','Example','00000000000',True,[{'description':'Service','quantity':1,'unit_price':'10','taxable':True}])

    def test_tax_total_and_buyer_threshold(self):
        # Published ABR checksum example; never represents the user's ABN.
        line=[{'description':'Fixture only','quantity':1,'unit_price':'1000','taxable':True}]
        with self.assertRaises(ValueError):
            prepare_invoice('1','Fixture','51824753556',True,line)
        out=prepare_invoice('1','Fixture','51824753556',True,line,buyer='Fixture buyer')
        self.assertEqual(out['gst'],'100.00')
        self.assertEqual(out['total'],'1100.00')


if __name__=='__main__':
    unittest.main()
