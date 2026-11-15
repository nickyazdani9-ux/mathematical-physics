import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from engineering.financial_dynamics.experiments import make_data, causal_features, run_suite, analyze


class ExperimentTests(unittest.TestCase):
    def test_generator_seed(self):
        assert_allclose(make_data('null',3,100),make_data('null',3,100))

    def test_future_mutation_cannot_change_past_features(self):
        r=make_data('regime',4,100)
        changed=r.copy(); changed[50:]+=10
        assert_allclose(causal_features(r)[:50],causal_features(changed)[:50])

    def test_all_run_artifacts_and_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            result=run_suite(Path(temp)/'run',seeds=(3,),n=200)
            self.assertEqual(len(result['results']),15)
            self.assertEqual(result['data_kind'],'synthetic')
            self.assertFalse(result['promotion_eligible'])
            proposal=analyze(Path(temp)/'run')
            self.assertFalse(proposal['auto_execute'])
            self.assertEqual(proposal['source_manifest_sha256'],result['manifest_sha256'])

    def test_artifact_tampering_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run'
            run_suite(path,seeds=(1,),n=200)
            with (path/'events.ndjson').open('a') as f:
                f.write('{}\n')
            with self.assertRaises(ValueError):
                analyze(path)

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run'
            run_suite(path,seeds=(1,),n=200)
            with self.assertRaises(FileExistsError):
                run_suite(path,seeds=(1,),n=200)


if __name__=='__main__':
    unittest.main()
