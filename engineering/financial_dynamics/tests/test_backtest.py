import unittest
import numpy as np
from numpy.testing import assert_allclose
from engineering.financial_dynamics.backtest import simulate, metrics, block_bootstrap_mean
from engineering.financial_dynamics.data import Vintage, asof_value, purged_walk_forward


class BacktestTests(unittest.TestCase):
    def test_cash_never_trades(self):
        r = simulate([100,200,50], [0,0], fee_bps=10, slippage_bps=10)
        assert_allclose(r['equity'], [1,1,1])

    def test_buy_hold_self_financing(self):
        r = simulate([100,110,121], [1,1], fee_bps=0, slippage_bps=0)
        assert_allclose(r['equity'], [1,1.1,1.21])
        self.assertAlmostEqual(r['turnover'], 2.)

    def test_entry_and_liquidation_costs(self):
        r = simulate([100,100], [1], fee_bps=100, slippage_bps=0)
        self.assertAlmostEqual(r['equity'][-1], .99/1.01)
        self.assertEqual(r['trades'], 2)

    def test_slippage_costs(self):
        r = simulate([100,100], [1], fee_bps=0, slippage_bps=100)
        self.assertAlmostEqual(r['equity'][-1], .99/1.01)

    def test_future_signal_does_not_change_earlier_equity(self):
        a = simulate([100,110,120,80],[.5,.5,0],fee_bps=2)
        b = simulate([100,110,120,80],[.5,.5,1],fee_bps=2)
        assert_allclose(a['equity'][:3], b['equity'][:3])

    def test_invalid_prices_weights_and_costs(self):
        for p,w in [([100,0],[1]), ([100,np.nan],[1]), ([100,101],[2]), ([100,101],[np.nan]),([100,101],[])]:
            with self.assertRaises(ValueError):
                simulate(p,w)
        with self.assertRaises(ValueError):
            simulate([1,2],[1],fee_bps=-1)

    def test_drawdown_includes_starting_cash(self):
        m=metrics([1,.8,.9],periods_per_year=252)
        self.assertAlmostEqual(m['max_drawdown'],.2)
        self.assertAlmostEqual(m['total_return'],-.1)

    def test_zero_volatility_sharpe_is_undefined(self):
        self.assertIsNone(metrics([1,1,1])['sharpe_naive'])

    def test_bootstrap_is_reproducible(self):
        a=block_bootstrap_mean(np.arange(20)/1000,block_size=4,seed=3)
        b=block_bootstrap_mean(np.arange(20)/1000,block_size=4,seed=3)
        self.assertEqual(a,b)
        self.assertLessEqual(a[0],a[1])

    def test_vintage_uses_availability_not_observation_time(self):
        rows=[Vintage(0,10,1.),Vintage(0,20,9.)]
        self.assertIsNone(asof_value(rows,0,9))
        self.assertEqual(asof_value(rows,0,10),1.)
        self.assertEqual(asof_value(rows,0,19),1.)
        self.assertEqual(asof_value(rows,0,20),9.)

    def test_conflicting_vintage_rejected(self):
        with self.assertRaises(ValueError):
            asof_value([Vintage(0,10,1),Vintage(0,10,2)],0,20)

    def test_walk_forward_label_purging(self):
        ts=np.arange(30)
        folds=list(purged_walk_forward(ts,ts+3,min_train=10,test_size=5,gap=2))
        self.assertGreater(len(folds),0)
        for train,test in folds:
            self.assertTrue(np.all((ts+3)[train] < ts[test[0]]-2))
            self.assertLess(train[-1],test[0])
        joined=np.concatenate([te for _,te in folds])
        self.assertEqual(len(joined),len(set(joined)))

    def test_unsorted_time_rejected(self):
        with self.assertRaises(ValueError):
            list(purged_walk_forward([0,2,1],[1,3,2],min_train=1,test_size=1))


if __name__ == '__main__':
    unittest.main()
