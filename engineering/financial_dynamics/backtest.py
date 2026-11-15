"""Single-asset, long-only, fractional-share bar simulator.

Targets must be decided BEFORE the corresponding execution open. Final
liquidation is charged. No market-impact, borrow, partial fills or broker.
"""
import math
import numpy as np


def simulate(opens, targets, fee_bps=5., slippage_bps=5., liquidate=True):
    p,w=np.asarray(opens,float),np.asarray(targets,float)
    if p.ndim!=1 or w.ndim!=1 or len(p)!=len(w)+1 or len(w)<1:
        raise ValueError('one more price than target required')
    if not np.isfinite(p).all() or np.any(p<=0) or not np.isfinite(w).all() or np.any((w<0)|(w>1)):
        raise ValueError('positive prices and long-only weights [0,1] required')
    if not np.isfinite([fee_bps,slippage_bps]).all() or min(fee_bps,slippage_bps)<0 or max(fee_bps,slippage_bps)>=10000:
        raise ValueError('invalid costs')
    fee,slip=fee_bps/1e4,slippage_bps/1e4
    buy=(1+fee)*(1+slip)-1
    sell=1-(1-fee)*(1-slip)
    cash,shares=1.,0.
    equity=[1.]
    turnover=costs=0.
    trades=0
    for i,target in enumerate(w):
        before=cash+shares*p[i]
        diff=target*before-shares*p[i]
        rate=buy if diff>=0 else sell
        # Solve target weight against wealth after its own execution cost.
        dq=diff/(p[i]*(1+target*rate if diff>=0 else 1-target*rate))
        cost=abs(dq)*p[i]*rate
        cash-=dq*p[i]+cost
        shares+=dq
        if cash < -1e-10 or shares < -1e-10:
            raise ArithmeticError('self-financing invariant failed')
        turnover+=abs(dq)*p[i]/before
        costs+=cost
        trades+=int(abs(dq)*p[i]>1e-12)
        equity.append(cash+shares*p[i+1])
    if liquidate and shares>0:
        before=cash+shares*p[-1]
        cost=shares*p[-1]*sell
        turnover+=shares*p[-1]/before
        costs+=cost
        equity[-1]=before-cost
        trades+=1
    return {'equity':np.array(equity),'costs':float(costs),'turnover':float(turnover),'trades':trades}


def metrics(equity, periods_per_year=252):
    e=np.asarray(equity,float)
    if e.ndim!=1 or len(e)<2 or not np.isfinite(e).all() or np.any(e<=0) or periods_per_year<=0:
        raise ValueError('positive finite equity series required')
    r=e[1:]/e[:-1]-1
    sd=float(np.std(r,ddof=1)) if len(r)>1 else 0.
    sharpe=float(np.mean(r)/sd*math.sqrt(periods_per_year)) if sd>1e-14 else None
    return {'total_return':float(e[-1]/e[0]-1),'max_drawdown':float(np.max(1-e/np.maximum.accumulate(e))),
            'sharpe_naive':sharpe,'periods':len(r),'annualization':periods_per_year,
            'mean_return':float(np.mean(r)), 'volatility':sd*math.sqrt(periods_per_year)}


def block_bootstrap_mean(returns, block_size=10, repetitions=500, seed=0):
    """Circular moving-block percentile interval; exploratory, not a promotion test."""
    x=np.asarray(returns,float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or not 1<=block_size<=len(x) or repetitions<20:
        raise ValueError('invalid bootstrap input')
    rng=np.random.default_rng(seed)
    n_blocks=math.ceil(len(x)/block_size)
    starts=rng.integers(0,len(x),size=(repetitions,n_blocks))
    indices=(starts[:,:,None]+np.arange(block_size))%len(x)
    samples=x[indices.reshape(repetitions,-1)[:,:len(x)]].mean(axis=1)
    return tuple(float(v) for v in np.quantile(samples,[.025,.975]))
