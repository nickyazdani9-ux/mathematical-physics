"""Reproducible synthetic experiments and a one-shot analysis worker.

Usage: python -m engineering.financial_dynamics.experiments run --output runs/demo
       python -m engineering.financial_dynamics.experiments analyze --output runs/demo
No external services, GPU rental, money movement or real market data.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from .geometry import PhaseState, selective_step
from .data import purged_walk_forward
from .backtest import simulate,metrics,block_bootstrap_mean


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def make_data(scenario,seed,n):
    if scenario not in ('null','quasiperiodic','regime') or n<20:
        raise ValueError('invalid synthetic scenario')
    rng=np.random.default_rng(seed)
    t=np.arange(n)
    noise=rng.normal(0,.008,n)
    if scenario=='null':
        return noise
    signal=.005*np.sin(.13*t)+.003*np.sin(np.sqrt(2)*.13*t)
    if scenario=='regime':
        signal[n//2:]=.005*np.sin(.21*t[n//2:]+1)
        noise[n//2:]*=2
    return signal+noise


def causal_features(returns):
    r=np.asarray(returns,float)
    if r.ndim!=1 or not np.isfinite(r).all():
        raise ValueError('finite scalar returns required')
    phase,h=PhaseState(0.,0),np.array([1.+0j,0j])
    features=[]
    for i,value in enumerate(r):
        phase,h,geo=selective_step(phase,h,value/.01,1.)
        # Fixed scale is part of synthetic protocol; no full-series fitting.
        trailing=r[max(0,i-9):i+1]
        features.append(np.r_[value,np.mean(trailing),np.std(trailing),geo])
    return np.array(features)


def ridge_predict(x,y,test,alpha=1.):
    mean=x.mean(axis=0)
    sd=x.std(axis=0);sd=np.where(sd<1e-10,1.,sd)
    x=(x-mean)/sd;test=(test-mean)/sd
    center=y.mean()
    beta=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@(y-center))
    return center+test@beta


def run_suite(output,seeds=(7,19,41),n=600):
    if n<200 or not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('need >=200 samples and distinct seeds')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    code={p.name:digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))}
    config={'seeds':list(seeds),'n':n,'scenarios':['null','quasiperiodic','regime'],
            'min_train':100,'test_size':64,'label_offset':2,'gap':0,
            'models':['cash','buy_hold','momentum','ridge','geometric_ridge'],
            'fee_bps':5,'slippage_bps':5,'ridge_alpha':1.,'signal_scale':.01}
    run_id=digest(canonical({'config':config,'code':code}).encode())[:20]
    timestamp=datetime.now(timezone.utc).isoformat()
    events=[];results=[];datasets={}
    for scenario in config['scenarios']:
        for seed in seeds:
            r=make_data(scenario,seed,n+2)
            datasets[f'{scenario}-{seed}']=digest(r.astype('<f8').tobytes())
            p=100*np.exp(np.r_[0,np.cumsum(r)])
            x=causal_features(r[:n]);y=r[2:n+2]
            # r[i] observed at p[i+1]; wait a full interval, execute at p[i+2].
            times=np.arange(n)+1
            folds=list(purged_walk_forward(times,times+2,min_train=100,test_size=64))
            ix=np.concatenate([test for _,test in folds])
            preds={name:[] for name in ['ridge','geometric_ridge']}
            for train,test in folds:
                for name,cols in [('ridge',3),('geometric_ridge',x.shape[1])]:
                    preds[name].extend(ridge_predict(x[train,:cols],y[train],x[test,:cols]))
            for model in config['models']:
                if model=='cash': weights=np.zeros(len(ix));mse=None
                elif model=='buy_hold':weights=np.ones(len(ix));mse=None
                elif model=='momentum':weights=(x[ix,1]>0).astype(float);mse=None
                else:
                    prediction=np.asarray(preds[model])
                    weights=np.clip(.5+prediction/config['signal_scale'],0,1)
                    mse=float(np.mean((prediction-y[ix])**2))
                report=simulate(p[ix[0]+2:ix[-1]+4],weights,fee_bps=5,slippage_bps=5)
                m=metrics(report['equity'])
                returns=report['equity'][1:]/report['equity'][:-1]-1
                row={'scenario':scenario,'seed':int(seed),'model':model,**m,
                     'mse':mse,'turnover':report['turnover'],'costs':report['costs'],
                     'trades':report['trades'],'mean_ci_exploratory':block_bootstrap_mean(returns,seed=seed),
                     'folds':len(folds),'evaluation':'development_walk_forward','data_kind':'synthetic'}
                results.append(row)
                events.append({'@timestamp':timestamp,'schema_version':1,'run_id':run_id,'event':'backtest_completed',**row})
    summary={'schema_version':1,'run_id':run_id,'config':config,'results':results,
             'data_kind':'synthetic','promotion_eligible':False,'final_holdout_accessed':False,
             'cloud_spend_usd':0,'elapsed_seconds':time.monotonic()-started,
             'limitations':['fixed geometric cell, ridge readout only','synthetic data only',
                            'single-asset long-only simulator','naive Sharpe is descriptive',
                            'no calibrated market-impact, statistical promotion gate or live adapter']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (output/'events.ndjson').write_text(''.join(canonical(e)+'\n' for e in events))
    bulk=[]
    for e in events:
        doc_id=digest(canonical(e).encode())
        bulk.extend([canonical({'create':{'_index':'financial-dynamics-v1','_id':doc_id}}),canonical(e)])
    (output/'elasticsearch.bulk.ndjson').write_text('\n'.join(bulk)+'\n')
    (output/'metrics.prom').write_text(f'financial_dynamics_completed_backtests {len(results)}\nfinancial_dynamics_cloud_spend_usd 0\n')
    manifest={'schema_version':1,'run_id':run_id,'config':config,'source_sha256':code,
              'dataset_sha256':datasets,'python':platform.python_version(),'numpy':np.__version__,
              'scipy':scipy.__version__,'created_at':timestamp,'artifacts':{
                  p.name:digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(canonical(manifest)+'\n')
    summary['manifest_sha256']=digest((output/'manifest.json').read_bytes())
    return summary


def analyze(output):
    """Read development artifacts, verify checksums, propose work; never execute it."""
    output=Path(output)
    manifest_raw=(output/'manifest.json').read_bytes()
    manifest=json.loads(manifest_raw)
    for name,expected in manifest['artifacts'].items():
        if Path(name).name!=name or digest((output/name).read_bytes())!=expected:
            raise ValueError('artifact integrity failure')
    summary=json.loads((output/'summary.json').read_text())
    if summary.get('final_holdout_accessed') or summary.get('data_kind')!='synthetic':
        raise ValueError('this prototype accepts synthetic development artifacts only')
    comparisons=[]
    for scenario in summary['config']['scenarios']:
        rows=[r for r in summary['results'] if r['scenario']==scenario]
        by={m:np.mean([r['total_return'] for r in rows if r['model']==m]) for m in summary['config']['models']}
        comparisons.append({'scenario':scenario,'mean_return_by_model':by})
    return {'schema_version':1,'parent_run_id':manifest['run_id'],
            'source_manifest_sha256':digest(manifest_raw),'auto_execute':False,'promotion_eligible':False,
            'comparisons':comparisons,'proposals':[
                {'id':'FD-E02','hypothesis':'Incremental geometry helps beyond lagged returns',
                 'test':'Matched parameter-budget ablations: remove phase, projective state, gate, then coupling.'},
                {'id':'FD-E03','hypothesis':'The apparent effect survives stochastic nulls',
                 'test':'GARCH and IAAFT nulls, new seeds, higher costs and altered frequencies.'}],
            'decision':'Engineering smoke test only. No real-market performance inference.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='run':
        result=run_suite(args.output)
        print(canonical({'run_id':result['run_id'],'backtests':len(result['results']),
                         'seconds':result['elapsed_seconds'],'promotion_eligible':False}))
    else:
        result=analyze(args.output)
        path=args.output/'analysis.json'
        with path.open('x') as f:
            f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(canonical(result))


if __name__=='__main__':
    main()
