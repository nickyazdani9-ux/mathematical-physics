"""Offline-testable lease protocol. No Vast.ai adapter is enabled.

SQLite is a single-controller reference implementation. A provider timeout
must be enforced externally; this module cannot stop a hung remote API or
survive host loss without an independent lease reconciler.
"""
from dataclasses import dataclass
import math
import re
import sqlite3
import time


@dataclass(frozen=True)
class ExperimentLease:
    run_id: str
    runtime_seconds: int
    max_cost_cents: int
    quote_cents: int

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',self.run_id):
            raise ValueError('invalid run ID')
        if any(type(v) is not int for v in [self.runtime_seconds,self.max_cost_cents,self.quote_cents]):
            raise ValueError('integer seconds and minor currency units required')
        if not 1<=self.runtime_seconds<=300 or not 0<=self.quote_cents<=self.max_cost_cents or self.max_cost_cents<=0:
            raise ValueError('invalid runtime or budget cap')


class LeaseStore:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=5,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS leases (run_id TEXT PRIMARY KEY, budget INTEGER NOT NULL, deadline REAL NOT NULL, state TEXT NOT NULL, instance TEXT)')

    def reserve(self,lease,allowance_cents,now):
        if type(allowance_cents) is not int or allowance_cents<0 or not math.isfinite(now):
            raise ValueError('invalid authority allowance or clock')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM leases WHERE run_id=?',(lease.run_id,)).fetchone():
                raise ValueError('run ID already used; reconcile instead of retrying allocation')
            # Completed runs also consume their conservative cap until ledger settlement.
            committed=self.db.execute('SELECT COALESCE(SUM(budget),0) FROM leases').fetchone()[0]
            if committed+lease.max_cost_cents>allowance_cents:
                raise PermissionError('cumulative experiment allowance exceeded')
            self.db.execute('INSERT INTO leases VALUES (?,?,?,?,?)',(lease.run_id,lease.max_cost_cents,now+lease.runtime_seconds,'ALLOCATING',None))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def update(self,run_id,state,instance=None):
        self.db.execute('UPDATE leases SET state=?,instance=COALESCE(?,instance) WHERE run_id=?',(state,instance,run_id))

    def reserved(self):
        return self.db.execute("SELECT COALESCE(SUM(budget),0) FROM leases WHERE state!='DESTROYED'").fetchone()[0]

    def close(self):
        self.db.close()


def run_leased(store,provider,lease,allowance_cents=0,clock=time.time):
    now=clock()
    store.reserve(lease,allowance_cents,now)
    deadline=now+lease.runtime_seconds
    instance=None
    out={'run_id':lease.run_id,'state':'ALLOCATING','outcome':'failed'}
    try:
        try:
            created=provider.create(lease.run_id,deadline)
            if not isinstance(created,str) or not created:
                raise ValueError('provider returned no instance identifier')
            instance=created
        except Exception as error:
            out.update(state='ALLOCATION_UNKNOWN',error_type=type(error).__name__)
            store.update(lease.run_id,out['state'])
            return out
        store.update(lease.run_id,'RUNNING',instance)
        remaining=deadline-clock()
        if remaining<=0:
            raise TimeoutError('allocation exhausted lease')
        result=provider.run(instance,timeout_seconds=remaining)
        if clock()>deadline:
            raise TimeoutError('run exceeded lease')
        if not isinstance(result,dict) or not result.get('artifact_digest'):
            raise ValueError('durable artifact acknowledgement missing')
        out.update(outcome='succeeded',artifact_digest=result['artifact_digest'])
    except Exception as error:
        out['error_type']=type(error).__name__
    finally:
        if instance:
            try:
                provider.destroy(instance)
                if not provider.is_destroyed(instance):
                    raise RuntimeError('destruction not confirmed')
                out['state']='DESTROYED'
            except Exception as error:
                out.update(state='CLEANUP_REQUIRED',cleanup_error_type=type(error).__name__)
            store.update(lease.run_id,out['state'],instance)
    return out
