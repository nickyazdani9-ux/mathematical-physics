# Financial dynamics: mathematics, behaviour and AI

Research checkpoint FD-20261027-v1. This programme extends the repository's mathematics and physics work into testable financial modelling. The organising idea is a coupled system of changing trends, phases, participant beliefs, information and liquidity. Quasi-periodicity, chaos and profitable predictability are hypotheses to test.

The programme combines nonlinear selective state spaces, differential geometry, complex analysis, macroeconomics, behavioural finance and AI. It also specifies the infrastructure needed to turn experiments into durable evidence: chronological validation, short-lived GPU jobs, an analysis worker, monitoring, credential mediation and accounting.

## Start here

| Document | Purpose |
|---|---|
| [PROGRAMME.md](PROGRAMME.md) | Research tracks, hypotheses, synthesis experiments and milestones |
| [MODEL.md](MODEL.md) | Winding/rotation and Riccati derivation, mathematical constraints and novelty questions |
| [VALIDATION.md](VALIDATION.md) | Full backtesting specification and promotion gates |
| [EXPERIMENT_INFRA.md](EXPERIMENT_INFRA.md) | Swarm/Vast lifecycle, bounded feedback and observability |
| [PAYMENTS_ACCOUNTING.md](PAYMENTS_ACCOUNTING.md) | Credential, payment, invoice and expense interfaces |
| [SOURCES.md](SOURCES.md) | 31 references with explicit reading depth and remaining gaps |
| [STATE.md](STATE.md) | What is implemented, measured, designed and still blocked |
| [Monitoring](../../monitoring/financial_dynamics/README.md) | Elasticsearch event mapping and Datadog collector setup contract |

## Run the offline foundation

From the repository root, using Python 3.12:

```bash
python -m pip install -r research/financial_dynamics/requirements.txt
python -m unittest discover -s engineering/financial_dynamics/tests -v
python -m engineering.financial_dynamics.experiments run --output runs/fd-example
python -m engineering.financial_dynamics.experiments analyze --output runs/fd-example
```

Use a new output directory for every run. The command creates 45 synthetic backtests, checksum manifests, JSON events, an Elasticsearch Bulk import file and two Prometheus-format run counters. The analysis command verifies the artifacts and writes proposed follow-up experiments. It does not launch them.

The current cell has fixed dynamics with a fitted ridge readout. It is an executable research seed, not a trained neural architecture. The bar simulator is single-asset, long-only and fractional-share. No real market backtest, rented GPU, deployed monitoring, payment or trade has been performed.

## Programme relationships

[Linear project](https://linear.app/atlas-inference/project/financial-dynamics-mathematics-behaviour-and-ai-743aad7e3c05). Mathematics and reusable numerical work stay here. The Swarm owns scheduling; Credential Manager owns secret custody and scoped authority; Accounting owns private entities, invoices and ledgers; F Secure owns host and workload controls. Operational records and credentials do not belong in this public repository.
