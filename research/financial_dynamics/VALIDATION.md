# Backtesting and evidence requirements

## Timing contract

Every record carries event time, provider publication time, received time, revision/vintage, exchange/calendar timezone and a provenance digest. Decision time is explicit. A feature is usable only after its availability time plus the model's declared processing delay. Economic reference period is not release time. ALFRED stores vintages; intraday claims need actual release timing, not just a daily vintage date. [S12](SOURCES.md#s12)

Train-only transformations include scaling, imputation, filtering, dimensionality reduction, vocabulary, feature selection and hyperparameter search. Avoid centered rolling features, whole-series Hilbert transforms, hindsight regime labels and splines fitted through future observations. Maintain a frozen as-of universe with delisted instruments and historical corporate actions. Distinguish executable raw prices, total-return accounting and provider-adjusted data.

Text inputs require publication/update/ingestion times and document snapshots. A modern pretrained LLM may memorise historical outcomes: rolling evaluation alone does not remove that contamination. Use cutoff-controlled models where possible, blind outcome-sensitive prompts, test timestamp leakage and report residual contamination. No untrusted document may change the execution or payment policy.

## Chronological evaluation

Use nested expanding or rolling walk-forward evaluation. Inner development folds select parameters; outer folds estimate the selected pipeline. Training labels that overlap the evaluation interval are purged. The included splitter uses only past training rows and requires label_end < test_start - gap; a symmetric cross-validation method would also need an embargo. Purging is based on label intervals and data availability, not an arbitrary fixed percentage.

Reserve a final time block or genuinely future shadow period under a separate evaluator identity. The proposal/analysis worker cannot read its labels, errors, plots or detailed metrics. One approved evaluation consumes a holdout version. Retuning after its result makes that period development data; a new independent period is needed. Keep every trial, seed, failed attempt, feature set, stopping choice and human intervention in the experiment family ledger.

## Baseline ladder

Cash and buy-and-hold; naive/persistence forecasts; lagged-return ridge; momentum/reversal; volatility scaling; AR/VAR, Kalman/state-space, GARCH/stochastic-volatility and regime-switching models; tree boosting; GRU/LSTM; CDE; S4/S5/LRU; then the proposed geometric family. Add matched parameter, compute, feature and clock-memory controls. Reference financial ML already studies nonlinear interactions. [S15,S29–S31](SOURCES.md)

The prototype implements only cash, buy-and-hold, sign-of-trailing-mean momentum, lagged-return ridge and geometric-feature ridge. It charges entry/rebalancing/terminal costs and waits a full synthetic interval between observing a feature and execution. Its folds fit standardisation on training rows only. It does not implement the whole ladder.

## Execution and accounting suite

Before real data claims, implement multi-asset cash/share accounting; fills against bid/ask; commissions/minimum fees; spread, nonlinear impact and liquidity; order latency; partial fills and queue priority; rejected/cancelled orders; stale/missing data; halts/auctions; currency conversion; distributions, splits and delisting; margin and borrow only if later authorised; funding and financing costs; capacity and position limits; deterministic event ordering; round-trip reconciliation to independent reference cases.

Check self-financing, conservation of shares through splits, no use of future fills, fill-volume limits, bankruptcy handling, consistent marks, portfolio turnover, cost monotonicity, accounting equality, reproducible replay, and continuous treatment across fold boundaries. Integer lots/minimum notional can make a $10 pilot infeasible at a chosen venue. Fractional simulated weights do not establish real executability.

## Metrics and uncertainty

| Layer | Required measurements |
|---|---|
| Prediction | Proper distributional score, MAE/RMSE as appropriate, calibration/coverage, information coefficient and horizon breakdown |
| Trading | Net/gross return, excess return versus matched benchmark, volatility, drawdown/duration, expected shortfall, turnover, exposure, capacity, trade count and average net trade |
| Stability | Performance by time/regime/asset, feature drift, parameter sensitivity, path/seed sensitivity, phase aliasing and solver residual |
| Evidence | Number/effective dependence of trials, uncertainty under serial dependence, PBO/DSR diagnostics, null/placebo outcomes, untouched evaluation record |
| Operations | End-to-end latency, missingness, GPU/CPU time, memory, runtime, estimated/actual cost, artifact completeness, orphan lease count |

Do not annualise intraday Sharpe with sqrt(time) without checking serial dependence and the actual calendar. The current `sharpe_naive` is explicitly descriptive; the block-bootstrap mean interval is exploratory and lacks selection adjustment. PBO and DSR are planned research components, not currently implemented eligibility gates. Their assumptions must be checked, including correlated trials and nonstationarity. [S16–S17](SOURCES.md)

Surrogate controls include white/coloured noise, AR/GARCH, stochastic volatility, phase-randomised and IAAFT series, time-shifted channels and synthetic known dynamics. A nonlinear test rejecting one null does not prove chaos. A positive Lyapunov estimate from noisy data needs sensitivity and finite-sample analysis. [S07](SOURCES.md#s07)

## Promotion protocol

G0: formulas and numerical invariants pass. G1: data lineage/timing and baseline reproduction pass. G2: development comparisons and full trial ledger pass. G3: preregistered frozen evaluation passes its economic and uncertainty thresholds. G4: shadow execution and operational drills pass. G5: a precisely scoped $10 pilot passes venue feasibility, loss cap, currency and account-authority checks. G6: further allocation requires new forward evidence and a revised envelope.

Numerical thresholds cannot be selected after seeing the result. Choose hurdle return after fees, tolerated drawdown/tail loss, dependent-sample confidence requirement and minimum observation duration during protocol registration for the selected market. Infrastructure, cloud spend and trading capital are distinct budgets. No automatic doubling, martingale recovery or leverage is part of this programme. Stop on data corruption, broken reconciliation, stale predictions, exceeded limits or unexplained fill divergence.
