# Research programme and synthesis

## Goals and constraints

1. Establish whether geometric memory adds reproducible information beyond simpler financial models.
2. Recover and test specific links between macro conditions, crowd behaviour, microstructure and price/volatility dynamics.
3. Build a reusable mathematics lab for nonlinear ODEs, PDEs, geometry and complex analysis, with finance as one empirical consumer.
4. Turn each idea into a versioned hypothesis, reproducible experiment, failure analysis and next decision.
5. Build a complete validation suite before the proposed $10 pilot; scale only against predeclared evidence and risk limits.

No Mamba-style zero-order-hold cell is used as the proposed architecture. Numerical integration still needs a step rule and convergence tests. Linear models remain controls. The fact that markets contain nonlinear processes does not establish a trading edge. Every experiment must distinguish mathematical validity, predictive validity and executable economic value.

Initial operating assumptions: local CPU first; synthetic data first; one GPU job at a time when funded; five-minute total lease ceiling as a starting design parameter; zero current cloud/payment allowance. These are configurable engineering defaults, not an estimate of available credits or a financial allocation. Asset universe, horizon, licensed feed and actual cost envelope remain to be selected from evidence.

## Tracks

| ID | Track and question | First concrete output | Falsification / dependency |
|---|---|---|---|
| FD-01 | Foundations: what exactly follows from the existing Riccati and winding manuscripts? | Assumption/theorem/conjecture register, independent residual checks, chart atlas | Reject claims failing residual or coordinate-consistency checks; S01–S02 |
| FD-02 | Nonlinear dynamics: does torus/projective memory help? | Controlled oscillator + CP1 state-space prototype; learned bounded gates | Equal-budget ablations; constant/variable frequency and null data; S03–S07 |
| FD-03 | ODE/PDE geometry: which structure deserves preservation? | Riccati chart tests, Lie/exponential integrator convergence, dissipative dynamics; later Fokker–Planck and population equations | Conservation only where justified; boundary, positivity and mass checks; S06,S13,S14 |
| FD-04 | Macro and political economy: when does information change regimes? | Vintage-aware rates, inflation, credit, liquidity, fiscal and commodity feature catalogue | Release lag, revision and slow-frequency controls; no monthly-to-minute sample inflation; S09,S12 |
| FD-05 | Behaviour and narratives: do attention and herding add information? | Timestamped narrative taxonomy; disagreement, attention diffusion, reference points and crowding features | Price-only/volume-only controls, delayed-text and shuffled-text placebos; separate professional and retail behaviour; S10,S11,S27,S28 |
| FD-06 | Microstructure and execution: can a signal survive trading? | Spread/depth, imbalance, queue, impact, auction and venue model | Fee/slippage/latency/partial-fill stress, real tick data before intraday claims; S08 |
| FD-07 | AI and pattern recognition: what representation transfers? | Ridge/tree/GRU/CDE/S4/S5/LRU comparison; joint versus separate encoders | Matched capacity/compute, pinned models, leakage probes; S03–S05,S15,S29–S31 |
| FD-08 | Statistical evidence: are patterns robust? | Nested walk-forward, trial ledger, block/bootstrap uncertainty, surrogate controls, PBO/DSR research | Dependent trials, multiple tests, data snooping and selection penalties; S07,S16,S17 |
| FD-09 | Risk and portfolios: is the signal useful after allocation? | Exposure decomposition, drawdown/tail loss, turnover/capacity and portfolio tests | Factor-matched benchmarks, concentration and stress constraints |
| FD-10 | Systems and observability: can every run be reproduced and terminated? | Durable manifests, leases, experiment queue, artifact and metric contracts | Host-loss, timeout, orphan, ingestion and restore drills; S18–S20 |
| FD-11 | Payments and accounting: can resources be funded and reconciled? | Payment-intent schema, receipt links, invoice drafts and expense ledger | Unknown payment quarantine, duplicate events, evidence completeness; S21–S26 |
| FD-12 | Synthesis and scientific governance: how do tracks change each other? | Hypothesis graph and constrained proposal worker | Separate proposer/evaluator; every derivative counts as another trial; final holdout inaccessible to proposer |

Each task record contains owner, source IDs, exact question, null/alternative, observable proxy, dataset/version, availability timestamp rule, intervention or comparison, selection budget, code revision, metric, uncertainty, decision rule, output and next dependency. Correlation, mechanism and causal effect are separate fields. An LLM explanation is a proposal, not experimental evidence.

## The synthesis mechanism

Use a multiscale latent state: slow macro/credit state; medium-horizon trends and narrative diffusion; fast liquidity/order flow; phase and nonlinear memory; latent uncertainty. Condition observation and transition models on the information available at decision time. Alternative models may fuse these through gating, mixture-of-experts, graph couplings or controlled vector fields. Do not force all channels into a single clock or impose physical conservation on market prices.

Example causal hypothesis: a policy surprise changes financing expectations; funding constraints alter participant positioning; crowded narratives amplify flows; limited depth increases impact. Measure each link and its delay. Confounding, anticipatory positioning, reverse causality and omitted news remain rival explanations. Macro news published after the decision cannot enter the feature tensor.

Use a hypothesis graph where nodes are mechanisms and edges are proposed dependencies, with explicit evidence strength. Do not convert several correlated LLM opinions into independent Bayesian observations. Calibration must measure predictions against later outcomes.

## First experiments

| ID | Experiment | Comparison / outcome |
|---|---|---|
| FD-E01 | Synthetic numerical and experiment smoke test | Implemented: null, planted quasi-periodic, regime-switch data × 3 seeds × 5 strategies; infrastructure evidence only |
| FD-E02 | Geometry ablation | Phase-only, projective-only, gate-free, uncoupled, full, and parameter-matched lag/clock controls |
| FD-E03 | Nonlinearity and chaos discrimination | AR/GARCH, stochastic-volatility and IAAFT surrogates; known oscillator and chaotic reference systems; out-of-distribution periods |
| FD-E04 | Causal macro conditioning | First-release/vintage features versus revised-data contamination and delayed release controls |
| FD-E05 | Narrative incremental value | Identical price model with/without timestamped text, disagreement and diffusion; future-text negative control |
| FD-E06 | Cross-track interaction | Macro × crowding × liquidity interaction versus additive and shuffled-channel versions |
| FD-E07 | Geometry/integration audit | dt, dt/2, dt/4; irregular sampling; chart switches; perturbation sensitivity; float32 versus float64 |
| FD-E08 | Execution survival | Costs/latency doubled, liquidity reduced, missing bars, halts, stressed spreads and venue limits |
| FD-E09 | Infrastructure fault injection | Unknown create outcome, controller death, worker timeout, upload failure, missing telemetry, uncertain payment |
| FD-E10 | Shadow forward evaluation | Sealed model and policy emit timestamped predictions; reconcile with later executable quotes, no orders |

The initial planted oscillator frequency overlaps the prototype's configured base frequency. That makes E01 a deliberately favourable engineering fixture. Its return figures cannot substantiate model superiority. E02/E03 must remove that overlap and use unseen frequencies, equal capacity and null rejection controls.

## Milestones and acceptance

| Milestone | Acceptance | Current state |
|---|---|---|
| M0 — Source and hypothesis baseline | Repository lineage, source ledger, 12 tracks, explicit exclusions and assumptions | Delivered initial baseline; literature expansion open |
| M1 — Numerical foundations | Independent residual checks, pole handling, phase conservation, convergence/conditioning evidence | Primitive tests implemented; full audit open |
| M2 — Point-in-time data and baselines | Licensed immutable data; universe/calendar/corporate actions; pinned baseline reproduction | Specification only |
| M3 — Full historical validation | Nested selection, locked chronology, all trial records, execution accounting and stress tests | Foundation only; market validation blocked on M2 |
| M4 — Bounded experiment infrastructure | Verified provider adapter, external reaper, artifact persistence, spend reconciliation, monitoring ingestion | Offline lease/analysis prototypes; deployment open |
| M5 — Shadow operation | Independent forward evidence, incident drills and preregistered risk limits | Not started |
| M6 — $10 pilot | Gate review, confirmed currency/venue feasibility, no leverage, cumulative loss and spend limits | Not started; no trade authority issued |
| M7 — Conditional scaling | New forward evidence at each tier, capacity/cost stability, explicit revised allocation | Not started; never increase solely after a lucky streak |

Calendar deadlines should follow data and infrastructure availability. Do not invent completion dates for an open research programme. Termination rules: failed numerical claim, leakage discovery, no incremental information after the registered budget, unresolvable data licensing, or economic value erased by costs. Negative results are retained and can close a track.
