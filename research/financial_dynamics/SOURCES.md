# Sources and reading ledger

Research checkpoint: 27 October 2026 (Australia/Melbourne). This is an initial evidence map, not an exhaustive systematic review. Retrieval levels are explicit. Abstract-only and discovered papers support topic selection, not reproduced empirical or theorem claims. Original derivations in the model note are independently checked algebraically and numerically. No wholesale paper text is included.

## S01

[Yazdani: Riccati manuscript](https://github.com/nickyazdani9-ux/mathematical-physics/blob/4f8e4a727761b904ae7a94289a25240e056d9d87/mathematics/riccati_odes/riccati_paper.tex)

Coverage: Source inspected.

Provisional manuscript; distinguish constant fixed points from instantaneous roots and WKB approximations.

## S02

[Yazdani: helical geometry manuscript](https://github.com/nickyazdani9-ux/mathematical-physics/blob/4f8e4a727761b904ae7a94289a25240e056d9d87/physics/wips/docs/gtor_revised_cors1-20.tex)

Coverage: Source inspected.

Helical phase theta=v+2*pi*k*u and periodic closure motivate representation. No market mechanism or physical claim is imported as established.

## S03

[Gu and Dao: Mamba](https://arxiv.org/html/2312.00752v2)

Coverage: Methods inspected; v2.

Input-dependent selection is separable from its discretisation and scan implementation. No Mamba cell is used in this prototype.

## S04

[Muca Cirone et al.: theoretical foundations of selective SSMs](https://arxiv.org/html/2402.19047v4)

Coverage: Introduction and theoretical framing inspected; v4 2025-01-06.

Connects selectivity to rough paths/signatures; motivates cross-channel interaction ablations. No finance performance implication.

## S05

[Kidger et al.: neural controlled differential equations](https://arxiv.org/html/2005.08926v2)

Coverage: Methods framing inspected; v2.

Continuous-time memory driven by observations; online interpolation must use available observations only.

## S06

[Cariñena and de Lucas: integrability through Riccati equations](https://arxiv.org/html/1002.0530v3)

Coverage: Theory framing inspected; v3; J Nonlinear Math Phys 2011.

Lie-system reductions and transformation conditions; general Riccati equations are not solved by algebraic roots alone.

## S07

[Schreiber and Schmitz: surrogate time series](https://arxiv.org/abs/chao-dyn/9909037)

Coverage: Abstract and metadata only; Physica D 2000.

Test nonlinear explanations against specified stochastic nulls. Rejection of one null is not proof of deterministic chaos.

## S08

[Bacry et al.: Hawkes processes in finance](https://arxiv.org/html/1502.04592v2)

Coverage: Definitions and stability section inspected; v2.

Event excitation and microstructure. For the standard nonnegative linear Hawkes model, integrated-kernel spectral radius below one is the stationary stability condition.

## S09

[Borio: financial cycle and macroeconomics](https://www.bis.org/publications/working-paper-395-financial-cycle-and-macroeconomics-what-have-we-learnt)

Coverage: Discovery/landing-page coverage only.

Prioritise credit, leverage and liquidity. Full paper and empirical calibration remain queued.

## S10

[Shiller: Narrative Economics](https://www.nber.org/system/files/working_papers/w23075/w23075.pdf)

Coverage: First 6 PDF pages inspected; January 2017 working paper.

Narrative diffusion is a research mechanism; reverse causality and measurement remain material.

## S11

[Alevy, Haigh and List: information cascades experiment](https://www.nber.org/papers/w12767)

Coverage: Abstract and metadata inspected; WP 2006; published J Finance 2007.

Experimental market professionals and students differ. Avoid assuming one universal retail psychology.

## S12

[ALFRED vintage data documentation](https://alfred.stlouisfed.org/help/downloaddata)

Coverage: Official documentation inspected.

Retrieve releases available at past dates; release/vintage dates require timestamp precision for intraday research.

## S13

[Li et al.: Fourier Neural Operator](https://arxiv.org/abs/2010.08895)

Coverage: Abstract and metadata only; v3.

Operator learning across PDE families; no claim that market dynamics obey a known PDE.

## S14

[Brunton et al.: Modern Koopman Theory](https://arxiv.org/abs/2102.12086)

Coverage: Abstract and metadata only; v2.

Nonlinear dynamics can be represented by a linear operator on observables; finite useful closures are difficult.

## S15

[Smith et al.: S5](https://arxiv.org/abs/2208.04933)

Coverage: Abstract and metadata only; v3.

Multi-input multi-output state space reference and computational baseline.

## S16

[Bailey et al.: Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

Coverage: First 8 PDF pages inspected; February 2015 version.

Multiple strategy trials require selection-bias diagnostics; CSCV/PBO does not replace a chronological untouched evaluation.

## S17

[Bailey and López de Prado: Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

Coverage: First 8 PDF pages inspected; July 2014 version.

Accounts for multiple selection and non-normality under its assumptions; dependence and trial correlation must also be assessed.

## S18

[Vast.ai Python SDK quickstart](https://docs.vast.ai/sdk/python/quickstart)

Coverage: Official lifecycle sections inspected 2026-10-27.

Create returns new_contract; storage starts at creation. Poll with deadlines, export results, destroy and confirm. Stopping alone is insufficient for full cleanup.

## S19

[Datadog OpenTelemetry Collector setup](https://docs.datadoghq.com/opentelemetry/setup/collector_exporter/)

Coverage: Official configuration inspected 2026-10-27.

Collector/OTLP integration supported. Pin chosen collector version and verify telemetry ingestion; documentation is not evidence of deployed monitoring.

## S20

[Elastic Docker Compose installation](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-docker-compose)

Coverage: Official setup inspected 2026-10-27.

Security/password setup and localhost port binding; version pin, persistent volume and retention validation remain deployment prerequisites.

## S21

[ATO tax invoices](https://www.ato.gov.au/businesses-and-organisations/gst-excise-and-indirect-taxes/gst/tax-invoices)

Coverage: Official page inspected 2026-10-27.

Required supplier/ABN/date/items/GST fields; buyer identity or ABN at AUD 1000 or more; applicable request timing.

## S22

[ATO GST credits](https://www.ato.gov.au/businesses-and-organisations/gst-excise-and-indirect-taxes/gst/claiming-gst-credits/when-you-can-claim-a-gst-credit)

Coverage: Official page inspected 2026-10-27.

Registration, business use, GST in purchase and evidence conditions; valid tax invoice normally needed above AUD 82.50.

## S23

[ATO business deductions](https://www.ato.gov.au/businesses-and-organisations/income-deductions-and-concessions/income-and-deductions-for-business/deductions)

Coverage: Official page inspected 2026-10-27.

Business purpose and records are required; exclude private portions and avoid deducting a GST amount already claimable as a credit.

## S24

[Stripe webhook documentation](https://docs.stripe.com/webhooks)

Coverage: Official event behaviour and verification sections inspected 2026-10-27.

Verify raw-body signatures with official library, handle duplicate events, retries and unordered delivery.

## S25

[Stripe idempotent requests](https://docs.stripe.com/api/idempotent_requests)

Coverage: Official API documentation inspected 2026-10-27.

Persist local operation IDs; API v1 idempotency retention does not provide indefinite deduplication.

## S26

[ANZ PayTo](https://www.anz.com.au/support/online-banking/payments/pay-to/)

Coverage: Official product documentation inspected 2026-10-27.

Merchant creates agreement; account holder authorises it through banking channel. Not evidence of a universal personal-account payment API.

## S27

[Kahneman and Tversky: Prospect Theory](https://kahneman.scholar.princeton.edu/document/11)

Coverage: Primary-source search extract; full methods queued.

Reference dependence and choice under risk inform hypotheses; trader behaviour must be measured rather than presumed.

## S28

[Brock and Hommes: heterogeneous beliefs and routes to chaos](http://research.economics.unsw.edu.au/vpanchenko/teaching/bh.pdf)

Coverage: Primary paper discovered; full read queued.

Heterogeneous expectations provide a simulation candidate, not evidence that any observed market is deterministic chaos.

## S29

[Gu et al.: S4](https://arxiv.org/abs/2111.00396)

Coverage: Primary abstract/search extract; v3 linked.

Structured long-memory baseline. Market evaluation remains separate from sequence benchmarks.

## S30

[Orvieto et al.: Linear Recurrent Unit](https://arxiv.org/abs/2303.06349)

Coverage: Primary abstract/search extract.

Strong direct recurrence reference for matched-compute comparison; full methods reproduction queued.

## S31

[Gu, Kelly and Xiu: empirical asset pricing via ML](https://www.nber.org/papers/w25398)

Coverage: Primary abstract/search extract; revised September 2019.

Nonlinear predictor interactions are existing financial research; replicate data and validation design before importing reported results.


