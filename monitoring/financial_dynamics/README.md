# Experiment observability

These are offline-validated schemas and a candidate collector configuration. No Elasticsearch, Kibana or Datadog service is deployed by this directory. The experiment runner creates import files; it does not make network calls.

## Elasticsearch and Kibana

1. Provision an authenticated TLS service with compatible, pinned Elasticsearch/Kibana versions, restricted network access and separate writer/reader identities. Set retention and restore policy before collecting data. See [official setup source S20](../../research/financial_dynamics/SOURCES.md#s20).
2. Install `elasticsearch-index-template.json` through the authenticated `PUT /_index_template/financial-dynamics-v1` API before creating the first matching index. The strict mapping accepts only the current experiment-event fields. Evolve the schema explicitly when adding fields.
3. Submit a run's `elasticsearch.bulk.ndjson` to `POST /_bulk` as `application/x-ndjson`. Inspect every item in the response: HTTP 200 alone is not success. Retry transient item failures with bounded backoff. Treat a 409 from `create` as an expected replay only after confirming that the existing document has the expected content. Quarantine mapping or permission errors; never discard them silently.
4. Create a Kibana data view `financial-dynamics-*`, time field `@timestamp`. Build panels grouped by scenario/model/seed, including net return, maximum drawdown, MSE, turnover, costs and trial count. Filter `data_kind=synthetic` and `evaluation=development_walk_forward` visibly. Do not combine different annualisation conventions or sum return percentages.
5. Verify 45 imported documents for the default run, including nullable Sharpe/MSE, then replay the same file and confirm no new documents. Test broken credentials, malformed events, partial batch failure, retention and restore. These server-side tests remain open.

`@timestamp` is the experiment execution timestamp, not the date of an underlying market observation. Synthetic equity is normalised to initial capital 1; `costs` are in those units, and `cloud_spend_usd` is a separate operational measure. The event hash is a deterministic ID for replay of identical artifacts. A re-execution at a new timestamp is a distinct event. `run_id` currently identifies code/configuration, not a globally unique dispatch; production also needs a unique execution ID.

The bulk import file contains the exact event documents. SQL remains authoritative for job/payment state; verified object-store artifacts remain authoritative for results. Elasticsearch is a query projection.

## Datadog and OpenTelemetry

`otel-collector.example.yaml` follows the reviewed official Datadog OTLP/HTTP route ([S19](../../research/financial_dynamics/SOURCES.md#s19)). It is a template, not a tested collector deployment. Select and pin a build that includes `otlp_http`; run that build's configuration validator before starting it. Inject the actual `DD_SITE` and narrowly scoped `DD_API_KEY` at runtime through Credential Manager.

The example receives only loopback traffic. Container/network deployment requires an explicit listener and access policy; changing to a public listener without authentication is not an acceptable shortcut. The queue is memory-backed and can lose telemetry on restart; use verified persistent queues where operational requirements demand them. Research artifacts must not depend on this queue for durability.

The current Python runner emits a Prometheus-format file with two run counters; it has no OTLP instrumentation. The collector will not ingest that file automatically. Implement and test the bridge or an OTLP producer as part of deployment. Keep run IDs in traces/logs; use bounded metric labels such as service, state and model family. Do not export provider tokens, bank details, card data, raw user documents or unrestricted prompts.

Required first alerts: lease deadline exceeded, unresolved allocation, teardown unconfirmed, artifact export failed, budget exhausted and unexpected holdout access. The full operational metrics and failure drills are defined in [EXPERIMENT_INFRA.md](../../research/financial_dynamics/EXPERIMENT_INFRA.md).
