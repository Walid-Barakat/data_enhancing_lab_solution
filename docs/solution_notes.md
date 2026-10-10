# Solution notes

## Requirements mapped to implemented work

| Task | Implementation | Evidence/output |
| --- | --- | --- |
| 1 | `read_tickets`, `characterize` | 81 fictional tickets, 2025-01-01–2025-12-31; 5 fixed-seed full samples; 3–84 characters |
| 2 | `PROMPT`, `SCHEMA` | Task/input/criteria/output, exact enums, dominant intent, unknowns and low confidence; prompt/schema files |
| 3 | `OpenAIExtractor.extract_ticket` and `.extract_record` | `wrap_openai`, `@traceable`, strict JSON Schema, all-field failure sentinel; three sample tests; live verification pending |
| 4 | `batch` | Row-preserving loop, progress every 10, bounded retries, flushed JSONL checkpoint, success/failure summary |
| 5 | `calibration_set`, `evaluate` | 20 human-labelled examples, >=3 per category, ambiguity, three accuracies, confusion matrix; instructor labels pending |
| 6 | `consistency` | 10 tickets x 2 fresh invocations, no checkpoint reuse; high/low cohorts and failed pairs |
| 7 | `export_data` | Original fields, five schema fields, `month`, review/BI booleans, metadata, master plus review and BI CSVs |

## What can be extracted, and what cannot?

Category, expressed sentiment, a conservative urgency estimate, and a factual issue summary can often be derived from text. Confidence records ambiguity in that derivation. The ticket text cannot establish actual business priority, transaction value, resolution time, lifetime value, a verified root cause, or whether a customer claim is true. The unknown ticket must not invent such facts. Month is derived deterministically from the timestamp, not guessed by the model. A billing category is not automatically negative sentiment, except the explicit lab duplicate-charge convention.

## Demo interpretation

The 81 rows are author-created examples for a fictional energy analytics application. Labels are authored separately; the rule extractor only reads raw text and never consults those labels. The 20-ticket balanced calibration deliberately includes ambiguous examples. Demo category accuracy is 16/20 (80%), sentiment 19/20 (95%), urgency 15/20 (75%). The strict category target is not passed. Feature suggestions that mention invoices and praise referring to resolved login problems illustrate keyword-rule mistakes. High confidence does not detect every error.

There are 81 successful schema-valid outputs, 15 review rows and 66 BI-eligible rows. 'Successful' describes schema/transport processing, not semantic correctness. Two fresh offline computations agree on all 10 consistency pairs: 4 high and 6 low. Rule determinism explains this result; it cannot establish LLM repeatability. No provider calls, tokens, provider cost, or live traces were produced.

## Robustness decisions

The LLM receives ticket text as a user message; the system prompt tells it to treat embedded instructions as data. Strict JSON Schema constrains syntax and enums but does not guarantee accurate interpretation or defeat every adversarial input. Local validation checks the actual returned object. Refusals, truncated completions, parse/schema errors and API failures are classified separately outside the five-field lab schema. All unsuccessful records carry `parse_error` in each of the five fields, preserving the lab's error convention and ensuring BI exclusion.

Checkpoint rows include the original-data hash and prompt/schema/mode/model fingerprint. Resume rejects changed data and stale configuration. A failed row can be retried explicitly; latest checkpoint entries take precedence for that ID. Consistency calls deliberately bypass all batch caching. JSONL flush + fsync protects completed rows if a later call crashes. The schema is source-defined, not accepted from untrusted tickets.

## Calibration and confidence

Balanced calibration is useful for detecting per-category problems, but its aggregate accuracy is not necessarily expected production accuracy because the category distribution differs. Ambiguous human labels should be independently adjudicated; preserve both annotator labels and the final decision. Revisions evaluated on the same 20 tickets risk overfitting. Hold out additional labelled tickets for final validation. Confidence is self-reported qualitative evidence, not a numerical error rate; estimate actual error rates within confidence cohorts before relying on it for routing.

## Reflection answers

1. **Understanding versus pattern matching.** This run uses explicit rules. A live LLM can use linguistic context, negation and mixed intents more effectively, but output plausibility alone does not prove understanding. Compare against human labels and inspect novel paraphrases and counterexamples.
2. **Who defines truth?** The demo author defined synthetic labels. Only the instructor's human labels establish the course reference. Two humans may disagree on dominant intent, implied negative sentiment, or urgency; an annotation guide and adjudication make these disagreements visible.
3. **Production review.** Route low confidence and every failure to a queue with the original text, proposed labels and error type. A human corrects or approves labels before BI inclusion, with an audit record. Reconcile included + excluded counts, sample high-confidence rows, and monitor error drift. Do not automatically retry semantic ambiguity until it happens to become high confidence.
4. **Economics.** Provider token cost is only part of total cost. Include reviewer time, rework, infrastructure, and setup. Let `N` be tickets, `F` fixed pipeline setup cost, `a` provider/infrastructure cost per ticket, `r` review fraction, `h` full human annotation cost per ticket, and `q` expected correction cost per ticket. Hybrid cost is `F + N(a + r*h + q)`; manual cost is `N*h`. Break-even is `N > F / (h - a - r*h - q)` only if the denominator is positive. If it is zero or negative, there is no scale where this hybrid is cheaper under these assumptions. Measure throughput and actual trace usage; do not invent a token price or wage. A fully failed API call can have unknown usage, so 'observed tokens' is not always total billed tokens.
5. **Topic in BI.** Topic is narrative text. Use it for row-level drilldowns, searchable tables, tooltips, or later human-reviewed issue clustering. Do not average it. Categories and monthly counts supply chart dimensions/measures; summaries preserve context.

## Suggested dashboard

Use a category bar chart, monthly ticket trend, and category-by-sentiment view filtered to BI-eligible records. Show review queue size and failures as separate operational measures. Add a searchable table with ticket ID, date, category, urgency and topic. Synthetic dataset dates and ticket mix are artificial; draw no real business conclusions from the demo visuals.
