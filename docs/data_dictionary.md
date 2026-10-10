# Export data dictionary

All fields occur once per original ticket. Read the master file with a header; it is UTF-8 CSV with standard quoting. `parse_error` is a failure sentinel, never an allowed successful enum value.

| Column | Type / values | Meaning |
| --- | --- | --- |
| `ticket_id` | text, unique | Original ID; preserve as text |
| `submitted_at` | timestamp | Original timestamp; validated; UTC used to derive month |
| `raw_text` | text | Complete original ticket; handle as customer data in a live run |
| `category` | billing, technical, feature_request, praise, other | Dominant support intent |
| `sentiment` | positive, neutral, negative | Expressed attitude, with lab duplicate-charge convention |
| `urgency` | high, medium, low | Conservative text-based operational priority, not a verified SLA |
| `topic` | text | One short factual sentence; needs review for unsupported facts |
| `confidence` | high, low | Qualitative extraction certainty; not a probability |
| `extraction_status` | success, failed | Transport/schema result; success does not prove semantic correctness |
| `error_type` | text or blank | Sanitized failure class or parse/refusal/truncation classification |
| `api_calls` | integer >=0 | Provider attempts for this record; demo zero |
| `input_tokens`, `output_tokens` | integer >=0 | Observed returned usage; not necessarily all retry usage |
| `latency_seconds` | number >=0 | End-to-end live record duration, including retries; demo zero |
| `estimated_cost_usd` | number or blank | Observed usage multiplied by user-supplied current rates; blank if unknown; demo zero provider cost |
| `trace_run_id` | text or blank | LangSmith parent record trace ID when tracing active; demo blank |
| `token_usage_complete` | boolean | False when failed/retried calls may hide billable usage |
| `data_origin` | synthetic_demo, instructor_input | Provenance; input origin is user-supplied in live mode, not independently authenticated |
| `extraction_method` | text | Demo adapter version or configured OpenAI model |
| `month` | text YYYY-MM | Timestamp converted to UTC, then calendar month |
| `review_required` | boolean | True when confidence low or extraction failed |
| `bi_eligible` | boolean | Inverse of review_required; provisional eligibility, not human approval |

`tickets_structured.csv` retains all81 tickets. `tickets_review_queue.csv` contains excluded rows. `tickets_bi_ready.csv` contains rows passing the automated gate. A production workflow should retain reviewed labels, reviewer, timestamp and approval status separately. Include `data_origin` prominently in dashboards so synthetic examples cannot be confused with operational data.
