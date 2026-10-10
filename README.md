# Data Enhancing Extra | Lab solution package

**Status: executable solution with an offline demonstration.** All seven tasks have implemented code. The included `tickets_structured.csv` contains **synthetic demo data extracted by rules**, not instructor data or OpenAI results. Neither required course CSV was supplied. Live extraction, instructor-ground-truth calibration, LangSmith trace verification, and GitHub submission remain to be completed. Do not submit the demo CSV as if it met those requirements.

## Quick start: offline demo

Use Python 3.11 or newer. From the extracted repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run_lab --mode demo
python -m unittest discover -s tests -v
```

No keys are needed for demo mode. It disables tracing and makes zero OpenAI calls. The CLI prints the exploration, three single-ticket examples, progress every ten rows, batch counts, calibration, consistency, and export summary. The notebook in `notebooks/` implements the same seven steps with explanatory cells. Start Jupyter using your normal notebook environment and open it; running all cells defaults to demo mode. Live mode is an explicit opt-in.

## Run the actual lab

1. Obtain `support_tickets.csv` and `support_tickets_labeled.csv` from the instructor and place them in `data/instructor/`.
2. Raw CSV columns must be `ticket_id`, `submitted_at`, `raw_text`; exactly 81 nonempty tickets, unique IDs, valid timestamps. Rename a course text column explicitly if it has another name.
3. The labelled CSV must have matching `ticket_id` values and human labels `category`, `sentiment`, `urgency`. Aliases `true_category`/`category_true` (and the corresponding sentiment/urgency names) are accepted when the canonical column is absent. Do not use model predictions as labels.
4. Mark ambiguous human-reviewed rows with an `is_ambiguous` column (true/false), or supply their IDs using `--ambiguity-ids ID1,ID2`. The calibration selector takes at least three per category, then fills to 20 in category order. It prioritizes very short and ambiguous texts without consulting prediction correctness. The supplied demo has four per category. If the data cannot supply three per category or ambiguity markers, it fails before provider calls.
5. Copy `.env.example` to `.env` and fill both API keys. The supplied endpoint is the EU endpoint from the brief; use the endpoint appropriate to your LangSmith account. Keep the project `m5-ai-data-enhancing`. The default model is `gpt-4o-mini`; set `OPENAI_MODEL` if your account uses another model supporting strict JSON Schema. Optional cost rates must be current USD per million input/output tokens. Blank rates leave live cost unknown.

```bash
cp .env.example .env
# Edit .env locally; never commit it.
python -m src.run_lab --mode live
# If interrupted, reuse matching successful records and retry failed ones:
python -m src.run_lab --mode live --resume --retry-failed
# Alternative input files and explicit ambiguous examples:
python -m src.run_lab --mode live --input data/instructor/support_tickets.csv --labels data/instructor/support_tickets_labeled.csv --ambiguity-ids ID1,ID2
```

A fresh run makes 81 batch invocations, 3 single-ticket tests, and 20 fresh consistency invocations: **104 logical invocations before retries**. Calibration reuses batch predictions. Transient failures have at most three attempts per invocation; nontransient errors are not retried. Resuming only caches batch records: the single-ticket and consistency tests run afresh and incur additional calls. Use a separate output folder for a revised prompt/model. Checkpoint fingerprints reject stale prompt/schema/model data and changed ticket contents. A crash-truncated JSONL line must be inspected before resuming; do not silently discard it.

Live runs update root `tickets_structured.csv` and `lab_summary.md`, and write audit exports in `outputs/live/`. The packaged PDF/HTML report documents the original **demo run** and does not update automatically. Use new live CSVs, metrics, summary, and trace evidence as submission evidence. Notebook: set `MODE='live'` and `RUN_LIVE=True`; configure the two instructor paths and ambiguity markers. Run from the repository root or notebook folder.

## Validation and acceptance

- Category accuracy is correct predictions divided by all 20 calibration tickets. Missing/failed predictions count as wrong. The strict criterion **>80%** means **17/20 or more**, not 16/20.
- Sentiment and urgency have their own accuracies. Inspect `category_confusion_matrix.csv` and `calibration_comparisons.csv` for errors and disputed labels. Improve the prompt on reviewed development examples, then use a fresh held-out set before claiming generalization; 20 balanced examples do not estimate the real category mixture.
- Consistency tests choose two tickets per category, aiming for one ambiguous and one ordinary example. Each receives two fresh invocations, independent of checkpoint reuse. Both successful and high-confidence means the high cohort; any low confidence means the low cohort. Failed pairs are reported separately and count as disagreement in the all-pairs rate; empty cohorts produce null, not 100%.
- `confidence` is a qualitative review flag. A repeatable or high-confidence prediction can still be wrong. The demo's 100% consistency is expected from rules and is not evidence of LLM stability.
- After a live run open the LangSmith project. Verify one successful chain and its OpenAI child have the correct ticket input, five-field output, model, token usage, latency, and (where available) cost. Verify an uncertain example too. Save permitted screenshots and run links under `docs/live_trace_evidence/`. Record the actual verification in `docs/live_validation.md`; do not merely change a status flag. Tracing is implemented but has not been verified live in this package.
- An API error, refusal, truncated response, invalid JSON, missing/extra key, invalid enum, or empty topic produces all five fields as `parse_error` plus diagnostic metadata. Batch processing preserves every ticket and continues.

## BI handoff

Connect Tableau or Power BI to `tickets_structured.csv`. Use `ticket_id` as text, `submitted_at` as datetime, `month` as text `YYYY-MM`, and `review_required`/`bi_eligible` as booleans. For routine category/sentiment/urgency charts, **filter `bi_eligible = true`** or use `tickets_bi_ready.csv`. The separate review file keeps uncertain/failed rows visible to operations. Do not plot `parse_error` as a genuine category. Display total tickets and excluded review tickets as separate counts; excluding rows can bias apparent customer sentiment. Topics belong in a searchable table, tooltip, or drilldown, not a sum or average. See the data dictionary and report.

## File map

| Path | Purpose |
| --- | --- |
| `src/pipeline.py` | Tasks 1–7: strict schema, traced OpenAI extractor, demo adapter, resilient batch, calibration, consistency, BI export |
| `src/run_lab.py` | CLI orchestrator and automatic root submission exports |
| `src/make_demo.py` | Reproduce the 81 fictional tickets and human-authored demo labels |
| `notebooks/data_enhancing_solution.ipynb` | Executed offline notebook with all seven tasks and reflections |
| `prompts/extraction_prompt.txt`, `prompts/ticket_schema.json` | Reviewable copies of the prompt and schema |
| `data/demo/` | Synthetic raw tickets and synthetic labels, explicitly separate |
| `data/instructor/` | Placeholder for missing course files |
| `tickets_structured.csv` | Required export filename; currently demo, regenerated by a live run |
| `lab_summary.md` | Required one short paragraph; currently accurately describes demo limitations |
| `outputs/demo/` | Master CSV, BI subset, review queue, calibration set/comparisons, confusion matrix, independent consistency pairs, metrics, checkpoint, sample tests |
| `docs/data_dictionary.md` | CSV column definitions and BI rules |
| `docs/solution_notes.md` | Requirement map, reasoning, reflection answers, cost model |
| `docs/submission_checklist.md` | Course acceptance checks and GitHub-only submission steps |
| `docs/live_validation.md` | Honest live-validation record to complete after the real run |
| `docs/verification.md` | Offline execution and test evidence |
| `data_enhancing_solution_report.pdf` | Six-page readable demo report with assumptions and next steps |
| `data_enhancing_demo_report.html` | Offline, searchable demo ticket report |
| `figures/` | Exact demo confusion matrix and category-count plots |
| `tests/test_pipeline.py` | Offline tests covering failure handling, denominators, checkpoint safety and fresh consistency calls |
| `.env.example`, `.gitignore`, `requirements.txt` | Configuration, secrets exclusions, dependencies |

## Submit on GitHub

Create a dedicated repository containing only this lab. After the actual run and validation, confirm root CSV/summary are live results, add permitted evidence and exports, and submit the **repository URL**, as the brief requires. This package does not create or publish a GitHub repository. Instructor CSVs are ignored by default; include them only if course sharing permission allows it. Check that no `.env`, credentials, confidential tickets, or unrestricted traces are committed.

## Sources

- Supplied brief: `LAB | Data Enhancing Extra.pdf` (requirements; not redistributed).
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat)
- [LangSmith: trace OpenAI](https://docs.langchain.com/langsmith/trace-openai)
- [LangSmith: annotate code with tracing](https://docs.langchain.com/langsmith/annotate-code)
- [LangSmith observability quickstart](https://docs.langchain.com/langsmith/observability-quickstart)

Documentation checked 10 October 2026. Model, rates, access, and service behavior can change; confirm your account settings before the live run.
