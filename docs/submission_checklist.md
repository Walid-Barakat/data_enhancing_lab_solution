# Submission checklist

## Before claiming completion

- [ ] Replace synthetic inputs with the two instructor CSVs; verify 81 ticket IDs and valid dates.
- [ ] Verify human labels and identify ambiguous examples before running the extractor.
- [ ] Run live mode with OpenAI and LangSmith credentials.
- [ ] Confirm 81 actual successful extractions; if any fail, inspect and retry explicitly.
- [ ] Confirm a balanced 20-ticket calibration with >=3 examples/category and ambiguous examples.
- [ ] Category accuracy is strictly above 80%: at least 17 correct of 20. Report sentiment and urgency separately.
- [ ] Inspect category errors; adjudicate label disagreements rather than silently changing labels to match the model.
- [ ] Confirm the 10-ticket consistency file contains two fresh live invocations/ticket and both confidence-cohort denominators.
- [ ] Inspect real LangSmith chains and child calls, input/output, token usage, latency, and cost. Add permitted evidence and run links.
- [ ] Root `tickets_structured.csv` contains actual instructor-input results, valid month values and review flags.
- [ ] Review queue and BI subset reconcile exactly with the master; failed/low-confidence rows do not silently enter charts.
- [ ] Root `lab_summary.md` has one short paragraph with actual calibration, main challenge, next improvement.
- [ ] README has run instructions and an accurate file map; packaged demo report remains clearly identified as demo evidence.
- [ ] Create a dedicated lab-only GitHub repository; review files for secrets and sharing permissions before push.
- [ ] Submit the GitHub repository URL, not this ZIP/download link, as the course submission.

## Minimal repository commands (run yourself after validation)

```bash
git init
git add README.md requirements.txt .gitignore .env.example src tests prompts notebooks docs figures data/demo outputs/demo tickets_structured.csv lab_summary.md data_enhancing_solution_report.pdf data_enhancing_demo_report.html
# Review staged contents and add permitted live export/evidence files explicitly.
git diff --cached --stat
git commit -m "Complete Data Enhancing Extra lab"
# Create an empty dedicated GitHub repository, then use its actual URL:
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git branch -M main
git push -u origin main
```

Do not copy placeholder URLs or invent a submitted repository. Course source CSVs are ignored by default; upload them only when permitted.
