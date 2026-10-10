# Verification record

Executed on 10 October 2026.

- Offline CLI completed all seven tasks on 81 synthetic tickets, with zero provider calls.
- All 16 offline unittest cases passed. `test_results.txt` contains the captured run. Mocked SDK tests do not verify the real OpenAI service.
- 20 calibration rows, four per category, include ambiguous and very short text. Accuracy is 16/20 category, 19/20 sentiment, 15/20 urgency; strict category criterion is not passed.
- Ten consistency pairs use 20 separate extractor invocations. Demo agrees on all10; high cohort4, low cohort6. This is rule repeatability, not LLM repeatability.
- Master 81 = review15 + BI subset66. IDs, schema enums, month values and review flags reconcile.
- All eight notebook code cells executed sequentially in a shared Python process with captured stream/table outputs; notebook structure validated with nbformat. Native Jupyter kernel execution was attempted but blocked by environment socket restrictions. No native-kernel success is claimed.
- PDF contains six pages; every page was rendered and visually reviewed for wrapping, table/chart legibility, and clipping.
- Self-contained HTML was checked for JavaScript syntax and filter behavior using a simulated DOM; no native browser visual verification was performed.
- Actual instructor-data validation, live traces, real token costs, and GitHub publication remain pending.

## Tested runtime

Python 3.12.14

- openai 3.28.0
- langsmith 0.14.7
- python-dotenv 1.2.4
- pandas 2.2.3
- nbformat 5.11.1
- nbclient 0.11.0
- matplotlib 3.10.8
