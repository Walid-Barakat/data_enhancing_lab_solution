"""CLI entry point: python -m src.run_lab --mode demo|live."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
from langsmith import tracing_context
from .pipeline import (DemoExtractor, OpenAIExtractor, characterize, read_tickets,
                       read_labels, batch, calibration_set, evaluate,
                       consistency, export_data, safe_extract)
ROOT = Path(__file__).resolve().parents[1]


def run(mode='demo', input_path=None, labels_path=None, output=None, resume=False,
        retry_failed=False, ambiguity_ids=(), write_root=True):
    live = mode == 'live'
    folder = 'instructor' if live else 'demo'
    input_path = Path(input_path) if input_path else ROOT / 'data' / folder / 'support_tickets.csv'
    labels_path = Path(labels_path) if labels_path else ROOT / 'data' / folder / 'support_tickets_labeled.csv'
    output = Path(output) if output else ROOT / 'outputs' / mode
    output.mkdir(parents=True, exist_ok=True)
    # Validate inputs and the calibration set before spending provider credits.
    raw = read_tickets(input_path)
    labels = read_labels(labels_path, raw)
    cal = calibration_set(raw, labels, ambiguity_ids)
    cal.to_csv(output / 'calibration_set.csv', index=False)
    description = characterize(raw)
    extractor = OpenAIExtractor() if live else DemoExtractor()
    examples = []
    trace_flush_error = ''
    with tracing_context(enabled=live):
        for text in ["I've been charged twice this month.",
                     'The dashboard fails to render. A dark mode would also be nice.', '???']:
            record = safe_extract(extractor, text)
            examples.append({'raw_text': text, **record.fields, **record.metadata()})
            print('Single-ticket test:', examples[-1])
        structured = batch(raw, extractor, output / 'checkpoint.jsonl', resume, retry_failed)
        metrics, matrix, comparisons = evaluate(cal, structured)
        agreement, pairs = consistency(cal, extractor)
        if live:
            from langsmith import Client
            try:
                Client().flush(timeout=15)
            except Exception as exc:
                trace_flush_error = type(exc).__name__
                print('Tracing flush did not complete; verify LangSmith manually:', trace_flush_error)
    pd.DataFrame(examples).to_csv(output / 'single_ticket_tests.csv', index=False)
    matrix.to_csv(output / 'category_confusion_matrix.csv', index_label='true_category')
    comparisons.to_csv(output / 'calibration_comparisons.csv', index=False)
    pairs.to_csv(output / 'consistency_pairs.csv', index=False)
    exported = export_data(structured, output)
    metering = pd.concat([structured, pd.DataFrame(examples), pairs], ignore_index=True)
    known_cost = metering.estimated_cost_usd.notna().all()
    usage_complete = bool(metering.token_usage_complete.all())
    result = {
        'mode': mode, 'data_origin': 'instructor_input' if live else 'synthetic_demo',
        'method': extractor.model, 'fingerprint': extractor.fingerprint,
        'live_traces_manually_verified': False,
        'trace_flush_error': trace_flush_error,
        'exploration': description, 'calibration': metrics, 'consistency': agreement,
        'batch_successes': int(exported.extraction_status.eq('success').sum()),
        'batch_failures': int(exported.extraction_status.ne('success').sum()),
        'review_count': int(exported.review_required.sum()),
        'bi_eligible_count': int(exported.bi_eligible.sum()),
        'category_counts_all_rows': exported.category.value_counts().to_dict(),
        'confidence_counts_all_rows': exported.confidence.value_counts().to_dict(),
        'metering': {'provider_call_attempts_this_output': int(metering.api_calls.sum()),
                     'input_tokens_observed': int(metering.input_tokens.sum()),
                     'output_tokens_observed': int(metering.output_tokens.sum()),
                     'token_usage_complete': usage_complete,
                     'estimated_cost_usd': float(metering.estimated_cost_usd.sum()) if known_cost else None,
                     'scope': 'batch records + 3 single-ticket tests + 20 consistency invocations; '
                              'resumed batch records may include earlier calls; failed retry history may omit usage'},
        'limitations': ([] if live else ['No OpenAI calls or LangSmith traces were made.',
            'Demo labels are authored synthetic examples, not instructor ground truth.',
            'Rule-based accuracy and repeatability do not establish LLM quality.']) +
            ['Verify LangSmith project inputs/outputs, usage, and cost manually after a live run.',
             'Low confidence is a qualitative signal, not a calibrated error probability.'],
    }
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    intro = ('The offline demonstration used 81 synthetic tickets and a rule-based adapter; '
             if not live else 'The live pipeline processed the instructor-input tickets; ')
    accuracy = ', '.join(f'{name} {metrics[name+"_correct"]}/20 ({metrics[name+"_accuracy"]:.0%})'
                         for name in ['category', 'sentiment', 'urgency'])
    challenge = ('the instructor CSVs and API credentials were unavailable, so these results do not '
                 'validate the required LLM extraction' if not live else
                 'mixed-intent and low-confidence tickets required explicit review rather than automatic BI inclusion')
    summary = (f'{intro}calibration on 20 balanced labelled tickets produced {accuracy}. '
               f'The main challenge was that {challenge}. Next, '
               f'{"run with the instructor data, verify LangSmith traces, and improve the prompt on reviewed errors" if not live else "review the confusion matrix, adjudicate ambiguous human labels, and improve the prompt before extending calibration"}.\n')
    (output / 'lab_summary.md').write_text(summary, encoding='utf-8')
    if write_root:
        shutil.copy2(output / 'tickets_structured.csv', ROOT / 'tickets_structured.csv')
        shutil.copy2(output / 'lab_summary.md', ROOT / 'lab_summary.md')
    print(json.dumps(result, indent=2))
    return result, exported, cal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo')
    parser.add_argument('--input', dest='input_path')
    parser.add_argument('--labels', dest='labels_path')
    parser.add_argument('--output')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--ambiguity-ids', default='', help='Comma-separated human-reviewed ambiguous ticket IDs')
    parser.add_argument('--no-write-root', action='store_true')
    args = vars(parser.parse_args())
    args['ambiguity_ids'] = [x.strip() for x in args['ambiguity_ids'].split(',') if x.strip()]
    args['write_root'] = not args.pop('no_write_root')
    run(**args)


if __name__ == '__main__': main()
