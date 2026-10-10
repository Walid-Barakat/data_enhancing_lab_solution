"""Safety and evaluation invariants, all offline; no provider calls."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from langsmith import tracing_context
from src.pipeline import *

ROOT = Path(__file__).resolve().parents[1]


class CountingExtractor(DemoExtractor):
    def __init__(self): super().__init__(); self.calls = 0
    def extract_record(self, text): self.calls += 1; return super().extract_record(text)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = read_tickets(ROOT / 'data/demo/support_tickets.csv')
        cls.labels = read_labels(ROOT / 'data/demo/support_tickets_labeled.csv', cls.raw)
        cls.cal = calibration_set(cls.raw, cls.labels)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.structured = batch(cls.raw, DemoExtractor())

    def test_unknown_and_duplicate_charge(self):
        self.assertEqual(DemoExtractor().extract_ticket('???')['confidence'], 'low')
        fields = DemoExtractor().extract_ticket("I've been charged twice this month.")
        self.assertEqual([fields[x] for x in ['category','sentiment','urgency']], ['billing','negative','high'])

    def test_schema_errors(self):
        good = DemoExtractor().extract_ticket('Hello there')
        for bad in ['not json', '[]', '{}', json.dumps({**good, 'extra': 'x'}),
                    json.dumps({**good, 'category': 'invented'}), json.dumps({**good, 'topic': ''})]:
            with self.subTest(bad=bad), self.assertRaises((ValueError,TypeError)):
                parse_response(bad)

    def test_input_ids_dates_and_blank_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            for changed in ['duplicate', 'date', 'blank_date', 'text']:
                raw = self.raw.copy()
                if changed=='duplicate': raw.loc[1,'ticket_id']=raw.loc[0,'ticket_id']
                elif changed=='date': raw.loc[0,'submitted_at']='invalid'
                elif changed=='blank_date': raw.loc[0,'submitted_at']=''
                else: raw.loc[0,'raw_text']=' '
                p=Path(tmp)/'raw.csv';raw.to_csv(p,index=False)
                with self.subTest(changed=changed), self.assertRaises(ValueError): read_tickets(p)

    def test_label_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels=self.labels.copy();labels.loc[0,'ticket_id']='wrong'
            p=Path(tmp)/'labels.csv';labels.to_csv(p,index=False)
            with self.assertRaises(ValueError): read_labels(p,self.raw)

    def test_calibration_has_twenty_and_all_categories(self):
        self.assertEqual(len(self.cal),20)
        self.assertTrue(self.cal.category.value_counts().ge(3).all())
        self.assertTrue(self.cal._ambiguous.any())
        self.assertIn('???',self.cal.raw_text.tolist())

    def test_calibration_allows_three_in_one_category(self):
        labels=self.labels.copy()
        ids=labels[labels.category.eq('billing')].ticket_id.iloc[3:]
        small_labels=labels[~labels.ticket_id.isin(ids)]
        raw=self.raw[~self.raw.ticket_id.isin(ids)]
        selected=calibration_set(raw,small_labels)
        self.assertEqual(len(selected),20)
        self.assertEqual(selected.category.eq('billing').sum(),3)

    def test_batch_failure_does_not_stop(self):
        class Broken(DemoExtractor):
            def extract_record(self,text):
                if text=='???': raise RuntimeError('fake key: do not store this')
                return super().extract_record(text)
        with contextlib.redirect_stdout(io.StringIO()): result=batch(self.raw,Broken())
        self.assertEqual(len(result),81)
        failed=result[result.extraction_status.eq('failed')].iloc[0]
        self.assertEqual(failed.error_type,'RuntimeError')
        self.assertTrue(all(failed[x]=='parse_error' for x in FIELDS))
        self.assertNotIn('fake key',result.to_csv(index=False))

    def test_failed_predictions_count_as_incorrect(self):
        pred=self.cal[['ticket_id','category','sentiment','urgency']].copy()
        pred['extraction_status']='success'
        pred.loc[0,'extraction_status']='failed'
        metrics,matrix,_=evaluate(self.cal,pred)
        self.assertEqual(metrics['category_accuracy'],19/20)
        self.assertEqual(matrix['failure'].sum(),1)
        metrics,_,_=evaluate(self.cal,pred.iloc[1:])
        self.assertEqual(metrics['category_accuracy'],19/20)

    def test_strict_threshold(self):
        pred=self.cal[['ticket_id','category','sentiment','urgency']].copy()
        pred['extraction_status']='success'
        pred.loc[:3,'extraction_status']='failed'
        m,_,_=evaluate(self.cal,pred)
        self.assertEqual(m['category_accuracy'],.8)
        self.assertFalse(m['category_above_80_percent'])
        pred.loc[3,'extraction_status']='success'
        self.assertTrue(evaluate(self.cal,pred)[0]['category_above_80_percent'])

    def test_two_independent_consistency_calls(self):
        ext=CountingExtractor();metrics,pairs=consistency(self.cal,ext)
        self.assertEqual(ext.calls,20)
        self.assertEqual(len(pairs),10)
        self.assertGreater(metrics['high_n'],0)
        self.assertGreater(metrics['low_n'],0)

    def test_failed_pairs_never_count_as_agreement(self):
        class Broken(DemoExtractor):
            def extract_record(self,text): return failure('mock_error')
        m,pairs=consistency(self.cal,Broken())
        self.assertEqual(m['failed_pairs'],10)
        self.assertEqual(m['agreement_rate_all_pairs'],0)
        self.assertIsNone(m['high_agreement'])
        self.assertFalse(pairs.agreement.any())

    def test_resume_reuses_successes_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            cp=Path(tmp)/'checkpoint.jsonl';ext=CountingExtractor()
            batch(self.raw,ext,cp);self.assertEqual(ext.calls,81)
            ext.calls=0;batch(self.raw,ext,cp,resume=True);self.assertEqual(ext.calls,0)
            changed=self.raw.copy();changed.loc[0,'raw_text']+=' changed'
            with self.assertRaises(ValueError):batch(changed,ext,cp,resume=True)
            ext.fingerprint='different'
            with self.assertRaises(ValueError):batch(self.raw,ext,cp,resume=True)

    def test_retry_failed_retries_only_failure(self):
        class OneFailure(CountingExtractor):
            def extract_record(self,text):
                self.calls+=1
                return failure('test') if text=='???' else DemoExtractor.extract_record(self,text)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            cp=Path(tmp)/'cp.jsonl';first=OneFailure();batch(self.raw,first,cp)
            ext=CountingExtractor();batch(self.raw,ext,cp,resume=True,retry_failed=True)
            self.assertEqual(ext.calls,1)

    def test_export_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out=export_data(self.structured,tmp)
            review=pd.read_csv(Path(tmp)/'tickets_review_queue.csv')
            bi=pd.read_csv(Path(tmp)/'tickets_bi_ready.csv')
            self.assertEqual(len(out),81)
            self.assertEqual(len(review)+len(bi),81)
            self.assertTrue(out.month.str.match(r'\d{4}-\d{2}').all())
            self.assertFalse(bi.review_required.any())

    def test_openai_contract_without_network(self):
        # Construct without keys, wrapping, or HTTP; exercise actual live method.
        ext=object.__new__(OpenAIExtractor);ext.model='mock-model'
        fields=DemoExtractor().extract_ticket('???')
        response=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10,completion_tokens=8),
            choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(fields),refusal=None))])
        create=Mock(return_value=response)
        ext.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tracing_context(enabled=False): record=ext.extract_record('???')
        self.assertEqual(record.status,'success')
        args=create.call_args.kwargs
        self.assertTrue(args['response_format']['json_schema']['strict'])
        self.assertEqual(args['messages'][1]['content'],'???')
        response.choices[0].message.content='bad JSON'
        with tracing_context(enabled=False): record=ext.extract_record('???')
        self.assertEqual(record.fields,ERROR_FIELDS)
        response.choices[0].message.refusal='no'
        with tracing_context(enabled=False): record=ext.extract_record('???')
        self.assertEqual(record.error_type,'refusal')

    def test_bounded_retry_and_nontransient_error(self):
        class Transient(Exception): status_code=429
        ext=object.__new__(OpenAIExtractor);ext.model='mock-model'
        create=Mock(side_effect=Transient())
        ext.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tracing_context(enabled=False),patch('src.pipeline.time.sleep'):
            record=ext.extract_record('text')
        self.assertEqual(create.call_count,3)
        self.assertEqual(record.api_calls,3)
        create.reset_mock();create.side_effect=ValueError('redacted')
        with tracing_context(enabled=False): record=ext.extract_record('text')
        self.assertEqual(create.call_count,1)
        self.assertEqual(record.error_type,'ValueError')


if __name__=='__main__': unittest.main()
