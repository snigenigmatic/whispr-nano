import unittest
from research.bridge_open.qwen_trace import token_termination, validate_trace

class TerminationTest(unittest.TestCase):
    def test_batch_padding_does_not_make_finished_row_a_cap(self):
        self.assertEqual(token_termination([42, 99, 0, 0], [99], 4),
                         dict(new_token_counts=2, eos_present=True, cap_observed=False))
        self.assertEqual(token_termination([42, 43, 44, 45], [99], 4),
                         dict(new_token_counts=4, eos_present=False, cap_observed=True))

    def test_corrupted_termination_or_token_mapping_is_rejected(self):
        cfg = dict(eos_token_ids=[99], max_new_tokens=4)
        t = dict(trace_schema=2, observation_status='captured', raw_sequences=[10, 42, 99, 0, 0],
                 prompt_tokens=1, new_token_sequences=[42, 99, 0, 0], raw_decoded_text='hello',
                 max_new_tokens=4, **token_termination([42, 99, 0, 0], [99], 4))
        self.assertEqual(validate_trace([t], cfg), [])
        self.assertTrue(validate_trace([dict(t, cap_observed=True)], cfg))
        self.assertTrue(validate_trace([dict(t, prompt_tokens=2)], cfg))
        self.assertTrue(validate_trace([], cfg))
