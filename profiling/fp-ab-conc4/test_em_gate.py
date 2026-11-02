import unittest

from compare_em_gate import compare


def fixture(correct=16):
    rows = [dict(sample_id=str(i), answer='D', prediction='D' if i < correct else 'A',
                 exact_match=i < correct) for i in range(59)]
    waves = [dict(valid=True, exit_code=0, quality=dict(samples=rows[start:start+4]),
                  input_hashes=[f'hash{i}' for i in range(start, min(start+4, 59))])
             for start in range(0, 59, 4)]
    return dict(complete=True, expected_samples=59, expected_waves=15, waves=waves,
                dataset_sha256='dataset', scorer_sha256='scorer', config=dict(output=512))


class EmGateTests(unittest.TestCase):
    def test_identical_passes(self):
        result = compare(fixture(), fixture())
        self.assertTrue(result['passed'])
        self.assertEqual(result['prediction_matches'], 59)

    def test_score_regression_fails(self):
        self.assertFalse(compare(fixture(), fixture(15))['passed'])

    def test_improvement_is_reported(self):
        result = compare(fixture(), fixture(17))
        self.assertTrue(result['passed'])
        self.assertEqual(result['gained_correct'], ['16'])

    def test_equal_but_bad_baseline_fails(self):
        self.assertFalse(compare(fixture(0), fixture(0))['passed'])

    def test_incomplete_fails(self):
        candidate = fixture()
        candidate['complete'] = False
        with self.assertRaises(ValueError):
            compare(fixture(), candidate)

    def test_wrong_input_fails(self):
        candidate = fixture()
        candidate['waves'][0]['input_hashes'][0] = 'different'
        with self.assertRaises(ValueError):
            compare(fixture(), candidate)

    def test_producer_failure_fails(self):
        candidate = fixture()
        candidate['waves'][0]['exit_code'] = 134
        with self.assertRaises(ValueError):
            compare(fixture(), candidate)

    def test_duplicate_fails(self):
        candidate = fixture()
        candidate['waves'][1]['quality']['samples'][0]['sample_id'] = '0'
        with self.assertRaises(ValueError):
            compare(fixture(), candidate)

    def test_changed_scorer_fails(self):
        candidate = fixture()
        candidate['scorer_sha256'] = 'different'
        with self.assertRaises(ValueError):
            compare(fixture(), candidate)

    def test_short_output_diagnostic_fails(self):
        candidate = fixture()
        candidate['diagnostic_output_tokens'] = 32
        with self.assertRaises(ValueError):
            compare(fixture(), candidate)


if __name__ == '__main__':
    unittest.main()
