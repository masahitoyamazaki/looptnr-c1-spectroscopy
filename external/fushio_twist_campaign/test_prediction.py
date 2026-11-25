"""Check the frozen decision on synthetic response data, never production data."""
import json
import tempfile
import unittest
from pathlib import Path
import toml
import campaign


class PredictionDecision(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_home = campaign.H
        campaign.H = Path(self.tmp.name)
        (campaign.H / 'data').mkdir()
        (campaign.H / 'predictions').mkdir()
        (campaign.H / 'protocol.json').write_text('{}')
        self.g = dict(id='synthetic', calibration_step=24, epsilon=.001,
                      calibration_ids=['zero'] + [f'{s}{j}' for j in range(3) for s in ['plus', 'minus']])

    def tearDown(self):
        campaign.H = self.old_home
        self.tmp.cleanup()

    def inputs(self, offset=.002, quadratic=0., cubic=0., missing=False, numerical=True):
        for label in self.g['calibration_ids']:
            h = 0. if label == 'zero' else (.001 / 2**int(label[-1])) * (1 if label.startswith('plus') else -1)
            delta = offset + 20*h + quadratic*h*h + cubic*h**3
            d = dict(status='planned_extent_complete', config=dict(label=label),
                     measurements=[dict(step=24, numerical_gate=numerical, paper_X=.01)],
                     twist_measurements=[dict(step=24, numerical_gate=True, splitting_x=delta)])
            if missing and label == 'plus2':
                d['twist_measurements'] = []
            (campaign.H / 'data' / (label+'.toml')).write_text(toml.dumps(d))

    def test_linear_prediction_and_no_retuning(self):
        self.inputs()
        r = campaign.make_prediction(self.g)
        self.assertEqual(r['status'], 'frozen')
        self.assertAlmostEqual(r['input_h'], -.0001)
        before = (campaign.H / 'predictions/synthetic.json').read_bytes()
        self.inputs(offset=.004)
        self.assertEqual(campaign.make_prediction(self.g), r)
        self.assertEqual((campaign.H / 'predictions/synthetic.json').read_bytes(), before)

    def test_missing_point_is_held(self):
        self.inputs(missing=True)
        self.assertEqual(campaign.make_prediction(self.g)['status'], 'held')

    def test_failed_numerical_gate_is_held(self):
        self.inputs(numerical=False)
        self.assertEqual(campaign.make_prediction(self.g)['status'], 'held')

    def test_even_response_is_held(self):
        self.inputs(quadratic=5000)
        self.assertIn('even response exceeds 5% of odd response', campaign.make_prediction(self.g)['reasons'])

    def test_nonlinear_slope_is_held(self):
        self.inputs(cubic=1e7)
        self.assertIn('epsilon/2 and epsilon/4 slopes differ by >5%', campaign.make_prediction(self.g)['reasons'])

    def test_extrapolation_is_held(self):
        self.inputs(offset=.05)
        self.assertIn('predicted input outside calibrated range', campaign.make_prediction(self.g)['reasons'])


if __name__ == '__main__':
    unittest.main()
