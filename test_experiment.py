"""Scientific correctness checks; run with python -m unittest -v."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

import experiment as exp


class ExperimentTests(unittest.TestCase):
    def test_budget_and_ranking(self):
        values = np.array([0.2, 0.8, 0.1, 0.9])
        np.testing.assert_array_equal(
            exp.selected_mask(values, 0.5), [False, True, False, True]
        )
        self.assertEqual(exp.selected_mask(values, 0).sum(), 0)
        self.assertEqual(exp.selected_mask(values, 1).sum(), 4)
        self.assertEqual(exp.selected_mask(np.array([]), 0.5).sum(), 0)

    def test_invalid_decisions_rejected(self):
        for budget in [-0.1, 1.1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                exp.selected_mask(np.array([1.0]), budget)
        for values in [[np.nan], [np.inf], [[1.0]]]:
            with self.assertRaises(ValueError):
                exp.selected_mask(np.array(values), 0.5)

    def test_ties_reproducible_and_not_file_order(self):
        values = np.ones(100)
        first = exp.selected_mask(values, 0.2)
        np.testing.assert_array_equal(first, exp.selected_mask(values, 0.2))
        self.assertFalse(first[:20].all())
        self.assertFalse(
            np.array_equal(first, exp.selected_mask(values, 0.2, seed=999))
        )

    def test_label_and_id_do_not_change_feature_groups(self):
        frame = pd.DataFrame(
            {
                "id": [1, 2],
                "label": [0, 1],
                "attack_cat": ["Normal", "DoS"],
                "x": [7.0, 7.0],
            }
        )
        self.assertEqual(exp.feature_hash(frame).nunique(), 1)

    def test_corrupt_bytes_refused_before_csv_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / exp.FILES[0]).write_text("corrupt")
            with patch.object(exp, "DATA", path):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    exp.load()

    def test_training_partitions_are_disjoint(self):
        frame = pd.DataFrame(
            {
                "id": range(100),
                "label": [0, 1] * 50,
                "attack_cat": ["Normal", "DoS"] * 50,
                "x": range(100),
            }
        )
        parts = exp.split_training(frame)
        self.assertEqual(sorted(np.concatenate(parts).tolist()), list(range(100)))
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(set(parts[i]) & set(parts[j]))
        for a, b in zip(parts, exp.split_training(frame)):
            np.testing.assert_array_equal(a, b)

    def test_training_feature_overlap_is_rejected(self):
        frame = pd.DataFrame(
            {
                "id": range(100),
                "label": [0, 1] * 50,
                "attack_cat": ["Normal", "DoS"] * 50,
                "x": [1.0] * 100,
            }
        )
        with self.assertRaisesRegex(ValueError, "Feature overlap"):
            exp.split_training(frame)

    def test_independent_preprocessor_state(self):
        template = StandardScaler()
        first, second = clone(template), clone(template)
        first.fit([[0.0], [2.0]])
        second.fit([[100.0], [200.0]])
        self.assertEqual(first.mean_[0], 1.0)
        self.assertFalse(hasattr(template, "mean_"))


if __name__ == "__main__":
    unittest.main()
