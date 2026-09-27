from datetime import date
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research.market_correction import FEATURES, annotate, corrected_probability, fit_correction, logit
from research.market_check import verify_model_directory


class MarketCorrectionTests(unittest.TestCase):
    def frame(self):
        rows = []
        for game in range(1, 9):
            p, fair = .3 + .05 * game, .4 + .025 * game
            for market, sign in ((0, 1), (1, -1)):
                rows.append(dict(game_pk=game, game_date=date(2026, 8, 12 + game),
                    market=market, time=100, mid=p if market == 0 else 1-p,
                    fair_gap=(fair-p)*sign, inning=3, entry_allowed=True,
                    outcome=int(game % 2 == 0) if market == 0 else int(game % 2 != 0)))
        return pd.DataFrame(rows)

    def anchors(self):
        return pd.DataFrame(dict(game_pk=list(range(1, 9)), offset=.15, available_time=50)).set_index("game_pk")

    def test_team_probabilities_and_features_are_complementary(self):
        frame = annotate(self.frame(), self.anchors())
        for _, part in frame.groupby("game_pk"):
            self.assertAlmostEqual(part.anchor_probability.sum(), 1.)
            np.testing.assert_allclose(part[FEATURES].sum(axis=0), 0., atol=1e-12)
        model = fit_correction(frame)
        p = corrected_probability(frame, model)
        np.testing.assert_allclose(p[::2] + p[1::2], 1.)

    def test_anchor_waits_for_observation_delay(self):
        anchors = self.anchors()
        anchors.loc[1, "available_time"] = 100
        frame = annotate(self.frame(), anchors)
        self.assertTrue(frame.loc[frame.game_pk.eq(1), "anchor_probability"].isna().all())

    def test_initial_model_state_maps_to_observed_pregame_probability(self):
        raw = self.frame()
        raw["fair_gap"] = np.where(raw.market.eq(0), .6, .4) - raw.mid
        anchors = self.anchors()
        anchors["offset"] = logit(.7) - logit(.6)
        frame = annotate(raw, anchors)
        np.testing.assert_allclose(frame.loc[frame.market.eq(0), "anchor_probability"], .7)

    def test_future_outcomes_cannot_change_fitted_coefficients_or_scaling(self):
        frame = annotate(self.frame(), self.anchors())
        later = frame.copy()
        later["game_date"] = date(2026, 9, 20)
        combined = pd.concat([frame, later], ignore_index=True)
        model = fit_correction(combined)
        mask = combined.game_date > date(2026, 9, 1)
        combined.loc[mask, "outcome"] = 1 - combined.loc[mask, "outcome"]
        combined.loc[mask, FEATURES] *= 100
        self.assertEqual(model, fit_correction(combined))

    def test_fill_outcomes_are_not_correction_features(self):
        frame = annotate(self.frame(), self.anchors())
        before = fit_correction(frame)
        frame["buy_time"] = 999999
        frame["buy_price"] = .01
        self.assertEqual(before, fit_correction(frame))

    def test_alternate_model_directory_cannot_bypass_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = {}
            for name in ("model.cbm", "prior.json"):
                (root / name).write_bytes(b"frozen")
                hashes["original/" + name] = hashlib.sha256(b"frozen").hexdigest()
            verify_model_directory(root, {"hashes": hashes})
            (root / "model.cbm").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "differs from the frozen model"):
                verify_model_directory(root, {"hashes": hashes})


if __name__ == "__main__":
    unittest.main()
