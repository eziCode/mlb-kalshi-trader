from datetime import date
import unittest

import pandas as pd

from research.state_refresh import labels_from_feed, post_features, prior_history, raw_features


class StateRefreshTests(unittest.TestCase):
    def test_walkoff_label_uses_completed_game_score(self):
        feed = {"gameData": {"status": {"abstractGameState": "Final"}}, "liveData": {
            "linescore": {"teams": {"home": {"runs": 4}, "away": {"runs": 3}}},
            "plays": {"allPlays": [{"about": {"endTime": "2026-04-01T23:00:00Z"}}]}}}
        self.assertEqual(labels_from_feed(feed)["home_win"], 1)
        feed["gameData"]["status"]["abstractGameState"] = "Live"
        self.assertIsNone(labels_from_feed(feed))

    def test_resumed_game_is_available_after_its_last_play(self):
        feed = {"gameData": {"status": {"abstractGameState": "Final"}}, "liveData": {
            "linescore": {"teams": {"home": {"runs": 4}, "away": {"runs": 3}}},
            "plays": {"allPlays": [{"about": {"endTime": stamp}} for stamp in
                      ("2026-04-01T23:00:00Z", "2026-08-01T17:00:00Z")]}}}
        self.assertEqual(labels_from_feed(feed)["available_date"], date(2026, 8, 1))

    def games(self):
        return pd.DataFrame([dict(game_pk=pk, game_date=date(2026, 4, day),
            available_date=date(2026, 4, end), home_team="A", away_team="B",
            home_win=1, final_home_score=4, final_away_score=1)
            for pk, day, end in [(1, 1, 1), (2, 1, 1), (3, 2, 2), (4, 3, 5)]])

    def test_same_day_doubleheader_does_not_use_game_id_order_as_time(self):
        priors, _ = prior_history(self.games(), date(2026, 4, 4))
        p = priors.set_index("game_pk").pregame_prob
        self.assertEqual(p[1], p[2])
        self.assertGreater(p[3], p[2])

    def test_ratings_ignore_results_completed_after_cutoff(self):
        games = self.games()
        priors, state = prior_history(games, date(2026, 4, 4))
        changed = games.copy()
        changed.loc[changed.game_pk.eq(4), ["home_win", "final_home_score", "final_away_score"]] = [0, 0, 99]
        new_priors, new_state = prior_history(changed, date(2026, 4, 4))
        self.assertEqual(state, new_state)
        pd.testing.assert_frame_equal(priors, new_priors)

    def test_third_out_rolls_half_inning_without_future_pitch(self):
        row = dict(game_pk=1, inning_after=5, inning_topbot_after=0, outs_when_up_after=3,
                   score_diff_after=2, balls_after=2, strikes_after=2,
                   runner_on_first_after=1, runner_on_second_after=1, runner_on_third_after=1)
        x = post_features(pd.DataFrame([row]), {1: .6}).iloc[0]
        self.assertEqual(x.batting_team_is_home, 1)
        self.assertEqual(x.inning, 5)
        self.assertEqual(x.outs_when_up, 0)
        self.assertEqual(x.batting_score_diff, 2)
        self.assertEqual(x.balls + x.strikes + x.runner_on_first + x.runner_on_second + x.runner_on_third, 0)

    def test_extra_inning_automatic_runner_is_rule_based(self):
        row = dict(game_pk=1, inning_after=9, inning_topbot_after=1, outs_when_up_after=3,
                   score_diff_after=0, balls_after=0, strikes_after=2,
                   runner_on_first_after=0, runner_on_second_after=0, runner_on_third_after=0)
        x = post_features(pd.DataFrame([row]), {1: .6}).iloc[0]
        self.assertEqual(x.inning, 10)
        self.assertEqual(x.batting_team_is_home, 0)
        self.assertEqual(x.runner_on_second, 1)
        self.assertAlmostEqual(x.pregame_batting_prob, .4)

    def test_outcomes_are_excluded_from_training_features(self):
        frame = pd.DataFrame([dict(inning_topbot="Top", pregame_prob=.6, inning=1,
            outs_when_up=0, home_score=0, away_score=0, balls=0, strikes=0,
            on_1b=None, on_2b=None, on_3b=None, home_win=1, post_home_score=12)])
        x = raw_features(frame)
        frame.home_win, frame.post_home_score = 0, 0
        pd.testing.assert_frame_equal(x, raw_features(frame))


if __name__ == "__main__":
    unittest.main()
