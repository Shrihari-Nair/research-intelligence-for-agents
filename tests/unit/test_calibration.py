import pytest

from research_os.router.calibration import choose_threshold, sweep_thresholds


def test_sweep_coverage_and_accuracy():
    conf = [0.9, 0.8, 0.6, 0.4]
    correct = [True, True, False, True]
    s = {round(r["threshold"], 2): r for r in sweep_thresholds(conf, correct, [0.5, 0.85])}
    assert s[0.5]["coverage"] == pytest.approx(0.75)
    assert s[0.5]["covered_accuracy"] == pytest.approx(2 / 3)
    assert s[0.85]["coverage"] == pytest.approx(0.25) and s[0.85]["covered_accuracy"] == 1.0


def test_choose_smallest_threshold_meeting_target():
    sweep = [{"threshold": 0.5, "coverage": 1.0, "covered_accuracy": 0.90},
             {"threshold": 0.7, "coverage": 0.8, "covered_accuracy": 0.975},
             {"threshold": 0.9, "coverage": 0.5, "covered_accuracy": 1.0}]
    t, rule = choose_threshold(sweep, 0.97)
    assert t == 0.7 and "0.97" in rule


def test_choose_when_none_meet_target_picks_best_accuracy():
    sweep = [{"threshold": 0.5, "coverage": 1.0, "covered_accuracy": 0.90},
             {"threshold": 0.9, "coverage": 0.5, "covered_accuracy": 0.95}]
    t, rule = choose_threshold(sweep, 0.99)
    assert t == 0.9 and "no threshold" in rule.lower()


def test_empty_covered_set_has_none_accuracy():
    s = sweep_thresholds([0.2], [True], [0.9])[0]
    assert s["coverage"] == 0.0 and s["covered_accuracy"] is None
