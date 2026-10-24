from src.data.normalize import normalize_match


def make(status, home, away):
    return {
        "id": 1,
        "utcDate": "2026-10-24T15:00:00Z",
        "status": status,
        "homeTeam": {"id": 10, "name": "A"},
        "awayTeam": {"id": 20, "name": "B"},
        "score": {"fullTime": {"home": home, "away": away}},
    }


def test_scheduled_match_has_no_goals_not_zero():
    m = normalize_match(make("SCHEDULED", None, None), "PL")
    assert m["home_goals"] is None and m["away_goals"] is None
    assert m["is_final"] is False


def test_finished_match_keeps_goals():
    m = normalize_match(make("FINISHED", 2, 1), "PL")
    assert (m["home_goals"], m["away_goals"], m["is_final"]) == (2, 1, True)


def test_live_match_partial_score_is_ignored():
    m = normalize_match(make("IN_PLAY", 1, 0), "PL")
    assert m["home_goals"] is None and m["is_final"] is False


def test_missing_score_block_does_not_crash():
    m = normalize_match({"id": 5, "status": "SCHEDULED"}, "CL")
    assert m["competition_code"] == "CL" and m["home_goals"] is None
