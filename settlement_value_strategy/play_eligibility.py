"""Shared live/backtest rules for consuming an MLB play atomically."""

from __future__ import annotations


def state_from_play(play: dict) -> dict:
    """Post-play model inputs shared by live trading and historical research.

    Runner records can contain several movements for the same player. Only
    that player's final movement belongs in the post-play base occupancy.
    """
    result = play.get("result") or {}
    about = play.get("about") or {}
    count = play.get("count") or {}
    if not {"homeScore", "awayScore"}.issubset(result) or "outs" not in count:
        raise ValueError("Resolved play lacks post-play score or outs")
    runners = {}
    for index, runner in enumerate(play.get("runners") or []):
        details = runner.get("details") or {}
        identity = (details.get("runner") or {}).get("id", f"unknown-{index}")
        order = (int(details.get("playIndex", -1)), index)
        if identity not in runners or order > runners[identity][0]:
            runners[identity] = (order, runner.get("movement") or {})
    occupied = {"1B": 0, "2B": 0, "3B": 0}
    for _, movement in runners.values():
        if not movement.get("isOut") and movement.get("end") in occupied:
            occupied[movement["end"]] = 1
    return {
        "inning": int(about.get("inning") or 1),
        "inning_topbot": int(not bool(about.get("isTopInning"))),
        "outs_when_up": int(count["outs"]),
        "score_diff": int(result["homeScore"]) - int(result["awayScore"]),
        "balls": 0, "strikes": 0,
        "runner_on_first": occupied["1B"],
        "runner_on_second": occupied["2B"],
        "runner_on_third": occupied["3B"],
    }


def incomplete_ball_in_play_reason(
    play: dict, pitch_number: int,
) -> str | None:
    """Return why a ball in play is incomplete, or ``None`` when usable."""
    pitch = next((
        event for event in play.get("playEvents") or []
        if event.get("isPitch")
        and int(event.get("pitchNumber") or event.get("index") or 0)
        == int(pitch_number)
    ), None)
    if not pitch or not bool((pitch.get("details") or {}).get("isInPlay")):
        return None
    result = play.get("result") or {}
    runners = play.get("runners")
    if not bool(play.get("about", {}).get("isComplete")):
        return "play is not complete"
    if not result.get("eventType"):
        return "result.eventType is missing"
    if not isinstance(runners, list) or not runners:
        return "play-specific runners are missing"
    if "homeScore" not in result or "awayScore" not in result:
        return "result score is missing"
    return None


def pregame_probability_from_rating_state(
    rating_state: dict, home_code: str, away_code: str,
) -> float:
    aliases = {
        "ARI": "AZ", "CHW": "CWS", "OAK": "ATH", "KCR": "KC",
        "SDP": "SD", "SFG": "SF", "TBR": "TB", "WAS": "WSH",
    }

    def rating(code: str) -> float:
        code = str(code).upper()
        key = code if code in rating_state["ratings"] else aliases.get(code, code)
        return float(rating_state["ratings"].get(
            key, rating_state["initial_rating"]
        ))

    difference = (
        rating(home_code) + float(rating_state["home_advantage"])
        - rating(away_code)
    )
    return 1.0 / (1.0 + 10.0 ** (-difference / 400.0))
