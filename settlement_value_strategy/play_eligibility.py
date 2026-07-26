"""Shared live/backtest rules for consuming an MLB play atomically."""

from __future__ import annotations


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
