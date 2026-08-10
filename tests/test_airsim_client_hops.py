"""Source contracts for AirSim hop execute (no live AirSim)."""

from pathlib import Path

_CLIENT = (
    Path(__file__).resolve().parents[1]
    / "airsim_benchmark"
    / "core"
    / "airsim_client.py"
)


def _src() -> str:
    return _CLIENT.read_text(encoding="utf-8")


def test_hold_altitude_does_not_use_tiny_lookahead():
    """Tiny lookahead on moveToZ after teleport blocks 60s and the drone falls."""
    src = _src()
    hold = src.split("def hold_altitude")[1].split("def move_to")[0]
    assert "lookahead=0.5" not in hold
    assert "timeout_sec=5.0" in hold


def test_move_to_uses_short_lookahead_without_forward_only():
    src = _src()
    move = src.split("def move_to")[1].split("def land")[0]
    assert "lookahead=0.5" in move
    assert "ForwardOnly" not in move
    assert "MaxDegreeOfFreedom" in move
    assert "moveByVelocityAsync" in move
    assert "moveToZAsync" not in move
    assert "YawMode" in move
    assert "plan_move_to" in move
