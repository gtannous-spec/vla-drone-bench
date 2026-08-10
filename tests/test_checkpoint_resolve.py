"""Eval must load LoRA adapters saved next to the regression head."""

from pathlib import Path

from airsim_benchmark.core.checkpoint_resolve import resolve_lora_dir


def test_explicit_lora_path_wins():
    assert resolve_lora_dir("/ckpt/best", "/other/lora") == "/other/lora"


def test_empty_when_no_adapter(tmp_path: Path):
    (tmp_path / "regression_head.pt").write_bytes(b"x")
    assert resolve_lora_dir(str(tmp_path), "") == ""


def test_same_dir_as_head_when_adapter_config_present(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "regression_head.pt").write_bytes(b"x")
    assert resolve_lora_dir(str(tmp_path), "") == str(tmp_path)


def test_parent_dir_when_path_is_head_file(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text("{}")
    head = tmp_path / "regression_head.pt"
    head.write_bytes(b"x")
    assert resolve_lora_dir(str(head), "") == str(tmp_path)


def test_controller_uses_resolve_lora_dir():
    src = (
        Path(__file__).resolve().parents[1]
        / "airsim_benchmark"
        / "controllers"
        / "openfly_controller.py"
    ).read_text(encoding="utf-8")
    assert "resolve_lora_dir" in src
    assert "MLP loaded without merging LoRA" in src


def test_slurm_defaults_lora_path_to_regression_dir():
    src = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_airsim_vla.slurm"
    ).read_text(encoding="utf-8")
    assert 'if [[ -n "$REGRESSION_HEAD_PATH" && -z "$LORA_PATH" ]]' in src
    assert 'LORA_PATH="$REGRESSION_HEAD_PATH"' in src
