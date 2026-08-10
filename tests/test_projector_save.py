"""Vision projector must be saved with the PEFT adapter (v12 dropped it)."""

import json
from pathlib import Path

from airsim_benchmark.core.checkpoint_resolve import adapter_saves_projector


def test_v12_best_did_not_save_projector():
    cfg = Path("data/regression_checkpoints_v12/best/adapter_config.json")
    if not cfg.is_file():
        return
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert not data.get("modules_to_save")


def test_adapter_saves_projector_false_when_null(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"modules_to_save": None})
    )
    assert adapter_saves_projector(str(tmp_path)) is False


def test_adapter_saves_projector_true_when_listed(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"modules_to_save": ["projector"]})
    )
    assert adapter_saves_projector(str(tmp_path)) is True


def test_train_regression_peft_saves_projector():
    src = (
        Path(__file__).resolve().parents[1]
        / "airsim_benchmark"
        / "training"
        / "train_regression.py"
    ).read_text(encoding="utf-8")
    assert 'modules_to_save=["projector"]' in src or "modules_to_save=['projector']" in src


def test_controller_warns_when_projector_not_in_adapter():
    src = (
        Path(__file__).resolve().parents[1]
        / "airsim_benchmark"
        / "controllers"
        / "openfly_controller.py"
    ).read_text(encoding="utf-8")
    assert "adapter_saves_projector" in src
    assert "stock OpenFly projector" in src
