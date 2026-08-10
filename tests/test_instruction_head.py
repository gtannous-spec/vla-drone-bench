"""Tests for InstructionConditionedHead (FiLM + Attention Pooling)."""

import torch
import pytest

from airsim_benchmark.training.instruction_head import InstructionConditionedHead

HIDDEN_DIM = 4096
ACTION_DIM = 8
FILM_DIM = 256


@pytest.fixture
def head():
    model = InstructionConditionedHead(HIDDEN_DIM, ACTION_DIM, FILM_DIM)
    model.eval()
    return model


def test_output_shape(head):
    """Input [1, 10, 4096] + [1, 4096] → output [1, 8]."""
    instr_hidden = torch.randn(1, 10, HIDDEN_DIM)
    action_vec = torch.randn(1, HIDDEN_DIM)
    out = head(instr_hidden, action_vec)
    assert out.shape == (1, ACTION_DIM)


def test_output_range(head):
    """All outputs in [-1, 1] due to Tanh."""
    instr_hidden = torch.randn(4, 10, HIDDEN_DIM)
    action_vec = torch.randn(4, HIDDEN_DIM)
    out = head(instr_hidden, action_vec)
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0


def test_different_instructions_different_output(head):
    """Same action_vec + different instr_hidden → different outputs.

    This is the KEY test: verifies FiLM actually conditions on instruction.
    """
    torch.manual_seed(42)
    action_vec = torch.randn(1, HIDDEN_DIM)
    instr_a = torch.randn(1, 10, HIDDEN_DIM)
    instr_b = torch.randn(1, 10, HIDDEN_DIM) * 3.0 + 1.0  # clearly different

    out_a = head(instr_a, action_vec)
    out_b = head(instr_b, action_vec)

    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-4, (
        f"Outputs should differ when instructions differ, but max diff={diff}"
    )


def test_single_instruction_token(head):
    """Works with [1, 1, 4096] — single instruction token."""
    instr_hidden = torch.randn(1, 1, HIDDEN_DIM)
    action_vec = torch.randn(1, HIDDEN_DIM)
    out = head(instr_hidden, action_vec)
    assert out.shape == (1, ACTION_DIM)


def test_attention_weights_sum_to_one(head):
    """Attention weights sum to 1.0 along the token dimension."""
    instr_hidden = torch.randn(2, 15, HIDDEN_DIM)
    weights = head._compute_attention_weights(instr_hidden)

    assert weights.shape == (2, 15, 1)
    sums = weights.sum(dim=1)  # [B, 1]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_batch_independence(head):
    """Each batch element is processed independently."""
    torch.manual_seed(0)
    instr_hidden = torch.randn(3, 10, HIDDEN_DIM)
    action_vec = torch.randn(3, HIDDEN_DIM)

    out_batch = head(instr_hidden, action_vec)
    out_single = head(instr_hidden[1:2], action_vec[1:2])

    assert torch.allclose(out_batch[1], out_single[0], atol=1e-5)


def test_film_generator_detectable():
    """State dict contains 'film_generator.weight' for head detection."""
    head = InstructionConditionedHead(HIDDEN_DIM, ACTION_DIM, FILM_DIM)
    sd = head.state_dict()
    assert "film_generator.weight" in sd
    assert "film_generator.bias" in sd
