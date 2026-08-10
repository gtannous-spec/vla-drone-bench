"""
instruction_head.py — FiLM-based regression head conditioned on instruction features.

Unlike ActionRegressionHead which only uses the last hidden state, this head
conditions action predictions on instruction token hidden states via FiLM
(Feature-wise Linear Modulation, Perez et al. 2018).

Forward pass:
    instr_hidden: [B, N_instr, hidden_dim] — hidden states at instruction token positions
    action_vec:   [B, hidden_dim]           — last token hidden state

    1. Attention-weighted pooling over instruction tokens → instr_vec [B, hidden_dim]
    2. FiLM conditioning: gamma/beta from instr_vec modulate action features
    3. Output head → [B, action_dim] in [-1, 1]

Detection: if "film_generator.weight" is in state_dict, it's a FiLM head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionConditionedHead(nn.Module):
    """Regression head that conditions action predictions on instruction features via FiLM."""

    def __init__(self, hidden_dim: int, action_dim: int = 8, film_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.film_dim = film_dim

        self.attn_scorer = nn.Linear(hidden_dim, 1)
        self.film_generator = nn.Linear(hidden_dim, film_dim * 2)
        self.action_encoder = nn.Sequential(
            nn.Linear(hidden_dim, film_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.output_head = nn.Sequential(
            nn.Linear(film_dim, action_dim),
            nn.Tanh(),
        )

    def _compute_attention_weights(self, instr_hidden: torch.Tensor) -> torch.Tensor:
        """Compute attention weights over instruction tokens.

        Args:
            instr_hidden: [B, N_instr, hidden_dim]

        Returns:
            Attention weights [B, N_instr, 1] summing to 1 along dim=1.
        """
        scores = self.attn_scorer(instr_hidden)  # [B, N, 1]
        weights = F.softmax(scores, dim=1)  # [B, N, 1]
        return weights

    def forward(
        self, instr_hidden: torch.Tensor, action_vec: torch.Tensor
    ) -> torch.Tensor:
        """Produce action predictions conditioned on instruction context.

        Args:
            instr_hidden: [B, N_instr, hidden_dim] — hidden states at instruction positions
            action_vec:   [B, hidden_dim]           — last token hidden state

        Returns:
            Action predictions [B, action_dim] in [-1, 1].
        """
        weights = self._compute_attention_weights(instr_hidden)  # [B, N, 1]
        instr_vec = (weights * instr_hidden).sum(dim=1)  # [B, hidden_dim]

        gamma_beta = self.film_generator(instr_vec)  # [B, film_dim * 2]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, film_dim] each

        features = self.action_encoder(action_vec)  # [B, film_dim]
        conditioned = gamma * features + beta  # [B, film_dim]

        output = self.output_head(conditioned)  # [B, action_dim]
        return output
