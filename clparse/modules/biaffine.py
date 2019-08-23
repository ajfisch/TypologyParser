"""Deep Biaffine Attention mechanisms for arc and label prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcBiaffineScorer(nn.Module):
    r"""Computes the biaffine attention scores for arc prediction.

    s_i^{arc} = H^{arc-head}U^{(1)}h_i^{arc-dep} + H^{arc-head}u^{(2)}
    """

    def __init__(self, input_dim, hidden_dim, dropout=0):
        """Initialize arc scorer.

        Args:
          input_dim: Number of input features.
          hidden_dim: Hidden dimension of internal MLP.
          dropout: Dropout ratio applied to hidden state.
        """
        super(ArcBiaffineScorer, self).__init__()
        self.head_ffnn = nn.Linear(input_dim, hidden_dim)
        self.child_ffnn = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq):
        head = F.relu(self.head_ffnn(seq))
        child = F.relu(self.child_ffnn(seq))
        child_proj = self.proj(child)

        batch_size, seq_len, dim = head.size()
        scores = head.bmm(child_proj.transpose(1, 2))
        return scores


class DeprelBiaffineScorer(nn.Module):
    r"""Computes the biaffine attention scores for label prediction.

    s_i^{label} = {h_{y_i}^{label-head}}^\top U^{(1)} h_i^{label-dep}
                  + ({h_{y_i}^{label-head}} \oplus h_i^{label-dep})^\top U^{(2)}
                  + b
    """
    def __init__(self, input_dim, hidden_dim, num_deprel, dropout=0):
        """Initialize label scorer.

        Args:
          input_dim: Number of input features.
          hidden_dim: Hidden dimension of internal MLP.
          num_deprel: Number of dependency relation classes.
          dropout: Dropout ratio applied to hidden state.
        """
        super(DeprelBiaffineScorer, self).__init__()
        self.head_ffnn = nn.Linear(input_dim, hidden_dim)
        self.child_ffnn = nn.Linear(input_dim, hidden_dim)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, num_deprel)
        self.linear = nn.Linear(2 * hidden_dim, num_deprel)
        self.drop = nn.Dropout(dropout)

    def forward(self, head, child):
        head = F.relu(self.head_ffnn(head))
        child = F.relu(self.child_ffnn(child))

        # Likelihood given both.
        batch_size, seq_len, dim = head.size()
        bilinear = self.bilinear(self.drop(head.view(-1, dim)),
                                 self.drop(child.view(-1, dim)))
        bilinear = bilinear.view(batch_size, seq_len, -1)

        # Prior likelihoods.
        combined = torch.cat([head, child], dim=-1)
        linear = self.linear(self.drop(combined))

        return bilinear + linear
