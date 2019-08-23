"""Typology as Selective Sharing parser implementation."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from clparse.parsers.parser import Parser
from clparse.utils import INF
from clparse.utils import mst_decode

logger = logging.getLogger()


class TaSSParser(Parser):
    """Typology as Selective Sharing parser.

    Typological features are used as part of a linear bias term to arc scoring.
    Shared features follow the same rules as Zhang and Barzilay, 2015.
    """

    KEYS_FOR_FORWARD = ['lang', 'pos', 'mask', 'selective', 'heads']

    KEYS_FOR_PREDICT = ['lang', 'pos', 'mask', 'selective']

    def init_params(self, args, dicos):
        """Initialize TaSSParser params."""
        super(TaSSParser, self).init_params(args, dicos)
        self.selective_bias = nn.Linear(args.selective_feature_dim, 1)

    def score_arcs(self, lang, seq, features, mask):
        """Given an encoded sequence, score arc heads with selective sharing.

        Args:
          lang: <int64>
          seq: <float32> [batch_size, seq_len + 1, hidden_dim]
          mask: <uint8> [batch_size, seq_len]

        Returns:
          arc_scores: <float32> [batch_size, seq_len + 1, seq_len + 1]
        """
        arc_scores = self.arc_scorer(self.drop(seq))
        arc_bias = self.selective_bias(features).squeeze(-1)
        arc_bias_with_root = F.pad(arc_bias, (1, 0, 1, 0))
        arc_scores += arc_bias_with_root
        arc_scores[:, 1:, 1:].transpose(1, 2).data[mask.data] = -INF
        diag_mask = torch.eye(arc_scores.size(1), out=mask.new())
        diag_mask = diag_mask.unsqueeze(0).expand_as(arc_scores)
        arc_scores.data[diag_mask.data] = -INF
        return arc_scores

    def forward(self, lang, pos, mask, heads, selective):
        """Score arcs and dependency labels for training.

        Args:
          lang: <int64> [batch_size]
          pos: <int64> [batch_size, seq_len]
          mask: <uint8> [batch_size, seq_len]
          heads: <int64> [batch_size, seq_len]
          selective: <float32> [batch_size, seq_len, features]

        Returns:
          arc_scores: <float32> [batch_size, seq_len, seq_len + 1]
          pred_deprels: <float32> [batch_size, seq_len, num_deprel]
        """
        # Hack when assuming homogeneous batch sizes.
        lang = lang[0].view(1, 1)
        embs = self.embed(lang, pos, mask)
        encs = self.encode(lang, embs, mask)
        encs_with_root = self.add_root(lang, encs)
        arc_scores = self.score_arcs(
            lang, encs_with_root, selective, mask)
        heads_with_root = F.pad(heads, (1, 0))
        deprel_scores = self.score_deprels(
            lang, encs_with_root, heads_with_root, mask)
        return arc_scores[:, 1:], deprel_scores[:, 1:]

    def predict(self, lang, pos, selective, mask):
        """Run inference for heads and labels given the predicted heads.

        Args:
          lang: <int64> [batch_size]
          pos: <int64> [batch_size, seq_len]
          mask: <uint8> [batch_size, seq_len]
          selective: <float32> [batch_size, seq_len, features]

        Returns:
          pred_heads: <int64> [batch_size, seq_len]
          pred_deprels: <int64> [batch_size, seq_len]
        """
        with torch.no_grad():
            lang = lang[0].view(1, 1)
            embs = self.embed(lang, pos, mask)
            encs = self.encode(lang, embs, mask)
            encs_with_root = self.add_root(lang, encs)
            arc_scores = self.score_arcs(lang, encs_with_root, selective, mask)
            pred_heads = torch.stack([mst_decode(s) for s in arc_scores], dim=0)
            deprel_scores = self.score_deprels(
                lang, encs_with_root, pred_heads, mask)
            pred_deprels = torch.max(deprel_scores, dim=2)[1]
        return pred_heads[:, 1:], pred_deprels[:, 1:]
