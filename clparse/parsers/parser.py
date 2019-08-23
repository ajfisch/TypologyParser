"""Baseline parser implementation."""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from clparse.data.dictionary import Dictionary
from clparse.modules.biaffine import ArcBiaffineScorer
from clparse.modules.biaffine import DeprelBiaffineScorer
from clparse.modules.lstm import BLSTM
from clparse.utils import INF
from clparse.utils import init_dir
from clparse.utils import mst_decode

logger = logging.getLogger()


class Parser(nn.Module):
    """Basic Deep Biaffine Attention parser."""

    KEYS_FOR_FORWARD = ['lang', 'pos', 'mask', 'heads']

    KEYS_FOR_PREDICT = ['lang', 'pos', 'mask']

    def __init__(self, args, dicos):
        """Initialize Parser.

        Args:
          args: Parser config
          dicos: Dict of Dictionary objects.
            - lang: Dictionary of languages.
            - deprel: Dictionary of dependency relations.
            - pos: Dictionary of part-of-speech tags.
        """
        super(Parser, self).__init__()
        self.args = args
        self.dicos = dicos
        self.init_params(args, dicos)

    def init_params(self, args, dicos):
        """Initialize model-specific parameters."""
        self.embedder = nn.Embedding(args.num_pos, args.pos_dim, padding_idx=0)
        self.encoder = BLSTM(
            input_dim=args.pos_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            variational_dropout=args.variational_dropout)
        self.arc_scorer = ArcBiaffineScorer(
            input_dim=2 * args.hidden_dim,
            hidden_dim=args.arc_dim,
            dropout=args.dropout)
        self.deprel_scorer = DeprelBiaffineScorer(
            input_dim=2 * args.hidden_dim,
            hidden_dim=args.deprel_dim,
            num_deprel=args.num_deprel,
            dropout=args.dropout)
        self.root = nn.Parameter(torch.randn(2 * args.hidden_dim))
        self.drop = nn.Dropout(args.dropout)

    def embed(self, lang, seq, mask):
        """Embed the sequence tokens.

        Args:
          lang: <int64>
          seq: <int64> [batch_size, seq_len]
          mask: <uint8> [batch_size, seq_len]

        Returns:
          embs: <float32> [batch_size, seq_len, embedding_dim]
        """
        return self.embedder(seq)

    def encode(self, lang, seq, mask):
        """Contextually encode the sequence tokens.

        Args:
          lang: <int64>
          seq: <float32> [batch_size, seq_len, input_dim]
          mask: <uint8> [batch_size, seq_len]

        Returns:
          encs: <float32> [batch_size, seq_len, hidden_dim]
        """
        return self.encoder(self.drop(seq), mask)[0]

    def add_root(self, lang, encs):
        """Append dummy root token encoding to sequence.

        Args:
          lang: <int64>
          encs: <float32> [batch_size, seq_len, hidden_dim]

        Returns:
          encs: <float32> [batch_size, seq_len + 1, hidden_dim]
        """
        root = self.root.view(1, 1, -1)
        root = root.expand(encs.size(0), 1, self.root.numel())
        return torch.cat([root, encs], dim=1)

    def score_arcs(self, lang, seq, mask):
        """Given an encoded sequence, score arc heads.

        Args:
          lang: <int64>
          seq: <float32> [batch_size, seq_len + 1, hidden_dim]
          mask: <uint8> [batch_size, seq_len]

        Returns:
          arc_scores: <float32> [batch_size, seq_len + 1, seq_len + 1]
        """
        arc_scores = self.arc_scorer(self.drop(seq))
        arc_scores[:, 1:, 1:].transpose(1, 2).data[mask.data] = -INF
        diag_mask = torch.eye(arc_scores.size(1), out=mask.new())
        diag_mask = diag_mask.unsqueeze(0).expand_as(arc_scores)
        arc_scores.data[diag_mask.data] = -INF
        return arc_scores

    def score_deprels(self, lang, seq, heads, mask):
        """Given an encoded sequence and predicted arcs, score arc labels.

        Args:
          lang: <int64>
          seq: <float32> [batch_size, seq_len + 1, hidden_dim]
          heads: <int64> [batch_size, seq_len + 1]
          mask: <uint8> [batch_size, seq_len]

        Returns:
          deprel_scores: <float32> [batch_size, seq_len + 1, num_deprel]
        """
        # Adjust padding to have root as the head.
        if torch.any(mask):
            heads = heads.clone()
            heads[:, 1:].data[mask.data] = 0

        # Grab heads and score.
        heads = heads.unsqueeze(-1).expand_as(seq)
        head_encs = seq.gather(1, heads)
        deprel_scores = self.deprel_scorer(self.drop(seq), self.drop(head_encs))

        # Manually adjust length 1 sequences to have root label.
        # Do this during eval only.
        if not self.training:
            singles = mask.eq(0).sum(dim=1).eq(1)
            if torch.any(singles):
                root_idx = self.dicos['deprel']['root']
                deprel_scores.data[singles.data, 1, root_idx] = INF

        return deprel_scores

    def forward(self, lang, pos, mask, heads):
        """Score arcs and dependency labels for training.

        Args:
          lang: <int64> [batch_size]
          pos: <int64> [batch_size, seq_len]
          mask: <uint8> [batch_size, seq_len]
          heads: <int64> [batch_size, seq_len]

        Returns:
          arc_scores: <float32> [batch_size, seq_len, seq_len + 1]
          pred_deprels: <float32> [batch_size, seq_len, num_deprel]
        """
        # Hack when assuming homogeneous batch sizes.
        lang = lang[0].view(1, 1)
        embs = self.embed(lang, pos, mask)
        encs = self.encode(lang, embs, mask)
        encs_with_root = self.add_root(lang, encs)
        arc_scores = self.score_arcs(lang, encs_with_root, mask)
        heads_with_root = F.pad(heads, (1, 0))
        deprel_scores = self.score_deprels(
            lang, encs_with_root, heads_with_root, mask)
        return arc_scores[:, 1:], deprel_scores[:, 1:]

    def predict(self, lang, pos, mask):
        """Run inference for heads and then labels given the predicted heads.

        Args:
          lang: <int64> [batch_size]
          pos: <int64> [batch_size, seq_len]
          mask: <uint8> [batch_size, seq_len]

        Returns:
          pred_heads: <int64> [batch_size, seq_len]
          pred_deprels: <int64> [batch_size, seq_len]
        """
        with torch.no_grad():
            lang = lang[0].view(1, 1)
            embs = self.embed(lang, pos, mask)
            encs = self.encode(lang, embs, mask)
            encs_with_root = self.add_root(lang, encs)
            arc_scores = self.score_arcs(
                lang, encs_with_root, mask)
            pred_heads = torch.stack([mst_decode(s) for s in arc_scores], dim=0)
            deprel_scores = self.score_deprels(
                lang, encs_with_root, pred_heads, mask)
            pred_deprels = torch.max(deprel_scores, dim=2)[1]
        return pred_heads[:, 1:], pred_deprels[:, 1:]

    def save(self, dirname):
        init_dir(dirname)
        logger.info('Saving model to %s' % dirname)
        torch.save(self.args, os.path.join(dirname, 'args.pt'))
        for k, v in self.dicos.items():
            filename = os.path.join(dirname, k + '.txt')
            v.save(filename)
        torch.save(self.state_dict(), os.path.join(dirname, 'weights.pt'))

    @classmethod
    def load(cls, dirname):
        args = torch.load(os.path.join(dirname, 'args.pt'))
        dicos = {
            'lang': Dictionary.load(os.path.join(dirname, 'lang.txt')),
            'pos': Dictionary.load(os.path.join(dirname, 'pos.txt')),
            'deprel': Dictionary.load(os.path.join(dirname, 'deprel.txt')),
        }
        state_dict = torch.load(os.path.join(dirname, 'weights.pt'),
                                map_location=lambda storage, loc: storage)
        model = cls(args, dicos)
        model.load_state_dict(state_dict)
        return model
