"""Typology as Feature parser implementation."""

import logging
import os

import torch
import torch.nn as nn

from clparse.data.dictionary import Dictionary
from clparse.modules.biaffine import ArcBiaffineScorer
from clparse.modules.biaffine import DeprelBiaffineScorer
from clparse.modules.lstm import BLSTM
from clparse.modules.typology import ClusterFeature
from clparse.modules.typology import HandFeature
from clparse.modules.typology import NeuralFeature
from clparse.parsers.parser import Parser
from clparse.utils import export_resource

logger = logging.getLogger()


class TaFParser(Parser):
    """Typology as Feature parser.

    Typological features are injected at the input level, before the encoder.
    """

    def init_params(self, args, dicos):
        """Initialize TaFParser params."""
        self.embedder = nn.Embedding(args.num_pos, args.pos_dim, padding_idx=0)
        input_dim = args.pos_dim

        # Append all different feature types.
        if not hasattr(args, 'use_cluster_features'):
            args.use_cluster_features = False
        if args.use_cluster_features:
            # Cluster features are coarse, low-dimensional, and 1-hot.
            self.cluster_typology = ClusterFeature(
                lang2idx=dicos['lang'],
                filename=args.typology_clusters,
                output_dim=args.hand_typology_dim,
                ff_layers=args.hand_typology_layers,
                activation=args.hand_typology_activation,
                dropout=args.hand_typology_dropout)
            input_dim += args.hand_typology_dim

        if args.use_hand_features:
            # Hand features are linguistically motivated and hand engineered.
            self.hand_typology = HandFeature(
                lang2idx=dicos['lang'],
                filename=args.hand_typology,
                output_dim=args.hand_typology_dim,
                ff_layers=args.hand_typology_layers,
                activation=args.hand_typology_activation,
                dropout=args.hand_typology_dropout)
            input_dim += args.hand_typology_dim

        if args.use_neural_features:
            # Neural features are computed dynamically from corpus POS tags.
            self.neural_typology = NeuralFeature(
                num_pos=args.num_pos,
                input_dim=args.pos_dim,
                hidden_dim=args.neural_typology_rnn_dim,
                num_layers=args.neural_typology_rnn_layers,
                output_dim=args.neural_typology_dim,
                ff_layers=args.neural_typology_layers,
                activation=args.neural_typology_activation,
                dropout=args.neural_typology_dropout)
            input_dim += args.neural_typology_dim

        # Initialize the rest of the encoder normally.
        self.encoder = BLSTM(
            input_dim=input_dim,
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
        """Embed sequence tokens.

        Typology features are added at each position.

        Args:
          lang: <int64>
          seq: <int64> [batch_size, seq_len]
          mask: <uint8> [batch_size, seq_len]
        """
        inputs = []
        batch_size, seq_len = seq.size()
        inputs.append(self.embedder(seq))
        if self.args.use_cluster_features:
            H = self.cluster_typology(lang)
            H = H.view(1, 1, -1).expand(batch_size, seq_len, H.numel())
            inputs.append(H)
        if self.args.use_hand_features:
            H = self.hand_typology(lang)
            H = H.view(1, 1, -1).expand(batch_size, seq_len, H.numel())
            inputs.append(H)
        if self.args.use_neural_features:
            N = self.neural_typology(seq, mask)
            N = N.view(1, 1, -1).expand(batch_size, seq_len, N.numel())
            inputs.append(N)
        return torch.cat(inputs, dim=-1)

    def save(self, dirname):
        super(TaFParser, self).save(dirname)
        # Save outside resources local to model directory for portability.
        if self.args.use_cluster_features:
            export_resource(self.args.typology_clusters,
                            os.path.join(dirname, 'clusters.pkl.gz'))
        if self.args.use_hand_features:
            export_resource(self.args.hand_typology,
                            os.path.join(dirname, 'typology.pkl.gz'))

    @classmethod
    def load(cls, dirname):
        args = torch.load(os.path.join(dirname, 'args.pt'))

        # Override args to set typology clusters and hand typology.
        args.typology_clusters = os.path.join(dirname, 'clusters.pkl.gz')
        args.hand_typology = os.path.join(dirname, 'typology.pkl.gz')

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
