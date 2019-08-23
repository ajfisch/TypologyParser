"""Typology helpers."""

import gzip
import logging
import pickle
import torch
import torch.nn as nn

logger = logging.getLogger()


class FeedForward(nn.Module):
    """A simple fully connected feed forward network."""

    # Activation types.
    A = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

    def __init__(
            self,
            input_dim,
            output_dim,
            layers,
            activation='tanh',
            dropout=0):
        """Initialize feed forward network."""
        super(FeedForward, self).__init__()
        modules = []
        for i in range(layers):
            if i == 0:
                modules.append(nn.Linear(input_dim, output_dim))
            else:
                modules.append(nn.Dropout(dropout))
                modules.append(nn.Linear(output_dim, output_dim))
            modules.append(self.A[activation]())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class ClusterFeature(nn.Module):
    """Feed-forward network on one-hot cluster id."""

    def __init__(
            self,
            lang2idx,
            filename,
            output_dim,
            ff_layers,
            activation='sigmoid',
            dropout=0):
        """Initialize ClusterFeature.

        Args:
          lang2idx: Mapping of language name to language index.
          filename: Path to saved clusters {language_name: cluster_id}.
          output_dim: Dimension of cluster embedding.
          ff_layers: Number of feed-forward layers in cluster embedding.
          activation: Activation of the feed-forward network.
          dropout: Dropout ratio in the feed-forward network.
        """
        super(ClusterFeature, self).__init__()
        with gzip.open(filename, 'rb') as f:
            self.clusters = pickle.load(f)
        self.lang2idx = lang2idx
        num_clusters = len(set(self.clusters.values()))
        self.register_buffer('weight', torch.eye(num_clusters, num_clusters))
        self.ffnn = FeedForward(
            input_dim=num_clusters,
            output_dim=output_dim,
            layers=ff_layers,
            activation=activation,
            dropout=dropout)

    def forward(self, lang):
        """Forward pass the cluster embedding of language id.

        Args:
          lang: <int64>

        Returns:
          output: <float32> [output_dim]
        """
        cluster_id = self.clusters[self.lang2idx[lang.item()]]
        features = self.weight[cluster_id].squeeze()
        output = self.ffnn(features)
        return output


class HandFeature(nn.Module):
    """Feed-forward network on hand-designed features (e.g., WALS)."""

    def __init__(
            self,
            lang2idx,
            filename,
            output_dim,
            ff_layers,
            activation='sigmoid',
            dropout=0):
        """Initialize HandFeature.

        Args:
          lang2idx: Mapping of language name to language index.
          filename: Path to saved features {language_name: features}.
          output_dim: Dimension of language embedding.
          ff_layers: Number of feed-forward layers in language embedding.
          activation: Activation of the feed-forward network.
          dropout: Dropout ratio in the feed-forward network.
        """
        super(HandFeature, self).__init__()
        self.lang2idx = lang2idx
        with gzip.open(filename, 'rb') as f:
            typologies = pickle.load(f)
        input_dim = len(next(iter(typologies.values())))
        self.register_buffer('weight', torch.empty(len(lang2idx), input_dim))
        for lang, idx in lang2idx.items():
            assert(lang in typologies), 'Typology for %s not found' % lang
            self.weight.data[idx] = torch.Tensor(typologies[lang])
        self.ffnn = FeedForward(
            input_dim=input_dim,
            output_dim=output_dim,
            layers=ff_layers,
            activation=activation,
            dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def reload(self, filename):
        """Reload new feature vectors from disk."""
        with open(filename, 'rb') as f:
            typologies = pickle.load(f)
        for lang, idx in self.lang2idx.items():
            assert(lang in typologies), 'Typology for %s not found' % lang
            self.weight.data[idx] = torch.Tensor(typologies[lang])

    def forward(self, lang):
        """Forward pass the language embedding of language id.

        Args:
          lang: <int64>

        Returns:
          output: <float32> [output_dim]
        """
        features = self.weight[lang].squeeze()
        output = self.ffnn(self.drop(features))
        return output


class NeuralFeature(nn.Module):
    """Neural average-pooled corpus encoder from Wang and Eisner 2018a."""

    def __init__(
            self,
            num_pos,
            input_dim,
            hidden_dim,
            num_layers,
            output_dim,
            ff_layers,
            activation='sigmoid',
            dropout=0):
        """Initialize NeuralFeature.

        Args:
          num_pos: Number of part-of-speech tags in vocab.
          input_dim: Input dimension of part-of-speech tags.
          hidden_dim: Hidden dimension of the GRU encoder.
          num_layers: Number of layers in the GRU encoder.
          output_dim: Dimension of the language embedding.
          ff_layers: Number of feed forward layers in language embedding.
          activateion: Activation of the feed-forward network.
          dropout: Dropout ratio in the feed-forward network.
        """
        super(NeuralFeature, self).__init__()
        self.pos_embed = nn.Embedding(num_pos, input_dim)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.ffnn = FeedForward(
            input_dim=hidden_dim,
            output_dim=output_dim,
            layers=ff_layers,
            activation=activation,
            dropout=dropout)
        self._cache = None

    def encode(self, seq, mask):
        """Encode batch and take the average of the final hidden states."""
        seq = self.pos_embed(seq)
        seq = seq.transpose(0, 1)
        if mask.data.eq(1).sum() > 0:
            lengths = mask.data.eq(0).long().sum(1).squeeze().cpu()
            seq = nn.utils.rnn.pack_padded_sequence(seq, lengths)
        hidden = self.encoder(seq)[1][-1]
        avg_hidden = hidden.mean(dim=0)
        return avg_hidden

    def set_cache(self, iterator):
        """Cache the average hidden state over an entire corpus.

        Args:
          iterator: Iterator that feeds batches of the corpus.
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            hidden = 0
            total = 0
            for batch in iterator:
                batch = batch.to(device)
                hidden += self.encode(batch['pos'], batch['mask']) * len(batch)
                total += len(batch)
            avg_hidden = hidden / total
            output = self.ffnn(avg_hidden)
        self._cache = output

    def clear_cache(self):
        self._cache = None

    def forward(self, seq, mask):
        """Forward pass the language embedding derived from corpus sequences.

        If the cache is set, return that. Otherwise return the stochastic
        encoding derived from the current batch (seq).

        Args:
          seq: <int64> [batch_size, seq_len]
          mask: <uint8> [batch_size, seq_len]

        Returns:
          output: <float32> [output_dim]
        """
        if self._cache is not None:
            return self._cache
        avg_hidden = self.encode(seq, mask)
        output = self.ffnn(self.drop(avg_hidden))
        return output
