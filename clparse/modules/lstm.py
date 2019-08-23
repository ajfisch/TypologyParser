"""LSTM layers."""

import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger()


class BLSTM(nn.Module):
    """Bidirectional LSTM with variational dropout."""

    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_layers,
            dropout=0,
            variational_dropout=0):
        """Initialize Bidirectional LSTM.

        Args:
          input_dim: Number of input features.
          hidden_dim: Number of hidden state dimensions.
          num_layers: Number of stacked layers.
          dropout: Dropout ratio applied to input.
          variational_dropout: Dropout ratio applied to hidden state.
        """
        super(BLSTM, self).__init__()
        dropout = 0 if num_layers == 1 else dropout
        lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True)
        if variational_dropout > 0:
            weights = []
            for i in range(num_layers):
                weights.append('weight_hh_l%d' % i)
                weights.append('weight_hh_l%d_reverse' % i)
            lstm = WeightDrop(
                module=lstm,
                weights=weights,
                dropout=variational_dropout,
                variational=True)
        self.lstm = lstm

    def forward(self, seq, mask):
        seq = seq.transpose(0, 1)
        if mask.data.eq(1).sum() > 0:
            # Compute sorted sequence lengths. Should be pre-sorted.
            total_length = seq.size(0)
            lengths = mask.data.eq(0).long().sum(1).view(-1).cpu()
            seq = nn.utils.rnn.pack_padded_sequence(seq, lengths)
            output, hidden = self.lstm(seq)
            output = nn.utils.rnn.pad_packed_sequence(
                output, total_length=total_length)[0]
        else:
            output, hidden = self.lstm(seq)
        output = output.transpose(0, 1).contiguous()
        return output, hidden


class WeightDrop(nn.Module):
    """Applies time-persistent dropout (weight drop / variational dropout).
    References:
      1) github.com/salesforce/awd-lstm-lm/ and
      2) github.com/fastai/fastai/blob/master/fastai/text/models/awd_lstm.py
    """

    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling
        # explodes We can't write boring code though, so ...
        # WIDGET DEMAGNETIZER Y2K EDITION! (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights.
        if issubclass(type(self.module), torch.nn.RNNBase):
            flatten = self.widget_demagnetizer_y2k_edition
            self.module.flatten_parameters = flatten

        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(
                name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.ones(raw_w.size(0), 1)
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = F.dropout(mask, p=self.dropout, training=self.training)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights
            # aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = F.dropout(raw_w, 0, training=False)
            setattr(self.module, name_w, w)
            if hasattr(self.module, 'reset'):
                self.module.reset()
