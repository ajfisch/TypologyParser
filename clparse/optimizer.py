"""Optimizer utititlies.

Adapted from various www.github.com/facebookresearch libraries.
"""

import inspect
import re
import torch.optim as optim


class AdamInverseSqrtWithWarmup(optim.Adam):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:

      lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
      lr = lrs[update_num]

    After warmup:

      lr = decay_factor / sqrt(update_num)

    where

      decay_factor = lr * sqrt(warmup_updates)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr

        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates**0.5

        self._num_updates = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * num_updates**-0.5

    def step(self, closure=None):
        super().step(closure)
        self._num_updates += 1

        # update learning rate
        new_lr = self.get_lr_for_step(self._num_updates)
        for param_group in self.param_groups:
            param_group['lr'] = new_lr


def get_optimizer(parameters, s):
    """Parse optimizer parameters.

    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    parameters = list(parameters)

    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            is_float = False
            try:
                float(split[1])
                is_float = True
            except ValueError:
                pass
            if is_float:
                optim_params[split[0]] = float(split[1])
            elif split[1] == 'True':
                optim_params[split[0]] = True
            elif split[1] == 'False':
                optim_params[split[0]] = False
            else:
                raise ValueError('Could not parse value.')
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.5),
                                 optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'sparseadam':
        optim_fn = optim.SparseAdam
        optim_params['betas'] = (optim_params.get('beta1', 0.5),
                                 optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    elif method == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['betas'] = (optim_params.get('beta1', 0.9),
                                 optim_params.get('beta2', 0.98))
        optim_params['warmup_updates'] = optim_params.get('warmup_updates',
                                                          4000)
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(parameters, **optim_params)
