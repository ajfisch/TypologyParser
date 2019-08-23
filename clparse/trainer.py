"""Training utilties."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from clparse.meters import AverageMeter
from clparse.meters import Timer
from clparse.optimizer import get_optimizer

logger = logging.getLogger()


class Trainer:
    """Trains parser."""

    def __init__(self, args, model, device):
        self.args = args
        self.device = device
        self.iters_since_last = 0
        self.loss = AverageMeter()
        self.timer = Timer()
        self.M = model
        self.keys = model.KEYS_FOR_FORWARD
        if getattr(args, 'multigpu', False):
            self.M = nn.DataParallel(self.M).to(device)
        self.optimizer = get_optimizer(self.M.parameters(), args.optimizer)

    def cross_entropy(self, scores, targets):
        flat_scores = scores.contiguous().view(-1, scores.size(-1))
        flat_targets = targets.view(-1)
        flat_loss = F.cross_entropy(flat_scores, flat_targets,
                                    ignore_index=-1, reduction='none')
        loss = flat_loss.view(scores.size(0), scores.size(1))
        return loss

    def check_loss(self, loss):
        if loss.sum() != loss.sum():
            logger.critical('Encountered NaN loss!')
            exit()

    def get_loss(self, arc_scores, heads, deprel_scores, deprels):
        arc_loss = self.cross_entropy(arc_scores, heads)
        deprel_loss = self.cross_entropy(deprel_scores, deprels)
        loss = torch.stack([arc_loss, deprel_loss], dim=2)
        self.check_loss(loss)
        return loss

    def step(self, inputs):
        self.M.train()
        inputs = inputs.to(self.device)
        batch_size = len(inputs)
        items = inputs['mask'].eq(0).sum().item()

        # Compute nll of scores.
        args = {k: inputs[k] for k in self.keys}
        arc_scores, deprel_scores = self.M(**args)
        loss = self.get_loss(arc_scores, inputs['heads'],
                             deprel_scores, inputs['deprels'])

        # Reduce & record.
        loss = loss.sum()
        self.loss.update(loss.item(), items)
        self.iters_since_last += 1

        # Update.
        self.optimizer.zero_grad()
        loss = loss / batch_size
        loss.backward()
        nn.utils.clip_grad_norm_(self.M.parameters(), self.args.clip_grad_norm)
        self.optimizer.step()

    def log(self, step):
        logger.info('| step {:6d} | loss: {} | it/sec: {:5.2f}'.format(
            step, self.loss, self.iters_since_last / self.timer.time()))
        self.loss.reset()
        self.timer.reset()
        self.iters_since_last = 0
