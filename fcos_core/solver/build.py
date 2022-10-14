# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import logging
from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    logger = logging.getLogger("fcos_core.trainer")
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if key.endswith(".offset.weight") or key.endswith(".offset.bias"):
            logger.info("set lr factor of {} as {}".format(
                key, cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
            ))
            lr *= cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )

from torch.optim.optimizer import Optimizer, required

class newSGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay1=0, tau=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, weight_decay1=weight_decay1,
                        tau = tau, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(newSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(newSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            tau = group['tau']
            weight_decay1 = group['weight_decay1']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                
                if tau != 0:
                    g = (d_p * p).abs()
                    m = p.numel()
                    mn = int(m*tau)  
                    if mn>0:
                        kth,_ = g.flatten().kthvalue(mn)
                        d_p = torch.where(g < kth, torch.zeros_like(d_p), d_p)
                    d_p.mul_(1 - tau)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                elif weight_decay1 != 0:
                    d_p = d_p.add(torch.sign(p), alpha=weight_decay1)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss

def make_cdr_optimizer(cfg, model):
    logger = logging.getLogger("fcos_core.trainer")
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if key.endswith(".offset.weight") or key.endswith(".offset.bias"):
            logger.info("set lr factor of {} as {}".format(
                key, cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
            ))
            lr *= cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = newSGD(params, lr, momentum=cfg.SOLVER.MOMENTUM, tau=0.2, weight_decay1=1e-3)
    return optimizer