import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from .train_utils import _add_margins, _add_DRW, squared_l2_norm, l2_norm

def plain_train(model, x_natural, y, **kwargs):

    logits = model(x_natural)
    loss_func = model.loss if not isinstance(model, DDP) else model.module.loss
    loss = loss_func(logits, y)
    return loss

def pgd_adv_train(model,
                  x_natural,
                  y,
                  optimizer,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf',
                  rand_start_mode='uniform',
                  rand_start_step=1,
                  deffer_opt=None,
                  samples_per_cls=None,
                  **kwargs):

    model.eval()
    batch_size = len(x_natural)

    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + rand_start_step * 0.001 * torch.randn(x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform':
        x_adv = x_natural.detach() + rand_start_step * epsilon * torch.rand(x_natural.shape).cuda().detach()
    else:
        raise NameError

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                loss_ce = nn.CrossEntropyLoss()(model(x_adv), y)

            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    weights = None
    loss_func = model.loss if not isinstance(model, DDP) else model.module.loss
    if deffer_opt is not None:
        if 'DRW' in deffer_opt:
            weights = _add_DRW(model.epoch, samples_per_cls, **deffer_opt['DRW'])
        if 'DM' in deffer_opt:
            logits = _add_margins(model.epoch, logits, y, samples_per_cls, **deffer_opt['DM'])
    loss = loss_func(logits, y, weights)
    return loss

def trades_train(model,
                 x_natural,
                 y,
                 optimizer,
                 step_size=0.003,
                 epsilon=0.031,
                 perturb_steps=10,
                 beta=1.0,
                 distance='l_inf',
                 **kwargs):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss


def inner_loss_train(model,
                  x_natural,
                  y,
                  optimizer,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf',
                  **kwargs):

    model.eval()

    outer_loss = model.loss if not isinstance(model, DDP) else model.module.loss
    inner_loss = model.adv_loss if not isinstance(model, DDP) else model.module.adv_loss

    # generate adversarial example
    x_adv = x_natural.detach() + epsilon*torch.rand(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = inner_loss(model(x_adv), y)
                # loss_ce = nn.CrossEntropyLoss()(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    loss = outer_loss(model(x_adv), y)
    return loss

def add_reg_train(model,
                       x_natural,
                       y,
                       optimizer,
                       step_size=0.003,
                       epsilon=0.031,
                       perturb_steps=10,
                       distance='l_inf',
                       beta_adv=1.0,
                       beta_nat=0.0,
                       beta_reg=0.0,
                       reg_opt=None,
                       **kwargs):

    model.eval()
    batch_size = len(x_natural)
    num_classes = model.num_classes

    outer_adv_loss = model.loss if not isinstance(model, DDP) else model.module.loss
    inner_loss = model.adv_loss if not isinstance(model, DDP) else model.module.adv_loss
    outer_nat_loss = model.nat_loss if not isinstance(model, DDP) else model.module.nat_loss

    # generate adversarial example
    x_adv = x_natural.detach() + epsilon*torch.rand(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = inner_loss(model(x_adv), y) # loss_ce = nn.CrossEntropyLoss()(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    loss = outer_adv_loss(model(x_adv), y) * beta_adv

    if outer_nat_loss is not None:
        loss += outer_nat_loss(model(x_natural), y) * beta_nat

    if reg_opt is not None:
        if reg_opt == 'trades':
            nat_probs = F.softmax(model(x_natural), dim=1)
            adv_probs = F.softmax(model(x_adv), dim=1)
            criterion_kl = nn.KLDivLoss(reduction='sum')
            loss += (1.0 / batch_size) * criterion_kl(torch.log(adv_probs + 1e-12), nat_probs) * beta_reg
        elif reg_opt == 'mart':
            nat_probs = F.softmax(model(x_natural), dim=1)
            adv_probs = F.softmax(model(x_adv), dim=1)
            criterion_kl = nn.KLDivLoss(reduction='sum')
            loss += (1.0 / batch_size) * criterion_kl(torch.log(adv_probs + 1e-12), nat_probs) \
                    * beta_reg * (1.0000001 - nat_probs)
        elif reg_opt == 'ALP':
            nat_probs = F.softmax(model(x_natural), dim=1)
            adv_probs = F.softmax(model(x_adv), dim=1)
            loss += (1.0 / batch_size) * torch.sum((adv_probs - nat_probs)**2)**0.5 * beta_reg
        elif reg_opt == 'ALP_2':
            loss += (1.0 / batch_size) * torch.sum((model(x_natural) - model(x_adv))**2)**0.5 * beta_reg
        elif reg_opt == 'norm':
            loss += (1.0 / batch_size) * torch.sum((model(x_natural))**2)**0.5 * beta_reg
        else:
            raise NotImplementedError

    return loss

def mixup_vertex_train(model,
                       x_natural,
                       y,
                       optimizer,
                       step_size=0.003,
                       epsilon=0.031,
                       perturb_steps=10,
                       distance='l_inf',
                       gamma=2.,
                       lambda_1 = 1.,
                       lambda_2 = 0.1,
                       **kwargs):

    model.eval()
    num_classes = model.num_classes
    # generate adversarial example
    x_adv = x_natural.detach() + epsilon*torch.rand(x_natural.shape).cuda().detach()
    alpha = torch.rand(len(x_natural),1,1,1).cuda()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv_ = x_natural + (x_adv - x_natural) * gamma

        x_adv_hat = alpha * x_natural + (1 - alpha) * x_adv_ # x_natural + delta * gamma * (1 - alpha)
    else:
        raise NotImplementedError

    model.train()
    x_adv_hat = Variable(torch.clamp(x_adv_hat, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv_hat)
    labels_one_hot = F.one_hot(y, num_classes).float()
    log_softmax = F.log_softmax(logits, dim=-1)
    alpha = alpha[:,:,0,0]
    pos_label = alpha*lambda_1 + (1-alpha)*lambda_2
    neg_label = (alpha*(1-lambda_1) + (1-alpha)*(1-lambda_2)) / (num_classes-1)
    soft_labels = labels_one_hot * pos_label + (1 - labels_one_hot) * neg_label

    loss = - soft_labels * log_softmax
    loss = torch.sum(loss, dim=-1)
    loss = loss.mean()

    return loss

def mixup_train(model,
               x_natural,
               y,
               optimizer,
               step_size=0.003,
               epsilon=0.031,
               perturb_steps=10,
               distance='l_inf',
               **kwargs):

    model.eval()
    mixed_x, y_a, y_b, lam = model.classifier.mixup_data(x_natural, y)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(mixed_x.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y_a)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise NotImplementedError

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss_func = model.loss if not isinstance(model, DDP) else model.module.loss
    loss = loss_func(logits, y_a, y_b, lam)
    return loss

