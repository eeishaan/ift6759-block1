#!/usr/bin/evn python3

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score
from torch.nn import NLLLoss
from torch.optim import SGD

from omsignal.experiments import OmExperiment
from omsignal.models.cnn import SimpleNet
from omsignal.utils.transform.basic import ClipAndFlatten


class SimpleNetExperiment(OmExperiment):
    def __init__(self, exp_file, device, model=SimpleNet, model_params={}, optimiser=SGD, optimiser_params={}, criterion=NLLLoss, criterion_params={}):
        super(SimpleNetExperiment, self).__init__(
            model,
            model_params,
            optimiser,
            optimiser_params,
            criterion,
            criterion_params,
            exp_file,
            device)

    def before_train(self, ctx):
        ctx['loss_total'] = 0
        ctx['predicted'] = []
        ctx['true_labels'] = []

    def after_forwardp(self, ctx, outputs, labels):
        pred = torch.argmax(outputs, 1).cpu().numpy()
        last_boundary = 0
        for b in ctx['boundaries']:
            # append the label with most votes
            ctx['predicted'].append(
                np.argmax(np.bincount(pred[last_boundary:b])))
            ctx['true_labels'].append(int(labels[last_boundary]))
            last_boundary = b

    def after_train(self, ctx):
        super().after_train(ctx)
        epoch = ctx['epoch']

        if epoch % 25 != 0:
            return

        acc = recall_score(ctx['true_labels'],
                           ctx['predicted'], average='macro')
        acc = (1 - ((1 - acc)/(1-1/32)))
        message = "Epoch: {} Train loss: {} accuracy: {}".format(
            epoch, ctx['running_loss'].item(), acc)
        print(message)
        val_loader = ctx.get('val_loader')
        if val_loader is not None:
            self.eval(val_loader)

    def before_eval(self, ctx):
        self.before_train(ctx)

    def after_minibatch_eval(self, ctx, outputs, labels):
        self.after_forwardp(ctx, outputs, labels)

    def after_eval(self, ctx):
        acc = recall_score(ctx['true_labels'],
                           ctx['predicted'], average='macro')
        acc = (1 - ((1 - acc)/(1-1/32)))
        message = "Eval accuracy: {}".format(acc)
        print(message)
