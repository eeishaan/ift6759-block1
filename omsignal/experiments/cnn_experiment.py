#!/usr/bin/evn python3

import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, recall_score
from torch.nn import MSELoss, NLLLoss
from torch.optim import SGD, Adam

from omsignal.experiments import OmExperiment
from omsignal.models.cnn import RegressionNet, SimpleNet
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
        pred = torch.argmax(outputs, 1)
        ctx['predicted'].extend(pred.cpu().numpy())
        ctx['true_labels'].extend(labels.cpu().numpy())

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


class RegressionNetEperiment(OmExperiment):
    def __init__(self,
                 exp_file,
                 device, model=RegressionNet,
                 model_params={},
                 optimiser=Adam,
                 optimiser_params={},
                 criterion=MSELoss, criterion_params={}):
        super(RegressionNetEperiment, self).__init__(
            model,
            model_params,
            optimiser,
            optimiser_params,
            criterion,
            criterion_params,
            exp_file,
            device)

    def before_train(self, ctx):
        ctx.update({
            'tr_mean': [],
            'rr_std': [],
            'pr_mean': [],
            'loss_pr_mean': 0,
            'loss_tr_mean': 0,
            'loss_rr_std': 0,
        })

    def before_eval(self, ctx):
        self.before_train(ctx)

    def after_minibatch_eval(self, ctx, outputs, labels):
        self.after_forwardp(ctx, outputs, labels)

    def after_forwardp(self, ctx, outputs, labels):
        rr_std, tr_mean, pr_mean = outputs
        ctx['rr_std'].extend(rr_std.tolist())
        ctx['tr_mean'].extend(tr_mean.tolist())
        ctx['pr_mean'].extend(pr_mean.tolist())
        ctx['labels'].extend(labels.tolist())

    def after_train(self, ctx):
        super().after_train(ctx)
        epoch = ctx['epoch']

        if epoch % 25 != 0:
            return
        tr_mean = ctx['tr_mean']
        pr_mean = ctx['pr_mean']
        rr_std = ctx['rr_std']
        labels = np.array(ctx['labels'])
        k_pr_mean = kendalltau(pr_mean, labels[:, 0])[0]
        k_tr_mean = kendalltau(tr_mean, labels[:, 1])[0]
        k_rr_std = kendalltau(rr_std, labels[:, 2])[0]
        kendall_avg = np.mean([k_pr_mean, k_tr_mean, k_rr_std])

        message = "Epoch : {} Train Loss: {} \n"\
            .format(epoch, ctx['running_loss'])
        message += "Train RR STD loss: {} \n"\
            .format(ctx['loss_rr_std'].item())
        message += "Train TR Mean loss: {} \n"\
            .format(ctx['loss_tr_std'].item())
        message += "Train PR Mean loss: {} \n"\
            .format(ctx['loss_pr_mean'].item())
        message += "Train Kendall TR: {} \n".format(k_tr_mean)
        message += "Train Kendall RR: {} \n".format(k_rr_std)
        message += "Train Kendall PR: {} \n".format(k_pr_mean)
        message += "Train Avg Kendall: {} \n".format(kendall_avg)
        print(message)
        val_loader = ctx.get('val_loader')
        if val_loader is not None:
            self.eval(val_loader)

    def after_eval(self, ctx):
        tr_mean = ctx['tr_mean']
        pr_mean = ctx['pr_mean']
        rr_std = ctx['rr_std']
        labels = np.array(ctx['labels'])
        k_pr_mean = kendalltau(pr_mean, labels[:, 0])[0]
        k_tr_mean = kendalltau(tr_mean, labels[:, 1])[0]
        k_rr_std = kendalltau(rr_std, labels[:, 2])[0]
        kendall_avg = np.mean([k_pr_mean, k_tr_mean, k_rr_std])

        message = "Eval Kendall TR: {} \n".format(k_tr_mean)
        message += "Eval Kendall RR: {} \n".format(k_rr_std)
        message += "Eval Kendall PR: {} \n".format(k_pr_mean)
        message += "Eval Avg Kendall: {} \n".format(kendall_avg)

        print(message)

    def compute_loss(self, ctx, outputs, labels):
        RR_std, TR_mean, PR_mean = outputs
        loss_pr_mean = self._criterion(PR_mean, labels[:, 0].unsqueeze(1))
        loss_tr_mean = self._criterion(TR_mean, labels[:, 1].unsqueeze(1))
        loss_rr_std = self._criterion(RR_std, labels[:, 2].unsqueeze(1))
        ctx['loss_pr_mean'] += loss_pr_mean
        ctx['loss_tr_mean'] += loss_tr_mean
        ctx['loss_rr_std'] += loss_rr_std
        loss = loss_rr_std + loss_tr_mean + loss_pr_mean
        return loss
