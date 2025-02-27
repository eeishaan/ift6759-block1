import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from omsignal.constants import GRAPH_DIR
from omsignal.experiments import OmExperiment
from omsignal.models.lstm import LSTMModel
from omsignal.utils.transform.basic import ClipAndFlatten
from omsignal.utils.vis import plot_confusion_matrix


class LSTMExperiment(OmExperiment):
    def __init__(self,
                 exp_file,
                 device,
                 model_params,
                 model=LSTMModel,
                 optimiser=Adam,
                 optimiser_params={},
                 criterion=CrossEntropyLoss,
                 criterion_params={}):
        super(LSTMExperiment, self).__init__(
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

    def train(self,
              dataloader,
              epochs,
              validation_dataloader=None,
              start_epoch=None):
        start_epoch = start_epoch if start_epoch is not None else self._start_epoch
        for epoch in range(start_epoch, epochs):
            ctx = {
                'epoch': epoch,
                'running_loss': 0,
                'val_loader': validation_dataloader
            }
            self.before_train(ctx)
            for _, (data, labels) in enumerate(dataloader):
                self._model.train()
                data, labels = data.to(self._device), labels.to(self._device)
                data, labels = self.before_forwardp(ctx, data, labels)
                self._optimizer()
                self._model.zero_grad()
                hidden = self._model.init_hidden(data.shape(0))
                outputs, hidden = self._model(data, hidden)
                loss = self._criterion(outputs, labels)
                ctx['running_loss'] += loss
                loss.backward()
                self._optimizer.step()
                self.after_forwardp(ctx, outputs, labels)
            self.after_train(ctx)
        plot_confusion_matrix(GRAPH_DIR / 'lstm.jpg',
                              ctx['true_labels'], ctx['predicted'])
