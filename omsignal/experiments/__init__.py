#!/usr/bin/env python3

import torch


class OmExperiment():
    '''
    Base class for experiments
    '''

    def __init__(
            self,
            model_cls,
            model_params,
            optimizer_cls,
            optimizer_params,
            criterion_cls,
            criterion_params,
            experiment_file,
            device):
        self._criterion = criterion_cls(**criterion_params)
        self._model = model_cls(**model_params)
        self._device = device
        self._model.to(device)
        self._experiment_file = experiment_file
        self._optimizer = optimizer_cls(
            self._model.parameters(), **optimizer_params)
        self._start_epoch = 0

    def after_eval(self, ctx):
        pass

    def after_forwardp(self, ctx, outputs, labels):
        pass

    def after_minibatch_eval(self, ctx, outputs, labels):
        pass

    def after_train(self, ctx):
        self.save_experiment(ctx)

    def before_minibatch_eval(self, ctx, data, labels):
        return data, labels

    def before_eval(self, ctx):
        pass

    def before_forwardp(self, ctx, data, labels):
        return data, labels

    def before_save(self, save_dict):
        pass

    def before_train(self, ctx):
        pass

    def eval(self, dataloader):
        self._model.eval()
        with torch.no_grad():
            ctx = {}
            self.before_eval(ctx)
            for _, (data, labels) in enumerate(dataloader):
                data, labels = data.to(self._device), labels.to(self._device)
                data, labels = self.before_minibatch_eval(ctx, data, labels)
                outputs = self._model(data)
                self.after_minibatch_eval(ctx, outputs, labels)
            self.after_eval(ctx)

    def load_experiment(self):
        checkpoint = torch.load(self._experiment_file)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._start_epoch = checkpoint['epoch']

    def save_experiment(self, ctx):
        save_dict = {
            'epoch': ctx.get('epoch'),
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }
        self.before_save(save_dict)
        torch.save(save_dict, self._experiment_file)

    def train(self, dataloader, epochs, validation_dataloader=None, start_epoch=None):
        start_epoch = start_epoch if start_epoch is not None else self._start_epoch
        for epoch in range(start_epoch, epochs):
            self._model.train()
            ctx = {
                'epoch': epoch,
                'running_loss': 0,
                'val_loader': validation_dataloader
            }
            self.before_train(ctx)
            for _, (data, labels) in enumerate(dataloader):
                data, labels = data.to(self._device), labels.to(self._device)
                data, labels = self.before_forwardp(ctx, data, labels)
                self._optimizer.zero_grad()
                outputs = self._model(data)
                loss = self._criterion(outputs, labels)
                ctx['running_loss'] += loss
                loss.backward()
                self._optimizer.step()
                self.after_forwardp(ctx, outputs, labels)
            self.after_train(ctx)
