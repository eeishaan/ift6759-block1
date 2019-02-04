#!/usr/bin/env python3

import torch


class OmModel():
    '''
    Base class for models
    '''

    def __init__(
            self,
            model_cls,
            model_params,
            optimizer_cls,
            optimizer_params,
            criterion_cls,
            criterion_params,
            model_file):
        self._criterion = criterion_cls(**criterion_params)
        self._model = model_cls(**model_params)
        self._model_file = model_file
        self._optimizer = optimizer_cls(**optimizer_params)
        self._start_epoch = 0

    def after_eval(self, ctx):
        pass

    def after_forwardp(self, ctx, outputs, labels):
        pass

    def after_minibatch_eval(self, ctx, outputs, labels):
        pass

    def after_train(self, ctx):
        self.save_model(ctx)

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
            for _, (data, labels) in dataloader:
                data, labels = self.before_minibatch_eval(ctx, data, labels)
                outputs = self._model(data)
                self.after_minibatch_eval(ctx, outputs, labels)
            self.after_eval(ctx)

    def load_model(self):
        checkpoint = torch.load(self._model_file)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._start_epoch = checkpoint['epoch']

    def save_model(self, ctx):
        save_dict = {
            'epoch': ctx.get('epoch'),
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }
        self.before_save(save_dict)
        torch.save(save_dict, self._model_file)

    def train(self, dataloader, epochs, start_epoch=0):
        start_epoch = start_epoch if start_epoch != 0 else self._start_epoch
        for epoch in range(start_epoch, epochs):
            self._model.train()
            ctx = {
                'epoch': epoch
            }
            self.before_train(ctx)
            for _, (data, labels) in enumerate(dataloader):
                data, labels = self.before_forwardp(ctx, data, labels)
                self._optimizer.zero_grad()
                outputs = self._model(data)
                loss = self._criterion(outputs, labels)
                running_loss += loss
                loss.backward()
                self._optimizer.step()
                self.after_forwardp(ctx, outputs, labels)
            self.after_train(ctx)
