import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from omsignal.experiments import OmExperiment
from omsignal.models.lstm import LSTMModel
from omsignal.utils.transform.basic import ClipAndFlatten

from omsignal.constants import CURR_DIR

class LSTMExperiment(OmExperiment):
    def __init__(self,
    			 exp_file,
    			 device,
    			 model=LSTMModel,
    			 model_params={device, n_layers=1},
    			 optimiser=Adam,
    			 optimiser_params={},
    			 criterion=CrossEntropyLoss,
    			 criterion_params={}):
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

    def visualisation(self, model, dataloader):
        class_correct = list(0. for i in range(32))
        class_total = list(0. for i in range(32))
        confusion = torch.zeros(32,32)
        count = 0
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                hidden = model.init_hidden(inputs.shape[0])
                output, hidden = model(inputs.to(device), hidden,
                    inputs.shape[0])
                _, predicted = torch.max(output.data, 1)
                labels = labels.long()
                labels = labels.view(inputs.shape[0])
                c = (predicted == labels.to(device)).squeeze()
                for i in range(c.shape[0]):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                    confusion[label,predicted[i].item()] += 1
                    count += 1

        plt.figure()
        plt.style.use('ggplot')
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('axes', labelsize=10)
        plt.imshow(confusion/torch.tensor(class_total).view(32,1))
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(CURR_DIR + '/graphs/confusion')

        x = range(1,num_epoch+1)
        plt.figure()
        plt.plot(x, err_train,"sk-", label="Trainset")
        plt.plot(x, err_test,"sr-", label="Testset")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend(fontsize=10)
        for i in range(32):
            if class_total[i]!=0:
                print('Accuracy of %5s : %2d %%' % (
                    i, 100 * class_correct[i]/class_total[i]))
            else:
                print('Accuracy of %5s : %2d %%' % (
                    i, 100 * class_correct[i]))
        plt.savefig(CURR_DIR + '/graphs/error')


    def train(self, dataloader, epochs, validation_dataloader=None, start_epoch=None):
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
                lrd = lr * (1./(1+5 * epoch/epochs))
                self._optimiser()
                self._model.zero_grad()
                hidden = self._model.init_hidden(data.shape(0))
                outputs, hidden = self._model(data, hidden)
                loss = self._criterion(outputs, labels)
                ctx['running_loss'] += loss
                loss.backward()
                self._optimizer.step()
                self.after_forwardp(ctx, outputs, labels)
            self.after_train(ctx)
        self.visualisation(dataloader, self._model)
