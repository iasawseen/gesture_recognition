from .core import AbstractModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from sru import SRU
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.utils import shuffle, resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from collections import deque
import functools


class GradReverse2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -2.0


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class Predictor(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(Predictor, self).__init__()
        self.dropout = dropout
        self.hidden_0 = nn.Linear(input_size, 256)
        self.norm = nn.LayerNorm(256)
        self.hidden_1 = nn.Linear(256, output_size)
        self.training = False
        self.init_weights()

    def forward(self, input):
        output = F.dropout(input, p=self.dropout, training=self.training)
        output = self.hidden_0(output)
        output = F.relu(output)
        output = self.norm(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.hidden_1(output)
        return output

    def set_training(self, flag):
        self.training = flag

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.hidden_0.weight)
        torch.nn.init.xavier_normal_(self.hidden_1.weight)


class DomainClassifier(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(DomainClassifier, self).__init__()
        self.dropout = dropout
        self.hidden_0 = nn.Linear(input_size, 256)
        self.norm = nn.LayerNorm(256)
        self.hidden_classes = nn.Linear(256, output_size)

    def forward(self, input):
        output = self.hidden_0(input)
        output = F.relu(output)
        # output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.norm(output)
        output = self.hidden_classes(output)
        return output

    def set_training(self, flag):
        self.training = flag


class FeatureGeneratorSRU(nn.Module):
    def __init__(self, input_size, gru_hidden_size, dropout):
        super(FeatureGeneratorSRU, self).__init__()
        self.dropout = dropout
        self.training = False

        self.rnn = SRU(input_size, gru_hidden_size, num_layers=4,
                       use_tanh=0, use_relu=0, use_selu=1,
                       weight_norm=True,
                       dropout=0.2,
                       bidirectional=True)

        self.norm = nn.LayerNorm(256)

    def forward(self, input):
        output, hidden = self.rnn(input)
        output = output.permute(1, 2, 0)
        features = F.avg_pool1d(output, kernel_size=output.size()[-1])
        features = torch.squeeze(features)
        features = self.norm(features)
        return features

    def set_training(self, flag):
        self.training = flag


class FeatureGeneratorGRU(nn.Module):
    def __init__(self, input_size, gru_hidden_size, dropout):
        super(FeatureGeneratorGRU, self).__init__()
        self.dropout = dropout
        self.training = False

        self.rnn = nn.GRU(input_size, gru_hidden_size, bidirectional=False,
                          dropout=0.2, num_layers=2)

    def forward(self, input):
        output, hidden = self.rnn(input)
        features = output[-1]
        features = torch.squeeze(features)
        return features

    def set_training(self, flag):
        self.training = flag


class FeatureGeneratorCNN(nn.Module):
    def __init__(self, input_size, gru_hidden_size, dropout):
        super(FeatureGeneratorCNN, self).__init__()
        self.dropout = dropout
        self.training = False

        self.cnn_0 = nn.Conv1d(8, 32, 8, dilation=1)
        self.norm_0 = nn.BatchNorm1d(32)

        self.cnn_1 = nn.Conv1d(32, 32, 8, dilation=2)
        self.norm_1 = nn.BatchNorm1d(32)

        self.cnn_2 = nn.Conv1d(32, 32, 8, dilation=4)
        self.norm_2 = nn.BatchNorm1d(32)

        self.cnn_3 = nn.Conv1d(32, 32, 8, dilation=8)
        self.norm_3 = nn.BatchNorm1d(32)

    def forward(self, input):
        input = input.permute(1, 2, 0)

        features = self.cnn_0(input)
        features = F.relu(features)
        features = self.norm_0(features)
        features = F.dropout(features, p=self.dropout, training=self.training)

        features = self.cnn_1(features)
        features = F.relu(features)
        features = self.norm_1(features)
        features = F.dropout(features, p=self.dropout, training=self.training)

        features = self.cnn_2(features)
        features = F.relu(features)
        features = self.norm_2(features)
        features = F.dropout(features, p=self.dropout, training=self.training)

        features = self.cnn_3(features)
        features = F.relu(features)
        features = self.norm_3(features)
        features = F.dropout(features, p=self.dropout, training=self.training)

        features = features.view(features.size()[0], features.size()[1] * features.size()[2])

        return features

    def set_training(self, flag):
        self.training = flag


class TorchNet(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, prediction_size, classes_size, lambd, dropout):
        super(TorchNet, self).__init__()

        self.feature_gen = FeatureGeneratorSRU(input_size, rnn_hidden_size, dropout)
        self.predictor = Predictor(rnn_hidden_size * 2, prediction_size, dropout)
        self.domain_classifier = DomainClassifier(rnn_hidden_size * 2, classes_size, dropout)

        self.lambd = lambd

    def set_lambda(self, lambd):
        self.lambd = lambd

    def set_training(self, flag):
        self.feature_gen.set_training(flag)
        self.predictor.set_training(flag)

    def forward(self, input):
        features = self.feature_gen(input)

        features_grad_inv = grad_reverse(features, self.lambd)

        preds = self.predictor(features)
        classes = self.domain_classifier(features_grad_inv)

        return preds, classes


class AdvancedRNN(AbstractModel):
    def reshape_x(self, x):
        x = np.reshape(x, newshape=(x.shape[0], x.shape[1], x.shape[3]))
        x = np.transpose(x, (1, 0, 2))
        return x

    def __init__(self, x, y, adaptation, batch_size=512):
        x = np.reshape(x, newshape=(x.shape[0], x.shape[1], x.shape[3]))

        y_class = y[:, -1]
        self.adaptation = adaptation
        self.length = x.shape[1]
        self.features = x.shape[2]
        self.pred_length = y.shape[1] - 1
        self.classes_num = int(np.max(y_class) - np.min(y_class)) + 1
        print('preds length:', self.pred_length)
        print('classes_num:', self.classes_num)
        self.batch_size = batch_size
        self.model = None

    def close(self):
        pass

    def build(self):
        pass

    def fit(self, train, val, test, batch_size, num_epochs):
        x_train, y_train = train
        x_val, y_val = val
        x_test, y_test = test

        # x_train = self.reshape_x(x_train)
        x_val = self.reshape_x(x_val)
        x_test = self.reshape_x(x_test)

        print('class_num: ', self.classes_num)
        print(y_train[:, -1].max(), y_val[:, -1].max(), y_test[:, -1].max())
        one_hot = OneHotEncoder(self.classes_num)

        def split_y(y):
            y_class = np.array(y[:, -1], dtype=np.int32)
            y = y[:, :-1]

            return y, y_class

        y_train, y_train_class = split_y(y_train)
        y_val, y_val_class = split_y(y_val)
        y_test, y_test_class = split_y(y_test)

        val_maxes = np.max(y_val, axis=0)
        val_mins = np.min(y_val, axis=0)
        val_ranges = val_maxes - val_mins

        adversarial = self.adaptation
        print('start learning, with ad: ', adversarial)
        # lambdas = np.linspace(0.01, 0.5, num_epochs + 1)
        lambdas = np.linspace(0.01, 0.99, num_epochs + 1)

        best_val_mse = 10000
        epochs_with_no_gain = 0

        self.model = TorchNet(self.features,
                              128,
                              self.pred_length,
                              self.classes_num,
                              lambd=1.0,
                              dropout=0.4).cuda()

        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad,
                                            self.model.parameters()), lr=.001)

        optimizer1 = torch.optim.Adam(filter(lambda x: x.requires_grad,
                                             self.model.parameters()), lr=.001)


        class_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        epoch_mses = []
        best_epoch = 0

        print('adversarial: {}'.format(adversarial))

        for epoch in range(1, num_epochs + 1):
            if adversarial:
                print('lambda: {:.4f}'.format(lambdas[epoch]))
            x_, y_, y_train_class_ = shuffle(x_train, y_train, y_train_class)

            x_ = self.reshape_x(x_)

            start_time = time.time()
            loss_sum = 0
            acc_sum = 0
            loss_qty = 0
            acc_qty = 0

            self.model.train()
            self.model.set_lambda(lambdas[epoch])
            self.model.set_training(True)

            swa = self.model.state_dict()

            for i in range(0, x_.shape[1], batch_size):
                x_batch = x_[:, i: i + batch_size, :]
                y_batch = y_[i: i + batch_size, :]
                y_batch_class = y_train_class_[i: i + batch_size, ]

                y_maxes = np.max(y_batch, axis=0)
                y_mins = np.min(y_batch, axis=0)
                y_ranges = y_maxes - y_mins

                x_batch = torch.from_numpy(x_batch.astype(np.float32)).cuda()
                y_batch = torch.from_numpy(y_batch.astype(np.float32)).cuda()
                y_batch_class = torch.from_numpy(y_batch_class.astype(np.long)).cuda()

                if not adversarial:
                    optimizer.zero_grad()
                    preds, classes = self.model(x_batch)
                    loss = mse_criterion(preds, y_batch)

                    loss.backward()
                    optimizer.step()

                    loss_sum += loss.item()
                    loss_qty += 1
                else:
                    preds, classes = self.model(x_batch)
                    loss = mse_criterion(preds, y_batch)
                    class_loss = class_criterion(classes, y_batch_class)

                    classes = classes.data.cpu().numpy()
                    y_batch_class = y_batch_class.data.cpu().numpy()
                    classes_arg_max = np.argmax(classes, axis=1)
                    acc = (classes_arg_max == y_batch_class).sum() / y_batch_class.shape[0]

                    acc_sum += acc
                    acc_qty += 1

                    optimizer.zero_grad()
                    class_loss.backward()
                    optimizer.step()

                    preds, classes = self.model(x_batch)
                    loss = mse_criterion(preds, y_batch)

                    loss_sum += loss.item()
                    loss_qty += 1

                    optimizer1.zero_grad()
                    loss.backward()
                    optimizer1.step()

            if not adversarial:
                train_mse = loss_sum / loss_qty
                train_acc = 0
            else:
                train_mse = loss_sum / loss_qty
                train_acc = acc_sum / acc_qty

            self.model.eval()
            self.model.set_training(False)

            val_preds = []

            with torch.no_grad():
                for i in range(0, x_val.shape[1], batch_size * 4):
                    x_val_batch = x_val[:, i: i + batch_size * 4, :]
                    x_val_batch = torch.from_numpy(x_val_batch.astype(np.float32)).cuda()
                    val_pred, _ = self.model(x_val_batch)
                    val_preds.append(val_pred.data.cpu().numpy())

            val_preds = np.vstack(val_preds)
            val_mse = mean_squared_error(val_preds, y_val)
            val_mae = mean_absolute_error(val_preds, y_val)
            y_val_norm = y_val / val_ranges
            val_preds_norm = val_preds / val_ranges

            val_nrmse = math.sqrt(mean_squared_error(y_val_norm, val_preds_norm))

            if not adversarial:
                torch.save(self.model.state_dict(), './model_sru_epoch_{}.pth'.format(epoch))
            else:
                torch.save(self.model.state_dict(), './model_sru_ad_epoch_{}.pth'.format(epoch))

            epoch_mses.append((epoch, val_mse))

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                epochs_with_no_gain = 0
            else:
                epochs_with_no_gain += 1

            tolerance = 6

            if epochs_with_no_gain > tolerance:
                print('more than {} epochs with no val gain'.format(tolerance))
                break

            print('epoch {:3d}: tr mse: {:.1f}, tr acc: {:.3f}, '
                  'v mse: {:.1f}, v mae: {:.1f}, v nrmse: {:.3f}, '
                  'Elapsed time {:.1f} s'.format(epoch,
                                                 train_mse,
                                                 train_acc,
                                                 val_mse, val_mae, val_nrmse,
                                                 time.time() - start_time))

        if not adversarial:
            best_model = torch.load('./model_sru_epoch_{}.pth'.format(best_epoch))
        else:
            best_model = torch.load('./model_sru_ad_epoch_{}.pth'.format(best_epoch))

        for name, param in self.model.named_parameters():
            if name in best_model:
                param.data.copy_(best_model[name])

    def predict(self, x):
        return self.predict_on_batch(x)

    def predict_on_batch(self, x, batch_size=2048):
        x = self.reshape_x(x)
        self.model.eval()
        self.model.set_training(False)
        preds = []

        with torch.no_grad():
            for i in range(0, x.shape[1], batch_size):
                x_batch = x[:, i: i + batch_size, :]
                x_batch = torch.from_numpy(x_batch.astype(np.float32)).cuda()
                y_batch, _ = self.model(x_batch)
                preds.append(y_batch.data.cpu().numpy())

        return np.vstack(preds)

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass
