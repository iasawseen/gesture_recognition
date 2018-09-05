from .core import AbstractModel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor

import os
from sys import platform
if not platform == 'win32':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import gpflow
import random


class GaussianProcessFlow(AbstractModel):
    def __init__(self, x, y, adaptation):
        print('x shape: {}'.format(x.shape))
        self.pred_length = y.shape[1] - 1
        self.gp = None
        self.sess = None

    def fit(self, train, val, test, batch_size, num_epochs):

        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        x_train, y_train = train

        print('x_train shape: {}'.format(x_train.shape))

        x_val, y_val = val

        y_train = y_train[:, :-1]
        y_val = y_val[:, :-1]

        print('start training gp')
        kernel = gpflow.kernels.RBF(input_dim=x_train.shape[1], variance=1., lengthscales=0.1)
        self.gp = gpflow.models.GPR(x_train, y_train, kern=kernel)
        self.gp.compile(session=self.sess)
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(self.gp)
        print('complete training gp')

    def predict(self, x):
        preds, _ = self.gp.predict_f(x)
        return preds

    def predict_on_batch(self, x, batch_size=1024):
        preds = []

        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i: i + batch_size, :]
            preds.append(self.predict(x_batch))

        return np.vstack(preds)

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass

    def close(self):
        tf.reset_default_graph()
        self.sess.close()


class GaussianProcess(AbstractModel):
    def __init__(self, x, y, adaptation):
        print('x shape: {}'.format(x.shape))
        self.pred_length = y.shape[1] - 1

        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

        rand = random.randint(0, 2**32 - 1)
        print('seed: {}'.format(rand))

        self.gps = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=7, normalize_y=True,
                                             random_state=rand)
                    for _ in range(self.pred_length)]

    def fit(self, train, val, test, batch_size, num_epochs):
        x_train, y_train = train

        print('x_train shape: {}'.format(x_train.shape))

        x_val, y_val = val

        y_train = y_train[:, :-1]
        y_val = y_val[:, :-1]

        for i, gp in enumerate(self.gps):
            print('start training gp {}'.format(i))
            self.gps[i].fit(x_train, y_train[:, i])
            print('complete training gp {}'.format(i))

    def predict(self, x):
        preds = [self.gps[i].predict(x).reshape((-1, 1)) for i in range(self.pred_length)]
        return np.hstack(preds)

    def predict_on_batch(self, x, batch_size=1024):
        preds = []

        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i: i + batch_size, :]
            preds.append(self.predict(x_batch))

        return np.vstack(preds)

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass

    def close(self):
        pass
