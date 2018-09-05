from .core import AbstractModel
from tensorflow.python.keras.layers import Input, Dense, Dropout, BatchNormalization, GRU, Flatten
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import regularizers, optimizers
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.utils import shuffle
import time
import os
from sys import platform
if not platform == 'win32':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import math

from tensorflow.contrib.rnn import MultiRNNCell, GRUCell, LayerNormBasicLSTMCell, DropoutWrapper


class AdvancedRNN(AbstractModel):
    def reshape_x(self, x):
        return np.reshape(x, newshape=(x.shape[0], x.shape[1], x.shape[3]))

    def _last_relevant(self, output, length, units):
        with tf.name_scope("last_relevant"):
            batch_size = tf.shape(output)[0]
            index = tf.range(0, batch_size) * length + (length - 1)
            flat = tf.reshape(output, [-1, units])
            return tf.gather(flat, index)

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
        self.hiddens = [128, 128]
        self.batch_size = batch_size
        self.keep_prob = None
        self.layers = len(self.hiddens)
        self.predictions = None
        self.reg_loss = None
        self.class_loss = None
        self.class_preds = None
        self.predictor_loss = None
        self.classifier_loss = None
        self.sum_loss = None
        self.train_regressor_op = None
        self.train_predictor_op = None
        self.train_classifier_op = None
        self.train_classifier_op_all = None
        self.train_sum = None
        self.acc = None
        self.acc_op = None
        self.mae = None
        self.data = None
        self.target = None
        self.class_target = None
        self.lr = None
        self.norm = None
        self._lambda = None
        self.ranges = None
        self.initial_state = None
        self.outputs = None
        self.state = None
        self.update_state = None
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.create_placeholders()
        self.build()

    def close(self):
        tf.reset_default_graph()
        self.sess.close()

    def create_placeholders(self):
        self.data = tf.placeholder(tf.float32,
                                   [None, self.length, self.features])
        self.target = tf.placeholder(tf.float32, [None, self.pred_length])
        self.class_target = tf.placeholder(tf.int32, [None, self.classes_num])
        self.ranges = tf.placeholder(tf.float32, [self.pred_length])
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.norm = tf.placeholder(tf.bool)
        self._lambda = tf.placeholder(tf.float32, shape=())

    def build(self):
        with tf.name_scope('recurrent_layers'):
            rnn_layers = [DropoutWrapper(LayerNormBasicLSTMCell(units, dropout_keep_prob=self.keep_prob),
                                         output_keep_prob=self.keep_prob)
                          for units in self.hiddens]

            multi_rnn_cell = MultiRNNCell(rnn_layers)
            outputs, _ = tf.nn.dynamic_rnn(multi_rnn_cell, self.data, dtype=tf.float32)

            x = self._last_relevant(outputs, self.length, self.hiddens[-1])

        features = tf.nn.dropout(x, keep_prob=self.keep_prob)

        x = slim.fully_connected(features, 256, activation_fn=tf.nn.relu, scope='fc1')

        x = tf.contrib.layers.layer_norm(x)
        x = tf.nn.dropout(x, keep_prob=self.keep_prob)

        self.predictions = slim.fully_connected(x, self.pred_length,
                                                activation_fn=None, scope='final')

        self.reg_loss = tf.losses.mean_squared_error(self.target, self.predictions)

        inv_concat = tf.scalar_mul(-self._lambda, features)
        inverse_gradient_layer = inv_concat + tf.stop_gradient(features - inv_concat)

        x_ad = slim.fully_connected(inverse_gradient_layer, 256, activation_fn=tf.nn.relu, scope='fc1_ad')
        x_ad = tf.contrib.layers.layer_norm(x_ad)

        self.class_preds = slim.fully_connected(x_ad, self.classes_num, activation_fn=None)
        self.acc, self.acc_op = tf.metrics.accuracy(labels=tf.argmax(self.class_target, 1),
                                                    predictions=tf.argmax(self.class_preds, 1))

        self.classifier_loss = tf.losses.softmax_cross_entropy(self.class_target, self.class_preds)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.train_regressor_op = tf.train.AdamOptimizer(learning_rate=self.lr).\
                minimize(self.reg_loss)
            self.train_classifier_op_all = tf.train.AdamOptimizer(learning_rate=self.lr).\
                minimize(self.classifier_loss)

        diff = tf.abs(tf.subtract(self.target, self.predictions))
        self.mae = tf.reduce_mean(diff)

    def fit(self, train, val, test, batch_size, num_epochs):
        x_train, y_train = train
        x_val, y_val = val
        x_test, y_test = test

        x_train = self.reshape_x(x_train)
        x_val = self.reshape_x(x_val)
        x_test = self.reshape_x(x_test)

        print('class_num: ', self.classes_num)
        print(y_train[:, -1].max(), y_val[:, -1].max(), y_test[:, -1].max())
        one_hot = OneHotEncoder(self.classes_num)

        def split_y(y):
            y_class = np.array(y[:, -1], dtype=np.int32)
            y = y[:, :-1]
            y_class_one_hot = one_hot.fit_transform(y_class.reshape((-1, 1))).todense()
            return y, y_class_one_hot

        y_train, y_train_class = split_y(y_train)
        y_val, y_val_class = split_y(y_val)
        y_test, y_test_class = split_y(y_test)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        def get_lr(epoch_index):
            if epoch_index < 64:
                return .001
            if epoch_index < 96:
                return .0007
            return .0005

        adversarial = self.adaptation
        print('start learning, with ad: ', adversarial)
        lambdas = np.linspace(0.1, 0.5, num_epochs)

        best_val_mse = 10000
        val_mse_eps = 10
        epochs_with_no_gain = 0

        for epoch in range(num_epochs):
            print('lambda: {:.4f}'.format(lambdas[epoch]))
            # if self.adaptation:
            #     adversarial = False if epoch % 2 == 0 else True
            x_, y_, y_train_class_ = shuffle(x_train, y_train, y_train_class)

            start_time = time.time()
            loss_sum = 0
            acc_sum = 0
            loss_qty = 0
            acc_qty = 0

            for i in range(0, x_.shape[0], batch_size):
                # print('hi')
                x_batch = x_[i: i + batch_size, :, :]
                y_batch = y_[i: i + batch_size, :]
                y_batch_class = y_train_class_[i: i + batch_size, ]

                y_maxes = np.max(y_batch, axis=0)
                y_mins = np.min(y_batch, axis=0)
                y_ranges = y_maxes - y_mins

                fd = {self.data: x_batch,
                      self.target: y_batch,
                      self.class_target: y_batch_class,
                      self.ranges: y_ranges,
                      self.lr: get_lr(epoch),
                      self.keep_prob: 0.4,
                      self.norm: 1,
                      self._lambda: lambdas[epoch]}

                if not adversarial:
                    loss, _ = self.sess.run([self.reg_loss,
                                             self.train_regressor_op], feed_dict=fd)
                    loss_sum += loss
                    loss_qty += 1
                else:
                    acc, _, loss, _, _ = self.sess.run([self.acc,
                                                        self.acc_op,
                                                        self.reg_loss,
                                                        self.train_regressor_op,
                                                        self.train_classifier_op_all], feed_dict=fd)
                    loss_sum += loss
                    loss_qty += 1

                    acc_sum += acc
                    acc_qty += 1

            if not adversarial:
                train_mse = loss_sum / loss_qty
                train_acc = 0
            else:
                train_mse = loss_sum / loss_qty
                train_acc = acc_sum / acc_qty

            val_maxes = np.max(y_val, axis=0)
            val_mins = np.min(y_val, axis=0)
            val_ranges = val_maxes - val_mins

            fd_val = {self.data: x_val,
                      self.target: y_val,
                      self.class_target: y_val_class,
                      self.ranges: val_ranges,
                      self.keep_prob: 1.0,
                      self.norm: 0}

            val_mse, val_mae, val_preds = self.sess.run([self.reg_loss, self.mae, self.predictions],
                                                        feed_dict=fd_val)

            y_val_norm = y_val / val_ranges
            val_preds_norm = val_preds / val_ranges

            val_nrmse = math.sqrt(mean_squared_error(y_val_norm, val_preds_norm))

            if val_mse < best_val_mse:
                best_val_mse = val_mse
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

    def predict(self, x):
        x = self.reshape_x(x)

        return self.sess.run(self.predictions, {self.data: x, self.keep_prob: 1.0, self.norm: 0})

    def predict_on_batch(self, x, batch_size=512):
        x = self.reshape_x(x)

        preds = []

        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i: i + batch_size, :, :]
            y_batch = self.sess.run(self.predictions, {self.data: x_batch,
                                                       self.keep_prob: 1.0,
                                                       self.norm: 0})
            preds.append(y_batch)

        return np.vstack(preds)

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass


class ArmNet(AbstractModel):
    def __init__(self, length, features, pred_length):
        self.length = length
        self.features = features
        self.pred_length = pred_length
        self.hiddens = [128, 128, 128]
        self.keep_prob = None
        self.layers = len(self.hiddens)
        self.predictions = None
        self.loss = None
        self.train_op = None
        self.accuracy = None
        self.sess = None
        self.data = None
        self.target = None
        self.lr = None
        self.norm = None
        self.create_placeholders()
        self.build()

    def create_placeholders(self):
        self.data = tf.placeholder(tf.float32,
                                   [None, self.length, self.features])
        self.target = tf.placeholder(tf.float32, [None, self.pred_length])
        self.lr = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.norm = tf.placeholder(tf.bool)

    def add_normalizer(self, input_tensor, drop=False):
        tensor = tf.contrib.layers.batch_norm(input_tensor, center=True, scale=True, is_training=self.norm)

        if not drop:
            return tensor
        return tf.nn.dropout(tensor, keep_prob=self.keep_prob)

    def build(self):
        with tf.name_scope('recurrent_layers'):
            rnn_layers = [DropoutWrapper(GRUCell(units), output_keep_prob=self.keep_prob)
                          for units in self.hiddens]

            multi_rnn_cell = MultiRNNCell(rnn_layers)
            outputs, _ = tf.nn.dynamic_rnn(multi_rnn_cell, self.data, dtype=tf.float32)

            prev = self._last_relevant(outputs, self.length)

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.keras.initializers.glorot_uniform()):

            prev = self.add_normalizer(prev, drop=True)
            prev = slim.fully_connected(prev, 512, activation_fn=tf.nn.tanh, scope='fc1')
            prev = self.add_normalizer(prev, drop=True)
            prev = slim.fully_connected(prev, 512, activation_fn=tf.nn.tanh, scope='fc2')
            prev = self.add_normalizer(prev, drop=True)
            self.predictions = slim.fully_connected(prev, self.pred_length,
                                                    activation_fn=None, scope='final')

        self.loss = tf.losses.mean_squared_error(self.target, self.predictions)
        self.train_op = tf.contrib.opt.NadamOptimizer(learning_rate=self.lr).minimize(self.loss)
        # self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        diff = tf.abs(tf.subtract(self.target, self.predictions))
        self.accuracy = tf.reduce_mean(diff)

    def fit(self, x, y, x_val, y_val, batch_size, epochs):
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        def get_lr(epoch_index):
            if epoch_index < 128:
                return .001
            if epoch_index < 256:
                return .0007
            return .0005

        def get_norm(epoch_index):
            if epoch_index < 16:
                return True
            return False

        def get_norm_mask(offset=5, warm_up=16, end=16):
            mask = np.zeros(shape=(epochs,), dtype=np.bool)
            indices = np.array(list(range(0, epochs, offset)))
            mask[indices] = True
            mask[:warm_up] = True
            mask[-end:] = False
            return mask

        norm_mask = get_norm_mask()

        for epoch in range(epochs):
            x_, y_ = shuffle(x, y)

            start_time = time.time()
            for i in range(0, x_.shape[0], batch_size):
                fd = {self.data: x_[i: i + batch_size, :, :],
                      self.target: y_[i: i + batch_size, :],
                      self.lr: get_lr(epoch), self.keep_prob: 0.7,
                      self.norm: norm_mask[epoch]}
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=fd)
                print('\r', 'batch {} was processed with loss: {:.3f}'.format(i // batch_size, loss), end='')
            print('\r', end='')

            fd = {self.data: x_val,
                  self.target: y_val,
                  self.keep_prob: 1.0,
                  self.norm: False}
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=fd)
            print('epoch {:4d}: val loss: {:.3f}, val abs diff: {:.3f}. '
                  'Elapsed time {:.1f} s'.format(epoch, loss, acc, time.time() - start_time))

    def predict(self, x):
        return self.sess.run(self.predictions, {self.data: x, self.keep_prob: 1.0, self.norm: False})

    def _last_relevant(self, output, length):
        with tf.name_scope("last_relevant"):
            batch_size = tf.shape(output)[0]
            index = tf.range(0, batch_size) * length + (length - 1)
            flat = tf.reshape(output, [-1, self.hiddens[-1]])
            return tf.gather(flat, index)


class StandardRNN(AbstractModel):
    def reshape_x(self, x):
        return np.reshape(x, newshape=(x.shape[0], x.shape[1], x.shape[3]))

    def __init__(self, x, y, adaptation):
        x = np.reshape(x, newshape=(x.shape[0], x.shape[1], x.shape[3]))
        assert len(x.shape) == 3
        window_size = x.shape[1]
        features = x.shape[2]
        pred_length = y.shape[1] - 1
        self.model = self.get_rnn_model(window_size, features, pred_length)

    def fit(self, train, val, test, batch_size, num_epochs):
        x_train, y_train = train
        x_val, y_val = val

        x_train = self.reshape_x(x_train)
        x_val = self.reshape_x(x_val)

        y_train = y_train[:, :-1]
        y_val = y_val[:, :-1]

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, verbose=True,
                                      patience=10, min_lr=0.0001)

        self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=12,
                       shuffle=True, validation_data=(x_val, y_val), callbacks=[reduce_lr])

    def predict(self, x):
        x = self.reshape_x(x)
        return self.model.predict(x, batch_size=128)

    def predict_on_batch(self, x, batch_size=512):
        x = self.reshape_x(x)
        return self.model.predict(x, batch_size=batch_size)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = load_model(file_path)

    @staticmethod
    def get_rnn_model(window_size, features, pred_length):
        inputs = Input(shape=(window_size, features))

        x = GRU(128, kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l2(0.01),
                return_sequences=True,
                input_shape=(window_size, features))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = GRU(128, kernel_regularizer=regularizers.l2(0.01),
                bias_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = Dense(256, activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  bias_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        preds = Dense(pred_length, activation='linear',
                      kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01))(x)
        model = Model(inputs=inputs, outputs=preds)
        optimer = optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimer, loss='mse', metrics=['mae'])

        return model

    def close(self):
        pass

