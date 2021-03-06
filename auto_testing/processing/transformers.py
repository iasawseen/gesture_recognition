import numpy as np
import random
from scipy.signal import butter, lfilter, freqz


def average_filter(array, window, mode='same'):
    columns = []
    for j in range(array.shape[1]):
        column = array[:, j]
        column = np.convolve(column, np.ones((window,)) / window, mode=mode)
        column = np.reshape(column, newshape=(column.shape[0], 1))
        columns.append(column)
    return np.concatenate(columns, axis=1)


def rectify_x(xy):
    x, y = xy
    x_rectified = np.abs(x)
    return x_rectified, y


def average_filter_emgs_angles(xy, emgs_window=4, angles_window=4):
    x, y = xy
    x_filtered = average_filter(x, window=emgs_window)
    y_filtered = average_filter(y, window=angles_window)
    return x_filtered, y_filtered


def average_filter_angles(xy, window):
    return average_filter_emgs_angles(xy, emgs_window=1, angles_window=window)


def add_window_to_xy(xy, window):
    x, (y, y_label) = xy
    x_windowed = list()
    y_windowed = list()
    for i in range(0, x.shape[0] - window):
        x_indices = list(range(i, i + window))
        x_windowed.append(x[x_indices, :])
        y_windowed.append(y[x_indices[-1], :])

    return np.array(x_windowed), (np.array(y_windowed), y_label)


def reshape_x_for_dilated(xy):
    x, y = xy
    x_reshaped = np.reshape(x, newshape=(x.shape[0], x.shape[1], 1, x.shape[2]))
    return x_reshaped, y


def compact_iterable(iterable, ratio):
    result = []
    offset = ratio

    for el in iterable:
        length = el[0].shape[0]
        indices_to_save = list(range(0, length, offset))
        indices_to_delete = list(set(range(length)) - set(indices_to_save))
        result.append((np.delete(el[0], indices_to_delete, axis=0),
                       np.delete(el[1], indices_to_delete, axis=0)))

    return result


def process_iterable(iterable, func):
    results = [func(el) for el in iterable]
    return results


def mimic_old_y(xy):
    x, (y, y_label) = xy
    indices = [3, *range(6, 8),
               8, *range(11, 13),
               13, *range(16, 18),
               18, *range(21, 23),
               23, *range(26, 28)]
    y_ = y[:, indices]
    return x, (y_, y_label)


def filter_angles(xy):
    x, (y, y_label) = xy
    indices = [*range(3),
               3, *range(6, 8),
               8, *range(11, 13),
               13, *range(16, 18),
               18, *range(21, 23),
               23, *range(26, 28)]
    y_ = y[:, indices]
    return x, (y_, y_label)


def shift(xy, shift_size=16):
    x, (y, y_label) = xy
    x_shifted = x[:-shift_size, :]
    y_shifted = y[shift_size:, :]
    return x_shifted, (y_shifted, y_label)


def compact(xy, ratio):
    x, (y, y_label) = xy
    x = x[::ratio, :].copy()
    y = y[::ratio, :].copy()

    return x, (y, y_label)


def low_pass_filter(xy):
    def _low_pass_filter(data, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        datas = [data[:, i] for i in range(data.shape[1])]
        data_low_passes = [lfilter(b, a, data_) for data_ in datas]
        x_low_pass = np.vstack(data_low_passes).T
        return x_low_pass

    x, (y, y_label) = xy

    order = 6
    fs = 200
    x_cutoff = 10
    y_cutoff = 4
    x_ = _low_pass_filter(x, x_cutoff, fs, order)
    y_ = _low_pass_filter(y, y_cutoff, fs, order)
    return x_, (y_, y_label)


def merge_xys(xys):
    result = [[[], []] for _ in range(len(xys[0]))]

    for xy in xys:
        for index, xy_tuple in enumerate(xy):
            result[index][0].append(xy_tuple[0])
            result[index][1].append(xy_tuple[1])

    merged = [(np.concatenate(xy_tuple[0]),
               np.concatenate(xy_tuple[1])) for xy_tuple in result]
    return merged


def merge_xys_test(xys):
    result = [[], []]

    for xy in xys:
        result[0].append(xy[0])
        y, y_label = xy[1]
        y_label_arr = np.full(shape=y.shape[0], fill_value=y_label).reshape((-1, 1))
        result[1].append(np.hstack((y, y_label_arr)))

    X = np.vstack(result[0])
    y = np.vstack(result[1])
    return X, y


def split_by_chunks(xy, val_test_size=0.25, chunks=20, overlapping=32):
    x, (y, y_label) = xy
    chunk_size = int(x.shape[0] * val_test_size / chunks)
    offset = int(x.shape[0] / chunks)
    borders = []
    x_vals = []
    y_vals = []
    x_tests = []
    y_tests = []

    for i in range(0, x.shape[0], offset):
        start_index = random.randint(i, i + offset - chunk_size)
        end_index = start_index + chunk_size
        borders.append((start_index, end_index))

    shuffled_borders = list(borders)
    random.shuffle(shuffled_borders)

    for border_pair in shuffled_borders[: len(shuffled_borders) // 2]:
        x_vals.append(x[border_pair[0] + overlapping: border_pair[1] - overlapping, :])
        y_vals.append(y[border_pair[0] + overlapping: border_pair[1] - overlapping, :])

    for border_pair in shuffled_borders[len(shuffled_borders) // 2:]:
        x_tests.append(x[border_pair[0] + overlapping: border_pair[1] - overlapping, :])
        y_tests.append(y[border_pair[0] + overlapping: border_pair[1] - overlapping, :])

    x_val = np.concatenate(x_vals)
    y_val = np.concatenate(y_vals)
    x_test = np.concatenate(x_tests)
    y_test = np.concatenate(y_tests)

    indices_to_delete = []

    for border_pair in borders:
        indices_to_delete.extend(list(range(border_pair[0], border_pair[1])))
    x_train = np.delete(x, indices_to_delete, axis=0)
    y_train = np.delete(y, indices_to_delete, axis=0)

    y_train_label = np.full(shape=y_train.shape[0], fill_value=y_label).reshape((-1, 1))
    y_val_label = np.full(shape=y_val.shape[0], fill_value=y_label).reshape((-1, 1))
    y_test_label = np.full(shape=y_test.shape[0], fill_value=y_label).reshape((-1, 1))

    y_train = np.concatenate((y_train, y_train_label), axis=1)
    y_val = np.concatenate((y_val, y_val_label), axis=1)
    y_test = np.concatenate((y_test, y_test_label), axis=1)

    return (x_train, y_train), \
           (x_val, y_val), \
           (x_test, y_test)
