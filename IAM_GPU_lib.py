from numba import cuda
import numpy as np
import operator
import math, numba, cv2


@cuda.jit
def cu_sub_st(source, target, result):
    si, ti = cuda.grid(2)

    if si < source.shape[0] and ti < target.shape[0]:
        for ii in range(0, result.shape[2]):
            result[si, ti, ii] = source[si, ii] - target[ti, ii]
        cuda.syncthreads()

@cuda.jit
def cu_sub_st_threechannel(source, target, result):
    si, ti = cuda.grid(2)

    if si < source.shape[0] and ti < target.shape[0]:
        for ii in range(0, result.shape[2]):
            a = (source[si, ii, :] - target[ti, ii, :])*(source[si, ii, :] - target[ti, ii, :])
            result[si, ti, ii] = math.sqrt(cu_sum_abs_1d(a))
        cuda.syncthreads()


@cuda.jit
def cu_Mult_sub_st(source, target, result, T1w, T2w, patch_size):
    si, ti = cuda.grid(2)

    if si < source.shape[0] and ti < target.shape[0]:
        for ii in range(0, result.shape[2]):
            T_weights = (T1w[si] * T2w[si])
            if T_weights > 0.2 and patch_size > 1:
                result[si, ti, ii] = 0
            else:
                result[si, ti, ii] = source[si, ii] - target[ti, ii]
        cuda.syncthreads()

@cuda.jit
def cu_alg1_penalisation_sub_st(source, target, result, T1w, T2w, patch_size, trsh):
    si, ti = cuda.grid(2)

    if si < source.shape[0] and ti < target.shape[0]:

        avg_val = (T1w[si, 1] + T2w[si, 1]) / 2.0
        for ii in range(0, result.shape[2]):
            if T1w[si,1]>trsh and T2w[si,1]>trsh:
                result[si, ti, ii] = (source[si, ii] - target[ti, ii])*(1-avg_val)

            elif (T1w[si,1]>trsh or T2w[si,1]>trsh)  and patch_size>1:
                result[si, ti, ii] = (source[si, ii] - target[ti, ii]) * (1-avg_val)

            else:
                result[si, ti, ii] = (source[si, ii] - target[ti, ii])

        cuda.syncthreads()

@cuda.jit
def cu_alg2_penalisation_sub_st(source, target, result, T1w, T2w, patch_size, trsh):
    si, ti = cuda.grid(2)

    if si < source.shape[0] and ti < target.shape[0]:

        avg_val = (T1w[si, 1] + T2w[si, 1]) / 2.0
        for ii in range(0, result.shape[2]):
            if T1w[si,1]>trsh and T2w[si,1]>trsh:
                result[si, ti, ii] = (source[si, ii] - target[ti, ii])*(1-avg_val)


            else:
                result[si, ti, ii] = (source[si, ii] - target[ti, ii])

        cuda.syncthreads()




@cuda.jit
def cu_max_mean_abs(inputs, results):
    si, ti = cuda.grid(2)

    if si < results.shape[0] and ti < results.shape[1]:
        results[si, ti, 0] = cu_max_abs_1d(inputs[si, ti, :])
        results[si, ti, 1] = cu_mean_abs_1d(inputs[si, ti, :])
        cuda.syncthreads()
    cuda.syncthreads()


@cuda.jit
def cu_max_mean_flair(FLAIR_source, FLAIR_max, FLAIR_mean):
    fi = cuda.grid(1)

    flair_tsh = 400
    if fi < FLAIR_max.shape[0]:
        FLAIR_max[fi] = cu_max_trsh_abs_1d(FLAIR_source[fi, :], flair_tsh)
        FLAIR_mean[fi] = cu_mean_abs_1d(FLAIR_source[fi, :])
        cuda.syncthreads()

@cuda.jit
def cu_distances(inputs, flag, outputs,  alpha):
    si, ti = cuda.grid(2)
    beta = 0.5

    if si < outputs.shape[0] and ti < outputs.shape[1]:
        outputs[si, ti] = flag[si] * (alpha * inputs[si, ti, 0] + (1 - alpha) * inputs[si, ti, 1])
        cuda.syncthreads()
    cuda.syncthreads()


@cuda.jit
def cu_sub_sqr_st(source, target, result):
    si, ti = cuda.grid(2)
    if si < source.shape[0] and ti < target.shape[0]:
        result[si, ti] = 0
        for ii in range(0, source.shape[1]):
            result[si, ti] += (source[si, ii] - target[ti, ii]) * (source[si, ii] - target[ti, ii])
        result[si, ti] = math.sqrt(result[si, ti])
        result[si, ti] /= source.shape[1]
        cuda.syncthreads()


@cuda.jit(device=True)
def cu_max_trsh_abs_1d(array, trsh):
    temp = -9999
    for i in range(0, array.shape[0]):
        if array[i] > temp:
            temp = array[i]
    if temp < trsh: temp = -9999
    return temp


@cuda.jit(device=True)
def cu_max_abs_1d(array):
    temp = -9999
    for i in range(0, array.shape[0]):
        if array[i] > temp:
            temp = array[i]
    if temp < 0: temp *= -1
    return temp


@cuda.jit(device=True)
def cu_mean_abs_1d(array):
    temp = 0
    for i in range(array.shape[0]):
        temp += array[i]
    if temp < 0: temp *= -1
    return temp / array.size


@cuda.jit(device=True)
def cu_mean_sqrt_abs_1d(array):
    temp = 0
    for i in range(array.shape[0]):
        temp += array[i]
    if temp < 0:
        temp *= -1
    else:
        temp = math.sqrt(temp)
    return temp / array.size


@cuda.jit(device=True)
def cu_sum_abs_1d(array):
    temp = 0
    for i in range(array.shape[0]):
        temp += array[i]
    return temp


@cuda.jit
def cu_sort_distance(array):
    i = cuda.grid(1)

    if i < array.shape[0]:
        for passnum in range(len(array[i, :]) - 1, 0, -1):
            for j in range(passnum):
                if array[i, j] > array[i, j + 1]:
                    temp = array[i, j]
                    array[i, j] = array[i, j + 1]
                    array[i, j + 1] = temp
    cuda.syncthreads()


@cuda.jit
def cu_mse_st(sub_sqr_result, FLAIR_source, FLAIR_max, FLAIR_mean, result):
    si, ti = cuda.grid(2)
    if si < result.shape[0] and ti < result.shape[1]:
        if FLAIR_max[si] < 0:
            result[si, ti] = 0
        else:
            result[si, ti] = sub_sqr_result[si, ti]# * (0.7 * FLAIR_max[si] + 0.3 * FLAIR_mean[si])
        cuda.syncthreads()


@cuda.jit
def cu_age_value(arrays, results):
    i = cuda.grid(1)

    if i < results.shape[0]:
        results[i] = cu_mean_abs_1d(arrays[i, :])
        cuda.syncthreads()
    cuda.syncthreads()


def kernel_sphere(vol):
    if vol == 1 or vol == 2:
        return np.array([[1]])
    elif vol == 3 or vol == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif vol == 5 or vol == 6:
        return np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
    elif vol == 7 or vol == 8:
        return np.array([[0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0]])
    elif vol == 9 or vol == 10:
        return np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0]])
    elif vol == 11 or vol > 11:
        return np.array(
            [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])


def get_area(x_c, y_c, x_dist, y_dist, img, ch):
    if ch==3:
        [ch_num, x_len, y_len] = img.shape
    else:
        [x_len, y_len] = img.shape
    even_x = np.mod(x_dist, 2) - 2
    even_y = np.mod(y_dist, 2) - 2

    x_top = x_c - np.floor(x_dist / 2) - (even_x + 1)
    x_low = x_c + np.floor(x_dist / 2)
    y_left = y_c - np.floor(y_dist / 2) - (even_y + 1)
    y_rght = y_c + np.floor(y_dist / 2)

    if x_top < 0: x_top = 0
    if x_low >= x_len: x_low = x_len
    if y_left < 0: y_left = 0
    if y_rght >= y_len: y_rght = y_len


    if ch==3:
        return img[:,int(x_top):int(x_low + 1), int(y_left):int(y_rght + 1)]
    else:
        return img[int(x_top):int(x_low + 1), int(y_left):int(y_rght + 1)]

