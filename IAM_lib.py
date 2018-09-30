import matplotlib

matplotlib.use('Agg')

from PIL import Image
from numba import cuda
from timeit import default_timer as timer
from matplotlib import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IAM_GPU_lib import *

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import skimage.morphology as skimorph
import skimage.filters as skifilters
import skimage.color as color
import scipy.ndimage.morphology as scimorph

import math, numba, cv2, csv, gc, time
import os, errno, sys, shutil

# Turn interactive plotting off
plt.ioff()


def initial_check(output_filedir, csv_filename, patch_size, blending_weights, delete_intermediary, num_sample):
    ## Check availability of input files and output path
    if output_filedir == "" or csv_filename == "":
        raise ValueError("Please set output folder's name and CSV data filename. See: help(iam_lots_gpu)")
        return 0

    ## Check compatibility between 'patch_size' and 'blending_weights'
    if len(patch_size) != len(blending_weights):
        raise ValueError(
            "Lengths of 'patch_size' and 'blending_weights' variables are not the same. Length of 'patch_size' is " + str(
                len(patch_size)) + ", while 'blending_weights' is " + str(len(blending_weights)) + ".")
        return 0

    ## If intermediary files to be deleted, don't even try to save JPEGs
    if delete_intermediary:
        save_jpeg = False

    ''' Set number of mean samples automatically '''
    ''' num_samples_all = [64, 128, 256, 512, 1024, 2048] '''
    ''' num_mean_samples_all = [16, 32, 32, 64, 128, 128] '''
    num_samples_all = num_sample
    num_mean_samples_all = []
    for sample in num_samples_all:
        if sample == 64:
            num_mean_samples_all.append(16)
        elif sample == 128:
            num_mean_samples_all.append(32)
        elif sample == 256:
            num_mean_samples_all.append(32)
        elif sample == 512:
            num_mean_samples_all.append(64)
        elif sample == 1024:
            num_mean_samples_all.append(128)
        elif sample == 2048:
            num_mean_samples_all.append(128)
        else:
            raise ValueError("Number of samples must be either 64, 128, 256, 512, 1024 or 2048!")
            return 0

    return num_mean_samples_all, num_samples_all


def data_load(dirOutput, data, input_dir, bin_tresh, colourchannel_approach, penalisation_approach):
    print('--\nNow processing data: ' + data)

    ''' Create output folder(s) '''
    dirOutData = dirOutput + '/' + data
    dirOutDataCom = dirOutput + '/' + data + '/IAM_combined_python/'
    dirOutDataFin = dirOutput + '/' + data + '/IAM_GPU_nifti_python/'
    dirOutDataPatch = dirOutput + '/' + data + '/IAM_combined_python/Patch/'

    if not os.path.exists(dirOutData):
        try:
            os.makedirs(dirOutData)
            os.makedirs(dirOutDataCom)
            os.makedirs(dirOutDataFin)
            os.makedirs(dirOutDataPatch)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    try:
        if penalisation_approach:
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/Alg1_Result/')
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/Alg1_Result/Image/')
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/Alg1_Result/Data/')
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/Alg2_Result/')
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/Alg2_Result/Image/')
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/Alg2_Result/Data/')
        if colourchannel_approach:
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/ColorChannel_Result/')
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/ColorChannel_Result/Image/')
            os.makedirs(dirOutput + '/' + data + '/IAM_combined_python/ColorChannel_Result/Data/')

    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    niiDir = input_dir + '/' + data + '/'
    FLAIR_nii = nib.load(niiDir + 'FLAIR.nii.gz')  # nib.load(row[2])
    icv_nii = nib.load(niiDir + 'ICV.nii.gz')  # nib.load(row[3])
    csf_nii = nib.load(niiDir + 'CSF.nii.gz')  # nib.load(row[4])
    T1w_nii = nib.load(niiDir + 'T1W.nii.gz')  # nib.load(row[6])
    GT_nii = nib.load(niiDir + 'WMH_label.nii.gz')  # nib.load(row[7])

    FLAIR_data = np.squeeze(FLAIR_nii.get_data())
    icv_data = np.squeeze(icv_nii.get_data())
    csf_data = np.squeeze(csf_nii.get_data())
    T1w_data = np.squeeze(T1w_nii.get_data())
    gt_data = np.squeeze(GT_nii.get_data())

    T2w_path = niiDir + 'T2W.nii.gz'

    if os.path.exists(T2w_path):
        T2w_nii = nib.load(T2w_path)
        T2w_data = np.squeeze(T2w_nii.get_data())
    else:
        print('T2w data of %s does not exist'%data)


    ''' Make sure that all brain masks are binary masks, not probability '''
    icv_data[icv_data > bin_tresh] = 1
    icv_data[icv_data <= bin_tresh] = 0
    csf_data[csf_data > bin_tresh] = 1
    csf_data[csf_data <= bin_tresh] = 0
    gt_data[gt_data > bin_tresh] = 1
    gt_data[gt_data <= bin_tresh] = 0

    ''' Read and open NAWM data if available '''
    nawm_nii = nib.load(niiDir + 'NAWM.nii.gz')
    print(niiDir)
    nawm_data = np.squeeze(nawm_nii.get_data())
    print('NAWM:', len(nawm_data > 0))
    del csf_nii, icv_nii, T1w_nii, GT_nii  # Free memory

    return [FLAIR_data, T1w_data, T2w_data], icv_data, csf_data, nawm_data, gt_data, FLAIR_nii


def preprocessing(icv_data, csf_data, nawm_data):
    ''' ICV Erosion '''
    original_icv_data = icv_data.astype(bool)
    original_csf_data = csf_data.astype(bool)
    for ii in range(0, icv_data.shape[2]):
        kernel = kernel_sphere(3)
        csf_data[:, :, ii] = skimorph.dilation(csf_data[:, :, ii], kernel)
        kernel = kernel_sphere(5)
        icv_data[:, :, ii] = skimorph.erosion(icv_data[:, :, ii], kernel)
        kernel = kernel_sphere(5)
        nawm_data[:, :, ii] = skimorph.dilation(nawm_data[:, :, ii], kernel)

    csf_data = csf_data.astype(bool)
    csf_data = ~csf_data
    csf_data = csf_data.astype(float)

    nawm_data = nawm_data.astype(bool)
    nawm_data = ~nawm_data
    nawm_data = nawm_data.astype(int)

    return original_icv_data, original_csf_data, nawm_data, csf_data, icv_data


def brain_vol_trsh(vol_slice, patch_size):
    FLAIR_TRSH = 0.50
    T1w_TRSH = 0.50
    if patch_size == 1:
        if vol_slice < 0.010:
            FLAIR_TRSH = T1w_TRSH = 0
        elif vol_slice < 0.035:
            FLAIR_TRSH = 0.15
            T1w_TRSH = 0.25
        elif vol_slice < 0.070 and vol_slice >= 0.035:
            FLAIR_TRSH = 0.60
            T1w_TRSH = 0.75
        elif vol_slice >= 0.070:
            FLAIR_TRSH = 0.80
            T1w_TRSH = 0.90

    elif patch_size == 2:
        if vol_slice < 0.010:
            FLAIR_TRSH = T1w_TRSH = 0
        elif vol_slice < 0.035:
            FLAIR_TRSH = T1w_TRSH = 0.15
        elif vol_slice < 0.070 and vol_slice >= 0.035:
            FLAIR_TRSH = T1w_TRSH = 0.60
        elif vol_slice >= 0.070:
            FLAIR_TRSH = T1w_TRSH = 0.80

    elif patch_size == 4 or patch_size == 8:
        if vol_slice < 0.035:
            FLAIR_TRSH = 0
        else:
            T1w_TRSH = 0.8

    return FLAIR_TRSH, T1w_TRSH


def extract_source_patches(counter_coord, mask_slice, x_c_sources, y_c_sources, patch_size, valid_img_slice):

    counter_x = counter_coord[0]
    counter_y = counter_coord[1]
    channel_num = valid_img_slice.shape[0]
    if channel_num < 3: channel_num = 1

    source_patch_len = counter_x * counter_y
    icv_source_flag = np.zeros([source_patch_len])
    icv_source_flag_valid = np.ones([source_patch_len])
    index_mapping = np.ones([source_patch_len]) * -1

    if channel_num == 3:
        img_source_patch = np.zeros([1,channel_num, patch_size, patch_size])
    else:
        img_source_patch = np.zeros([1, patch_size, patch_size])

    flag = 1
    index = 0
    index_source = 0

    for isc in range(0, counter_x):
        for jsc in range(0, counter_y):
            icv_source_flag[index] = mask_slice[int(x_c_sources[isc]), int(y_c_sources[jsc])]

            if icv_source_flag[index] == 1:

                if flag:
                    flag = 0
                    if channel_num == 3:
                        img_source_patch[0, :, :, :] = get_area(x_c_sources[isc], y_c_sources[jsc], patch_size, patch_size, valid_img_slice, channel_num)
                    else:
                        img_source_patch[0, :, :] = get_area(x_c_sources[isc], y_c_sources[jsc], patch_size, patch_size, valid_img_slice, channel_num)

                else:
                    img_source_area = get_area(x_c_sources[isc], y_c_sources[jsc], patch_size, patch_size,
                                               valid_img_slice, channel_num)
                    if channel_num == 3:
                        img_source_patch = np.concatenate((img_source_patch, np.reshape(img_source_area, (1, channel_num, patch_size, patch_size))))
                    else:
                        img_source_patch = np.concatenate(
                            (img_source_patch, np.reshape(img_source_area, (1, patch_size, patch_size))))

                index_mapping[index] = index_source
                index_source += 1
            index += 1

    icv_source_flag_valid = icv_source_flag_valid[0:index_source]

    return img_source_patch, icv_source_flag_valid, index_mapping


def extract_target_patches_all_modalities(x_len, y_len, img_TRSH, mask_slice, FLAIR, T1w, T2w, patch_size):
    ch = FLAIR.shape[0]
    if ch < 3: ch = 1
    FLAIR_target_patches = []
    T1w_target_patches = []
    T2w_target_patches = []
    img_idx_debug = 0
    for iii in range(0, x_len):
        for jjj in range(0, y_len):
            if mask_slice[iii, jjj] != 0 and np.random.rand(1) < img_TRSH:
                FLAIR_target_patches.append(get_area(iii, jjj, patch_size, patch_size, FLAIR, ch))
                T1w_target_patches.append(get_area(iii, jjj, patch_size, patch_size, T1w, ch))
                T2w_target_patches.append(get_area(iii, jjj, patch_size, patch_size, T2w, ch))
                img_idx_debug += 1

    return FLAIR_target_patches, T1w_target_patches, T2w_target_patches, img_idx_debug

def extract_monomodal_target_patches(x_len, y_len, img_TRSH,  mask_slice,  valid_img_slice, patch_size):
    ch = valid_img_slice.shape[2]
    if ch < 3: ch = 1
    img_target_patches = []
    img_idx_debug = 0
    for iii in range(0, x_len):
        for jjj in range(0, y_len):
            if mask_slice[iii, jjj] != 0 and np.random.rand(1) < img_TRSH:
                img_target_patches.append(get_area(iii, jjj, patch_size, patch_size, valid_img_slice,ch))
                img_idx_debug += 1

    return img_target_patches, img_idx_debug

def extract_target_patches(x_len, y_len, img_TRSH, mask_slice, FLAIR, T1w, T2w, patch_size):

    FLAIR_target_patches = []
    T1w_target_patches = []
    T2w_target_patches = []
    img_idx_debug = 0
    for iii in range(0, x_len):
        for jjj in range(0, y_len):
            if mask_slice[iii, jjj] != 0 and np.random.rand(1) < img_TRSH:
                FLAIR_target_patches.append(get_area(iii, jjj, patch_size, patch_size, FLAIR, FLAIR.shape[0]))
                T1w_target_patches.append(get_area(iii, jjj, patch_size, patch_size, T1w, T1w.shape[0]))
                T2w_target_patches.append(get_area(iii, jjj, patch_size, patch_size, T2w, T2w.shape[0]))
                img_idx_debug += 1

    return FLAIR_target_patches, T1w_target_patches, T2w_target_patches, img_idx_debug


def calculate_age_value(FLAIR_source_patch, FLAIR_target_patches, num_samples,
                        icv_source_flag_valid, num_mean_samples, alpha, FLAIR_age_values_valid):
    FLAIR_target_patches_np = np.array(FLAIR_target_patches)
    np.random.shuffle(FLAIR_target_patches_np)

    if FLAIR_target_patches_np.shape[0] > num_samples:
        FLAIR_target_patches_np = FLAIR_target_patches_np[0:num_samples, :, :]

    ''' Reshaping array data '''

    if len(np.array(FLAIR_target_patches).shape) == 3:
        FLAIR_source_patch_cuda_all = np.reshape(FLAIR_source_patch,
                                                 (FLAIR_source_patch.shape[0],
                                                  FLAIR_source_patch.shape[1] *
                                                  FLAIR_source_patch.shape[2]))
        FLAIR_target_patches_np_cuda_all = np.reshape(FLAIR_target_patches_np,
                                                      (FLAIR_target_patches_np.shape[0],
                                                       FLAIR_target_patches_np.shape[1] *
                                                       FLAIR_target_patches_np.shape[2]))
    else:
        FLAIR_source_patch_cuda_all = np.reshape(FLAIR_source_patch,
                                                 (FLAIR_source_patch.shape[0],
                                                  FLAIR_source_patch.shape[1] * FLAIR_source_patch.shape[2] * FLAIR_source_patch.shape[3]))
        FLAIR_target_patches_np_cuda_all = np.reshape(FLAIR_target_patches_np,
                                                      (FLAIR_target_patches_np.shape[0],
                                                       FLAIR_target_patches_np.shape[1] * FLAIR_target_patches_np.shape[2] * FLAIR_target_patches_np.shape[3]))

    source_len = icv_source_flag_valid.shape[0]
    loop_len = 512  # def: 512
    loop_num = int(np.ceil(source_len / loop_len))

    for il in range(0, loop_num):
        ''' Only process sub-array '''
        FLAIR_source_patches_loop = FLAIR_source_patch_cuda_all[il * loop_len:(il * loop_len) + loop_len, :]

        '''FLAIR  SUBTRACTION '''
        sub_result_gm = cuda.device_array((FLAIR_source_patches_loop.shape[0],
                                           FLAIR_target_patches_np_cuda_all.shape[0],
                                           FLAIR_target_patches_np_cuda_all.shape[1]))
        sub_difference_result = cuda.device_array((FLAIR_source_patches_loop.shape[0],
                                                   FLAIR_target_patches_np_cuda_all.shape[0], 2))
        TPB = (4, 256)  # threads per block
        BPGx = int(math.ceil(FLAIR_source_patches_loop.shape[0] / TPB[0]))
        BPGy = int(math.ceil(FLAIR_target_patches_np_cuda_all.shape[0] / TPB[1]))
        BPGxy = (BPGx, BPGy)  # blocks per grid
        cu_sub_st[BPGxy, TPB](FLAIR_source_patches_loop, FLAIR_target_patches_np_cuda_all,
                              sub_result_gm)  # source - target
        cu_max_mean_abs[BPGxy, TPB](sub_result_gm, sub_difference_result)  # make a array [max(s-t),mean(s-t)]
        sub_result_gm = 0  # Free memory
        '''  DISTANCE '''
        distances_result = cuda.device_array((FLAIR_source_patches_loop.shape[0],
                                              FLAIR_target_patches_np_cuda_all.shape[
                                                  0]))
        cu_distances[BPGxy, TPB](sub_difference_result,
                                 icv_source_flag_valid[
                                 il * loop_len:(il * loop_len) + loop_len],
                                 distances_result, alpha)  # Calculate age value
        sub_difference_result = 0  # Free memory

        ''' SORT '''
        TPB = 256
        BPG = int(math.ceil(distances_result.shape[0] / TPB))
        cu_sort_distance[BPG, TPB](distances_result)  # Sort distance small -> larget

        ''' MEAN (AGE-VALUE) '''
        idx_start = 8

        distances_result_for_age = distances_result[:, idx_start:idx_start + num_mean_samples]
        distances_result = 0  # Free memory
        cu_age_value[BPG, TPB](distances_result_for_age, FLAIR_age_values_valid[il * loop_len:(il * loop_len) + loop_len])  # mean(age value) of each source
        distances_result_for_age = 0  # Free memory
        del FLAIR_source_patches_loop  # Free memory
    return FLAIR_age_values_valid


def calculate_threshold_age_value(FLAIR_source_patch, FLAIR_target_patches, num_samples, icv_source_flag_valid,
                                  num_mean_samples, alpha, T1w_age_values_valid, T2w_age_values_valid,
                                  Threshold_Mult_age_values_valid, trsh, alg_num):
    FLAIR_target_patches_np = np.array(FLAIR_target_patches)
    np.random.shuffle(FLAIR_target_patches_np)

    if FLAIR_target_patches_np.shape[0] > num_samples:
        FLAIR_target_patches_np = FLAIR_target_patches_np[0:num_samples, :, :]

    ''' Reshaping array data '''
    FLAIR_source_patch_cuda_all = np.reshape(FLAIR_source_patch,
                                             (FLAIR_source_patch.shape[0],
                                              FLAIR_source_patch.shape[1] *
                                              FLAIR_source_patch.shape[2]))
    T1w_age_values_valid = np.reshape(T1w_age_values_valid,
                                             (T1w_age_values_valid.shape[0],
                                              T1w_age_values_valid.shape[1] *
                                              T1w_age_values_valid.shape[2]))
    T2w_age_values_valid = np.reshape(T2w_age_values_valid,
                                             (T2w_age_values_valid.shape[0],
                                              T2w_age_values_valid.shape[1] *
                                              T2w_age_values_valid.shape[2]))
    FLAIR_target_patches_np_cuda_all = np.reshape(FLAIR_target_patches_np,
                                                  (FLAIR_target_patches_np.shape[0],
                                                   FLAIR_target_patches_np.shape[1] *
                                                   FLAIR_target_patches_np.shape[2]))

    source_len = icv_source_flag_valid.shape[0]
    loop_len = 512  # def: 512
    loop_num = int(np.ceil(source_len / loop_len))

    for il in range(0, loop_num):
        ''' Only process sub-array '''
        FLAIR_source_patches_loop = FLAIR_source_patch_cuda_all[il * loop_len:(il * loop_len) + loop_len, :]
        T1w_age_values_valid_loop = T1w_age_values_valid[il * loop_len:(il * loop_len) + loop_len]
        T2w_age_values_valid_loop = T2w_age_values_valid[il * loop_len:(il * loop_len) + loop_len]

        '''FLAIR  SUBTRACTION '''
        sub_distance_gm = cuda.device_array((FLAIR_source_patches_loop.shape[0],
                                             FLAIR_target_patches_np_cuda_all.shape[0],
                                             FLAIR_target_patches_np_cuda_all.shape[1]))

        TPB = (4, 256)  # threads per block
        BPGx = int(math.ceil(FLAIR_source_patches_loop.shape[0] / TPB[0]))
        BPGy = int(math.ceil(FLAIR_target_patches_np_cuda_all.shape[0] / TPB[1]))
        BPGxy = (BPGx, BPGy)  # blocks per grid

        sub_difference_result = cuda.device_array(
            (FLAIR_source_patches_loop.shape[0], FLAIR_target_patches_np_cuda_all.shape[0], 2))

        patch_size = FLAIR_source_patch.shape[2]

        if alg_num ==1:
            cu_alg1_penalisation_sub_st[BPGxy, TPB](FLAIR_source_patches_loop, FLAIR_target_patches_np_cuda_all,
                                              sub_distance_gm, T1w_age_values_valid_loop, T2w_age_values_valid_loop,
                                              patch_size, trsh)  # source - target
        elif alg_num==2:
            cu_alg2_penalisation_sub_st[BPGxy, TPB](FLAIR_source_patches_loop, FLAIR_target_patches_np_cuda_all,
                                                    sub_distance_gm, T1w_age_values_valid_loop, T2w_age_values_valid_loop,
                                                    patch_size, trsh)  # source - target
        cu_max_mean_abs[BPGxy, TPB](sub_distance_gm,
                                    sub_difference_result)  # make a array [max(s-t),mean(s-t)]

        sub_distance_gm = 0

        '''  DISTANCE '''
        distance_result = cuda.device_array(
            (FLAIR_source_patches_loop.shape[0], FLAIR_target_patches_np_cuda_all.shape[0]))
        cu_distances[BPGxy, TPB](sub_difference_result,
                                 icv_source_flag_valid[il * loop_len:(il * loop_len) + loop_len],
                                 distance_result, alpha)  # Calculate age value

        sub_difference_result = 0  # Free memory
        ''' SORT '''
        TPB = 256
        BPG = int(math.ceil(distance_result.shape[0] / TPB))
        cu_sort_distance[BPG, TPB](distance_result)

        ''' MEAN (AGE-VALUE) '''
        idx_start = 0

        distances_result_for_age = distance_result[:,
                                   idx_start:idx_start + num_mean_samples]
        distance_result = 0  # Free memory
        cu_age_value[BPG, TPB](distances_result_for_age, Threshold_Mult_age_values_valid[il * loop_len:(
                                                                                                       il * loop_len) + loop_len])  # mean(age value) of each source
        distances_result_for_age = 0  # Free memory

        del FLAIR_source_patches_loop  # Free memory
    return Threshold_Mult_age_values_valid


def calculate_three_channel_age_value(rgb_source_patch, rgb_target_patches, num_samples, icv_source_flag_valid,
                                      num_mean_samples, alpha, rgb_age_values_valid):
    rgb_target_patches_np = np.array(rgb_target_patches)
    np.random.shuffle(rgb_target_patches_np)

    if rgb_target_patches_np.shape[0] > num_samples:
        rgb_target_patches_np = rgb_target_patches_np[0:num_samples, :, :, :]


    ''' Reshaping array data '''
    rgb_source_patch_cuda_all = np.reshape(rgb_source_patch, (rgb_source_patch.shape[0],
                                                              rgb_source_patch.shape[1] * rgb_source_patch.shape[2],
                                                              rgb_source_patch.shape[3]))
    rgb_target_patches_np_cuda_all = np.reshape(rgb_target_patches_np, (rgb_target_patches_np.shape[0],
                                                                        rgb_target_patches_np.shape[1] *
                                                                        rgb_target_patches_np.shape[2],
                                                                        rgb_target_patches_np.shape[3]))

    source_len = icv_source_flag_valid.shape[0]
    loop_len = 512  # def: 512
    loop_num = int(np.ceil(source_len / loop_len))

    for il in range(0, loop_num):
        ''' Only process sub-array '''
        rgb_source_patches_loop = rgb_source_patch_cuda_all[il * loop_len:(il * loop_len) + loop_len, :]

        '''rgb  SUBTRACTION '''
        sub_result_gm = cuda.device_array((rgb_source_patches_loop.shape[0],
                                           rgb_target_patches_np_cuda_all.shape[0],
                                           rgb_target_patches_np_cuda_all.shape[1]))
        sub_difference_result = cuda.device_array((rgb_source_patches_loop.shape[0],
                                                   rgb_target_patches_np_cuda_all.shape[0], 2))
        TPB = (4, 256)  # threads per block
        BPGx = int(math.ceil(rgb_source_patches_loop.shape[0] / TPB[0]))
        BPGy = int(math.ceil(rgb_target_patches_np_cuda_all.shape[0] / TPB[1]))
        BPGxy = (BPGx, BPGy)  # blocks per grid
        distances_result = cuda.device_array((rgb_source_patches_loop.shape[0], rgb_target_patches_np_cuda_all.shape[0]))


        cu_sub_st_threechannel[BPGxy, TPB](rgb_source_patches_loop, rgb_target_patches_np_cuda_all, distances_result)  # source - target
        cu_max_mean_abs[BPGxy, TPB](sub_result_gm, sub_difference_result)  # make a array [max(s-t),mean(s-t)]
        sub_result_gm = 0  # Free memory
        '''  DISTANCE '''

        cu_distances[BPGxy, TPB](sub_difference_result,icv_source_flag_valid[ il * loop_len:(il * loop_len) + loop_len],
                                 distances_result, alpha)  # Calculate age value
        sub_difference_result = 0  # Free memory

        ''' SORT '''
        TPB = 256
        BPG = int(math.ceil(distances_result.shape[0] / TPB))
        cu_sort_distance[BPG, TPB](distances_result)  # Sort distance small -> larget

        ''' MEAN (AGE-VALUE) '''
        idx_start = 0

        distances_result_for_age = distances_result[:,
                                   idx_start:idx_start + num_mean_samples]
        distances_result = 0  # Free memory
        cu_age_value[BPG, TPB](distances_result_for_age,
                               rgb_age_values_valid[
                               il * loop_len:(il * loop_len) + loop_len])  # mean(age value) of each source
        distances_result_for_age = 0  # Free memory
        del rgb_source_patches_loop  # Free memory
    return rgb_age_values_valid


def mapping_result(index_mapping, age_values_all, age_values_valid, bool_var):
    index = 0
    for idx_val in index_mapping:
        if idx_val != -1:
            age_values_all[index] = age_values_valid[int(idx_val)]
        index += 1

    ''' Normalisation to probabilistic map (0...1) '''
    if (np.max(age_values_all) - np.min(age_values_all)) == 0:
        all_mean_distance_normed = age_values_all
    else:
        all_mean_distance_normed = np.divide((age_values_all - np.min(age_values_all)),
                                             (np.max(age_values_all) - np.min(age_values_all)))

    if bool_var:
        return all_mean_distance_normed, age_values_all
    else:
        return all_mean_distance_normed


def FLAIR_penalisation(patch_size, blending_weights, slice_age_map_all, penalty_slice, icv_slice):
    ''' >>> Part 1 <<< '''
    ''' Combined all patches age map information '''
    combined_age_map = 0
    for bi in range(len(patch_size)):
        combined_age_map += np.multiply(blending_weights[bi], slice_age_map_all[bi, :, :])

    ''' Global Normalisation - saving needed data '''
    penalty_combined_age_map = np.multiply(np.multiply(combined_age_map, penalty_slice), icv_slice)  ### PENALTY

    return penalty_combined_age_map


def calculate_dsc(gt_map, prob_map, dsc_trsh):
    JAC = np.sum(gt_map & (prob_map > dsc_trsh)) / np.sum(gt_map | (prob_map > dsc_trsh))
    return 2 * JAC / (1 + JAC)


def normalisation(input_mat):
    return np.divide((input_mat - np.min(input_mat)), (np.max(input_mat) - np.min(input_mat)))


def rgb_to_lab(rgb_slice):
    slice_shape = rgb_slice.shape
    rgb2lab_slice = np.dstack((rgb_slice[0], rgb_slice[1], rgb_slice[2]))
    rgb2lab_slice = rgb2lab_slice/255.0
    lab_slice = color.rgb2lab(rgb2lab_slice)
    lab_slice = np.array([lab_slice[:,:,0],lab_slice[:,:,1],lab_slice[:,:,2]])

    return lab_slice

def modality_to_rgb(FLAIR, T1w, T2w):
    normed_FLAIR = (normalisation(FLAIR) * 255.0)
    normed_T1w = (normalisation(T1w) * 255.0)
    normed_T2w = (normalisation(T2w) * 255.0)

    result_slice = [normed_FLAIR.astype(float), normed_T1w.astype(float) ,normed_T2w.astype(float)]

    rgb_slice = np.array(result_slice)

    return rgb_slice


def modality_to_gray(FLAIR, T1w, T2w, weight):
    normed_FLAIR = (normalisation(FLAIR) * 255.0)
    normed_T1w = (normalisation(T1w) * 255.0)
    normed_T2w = (normalisation(T2w) * 255.0)

    result_slice = weight[0] * normed_FLAIR.astype(float) + weight[1] * normed_T1w.astype(float) + weight[2] * normed_T2w.astype(float)

    gray_slice = np.array(result_slice)

    return gray_slice

def extract_valid_grad_var_slice(brain_slice, mask_slice):
    [x_len, y_len] = brain_slice.shape
    new_var_slice = np.zeros(brain_slice.shape)
    grad_y, grad_x = np.gradient(brain_slice)
    grad_slice = np.sqrt(grad_y**2 + grad_x**2)

    for xx in range(x_len):
        for yy in range(y_len):
            if mask_slice[xx, yy]:
                var_patch = np.nonzero(brain_slice[xx-1:xx+1,yy-1:yy+1])
                new_var_slice[xx,yy] = np.mean(var_patch)

    grad_slice = np.multiply(grad_slice, mask_slice.astype(float))

    return grad_slice, new_var_slice

def standard_normalisation(brain_data, mask_brain):
    nonzero_brain = brain_data[mask_brain==1]
    mean_val = np.mean(nonzero_brain)
    std_val = np.std(nonzero_brain)
    normalised_brain=(brain_data-mean_val)*std_val
    normalised_brain[mask_brain==0] = 0

    return normalised_brain


def age_value_mapping(slice_agemap, patch_size, icv_slice):
    slice_agemap = cv2.resize(slice_agemap, None, fx=patch_size, fy=patch_size, interpolation=cv2.INTER_CUBIC)

    return np.multiply(icv_slice, skifilters.gaussian(slice_agemap, sigma=0.5, truncate=2.0))

