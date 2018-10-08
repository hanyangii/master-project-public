import matplotlib

matplotlib.use('Agg')

from numba import cuda
from timeit import default_timer as timer
from matplotlib import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IAM_lib import *

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.morphology as skimorph
import skimage.filters as skifilters
import scipy.ndimage.morphology as scimorph

import math, numba, cv2, csv, gc, time
import os, errno, sys, shutil

# Turn interactive plotting off
plt.ioff()


def save_result_img(FLAIR_data, final_result, T1w_max_agemap, T2w_max_agemap, original_icv_data, original_csf_data, gt_data, save_path):

    for zz in range(0, FLAIR_data.shape[2]):
        save_path_jpg = save_path + str(zz) + '_combined.jpg'
        fig2, axes2 = plt.subplots(2, 2)
        fig2.set_size_inches(16, 16)

        axes2[0, 0].set_title('Final result')
        im1 = axes2[0, 0].imshow(np.rot90(final_result[:, :, zz]), cmap="jet", vmin=0, vmax=1)
        divider1 = make_axes_locatable(axes2[0, 0])
        cax1 = divider1.append_axes("right", size="7%", pad=0.05)
        cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)


        axes2[0, 1].set_title('T1w max agemap')
        im2 = axes2[0, 1].imshow(np.rot90(T1w_max_agemap[zz]), cmap="jet", vmin=0, vmax=1)
        divider2 = make_axes_locatable(axes2[0, 1])
        cax2 = divider2.append_axes("right", size="7%", pad=0.05)
        cbar2 = plt.colorbar(im2, ticks=[0, 0.5, 1], cax=cax2)

        axes2[1, 0].set_title('T2w max agemap')
        im3 = axes2[1, 0].imshow(np.rot90(T2w_max_agemap[zz]), cmap="jet",
                                 vmin=0, vmax=1)
        divider3 = make_axes_locatable(axes2[1, 0])
        cax3 = divider3.append_axes("right", size="7%", pad=0.05)
        cbar3 = plt.colorbar(im3, ticks=[0, 0.5, 1], cax=cax3)

        axes2[1, 1].set_title('WMH Ground Truth')
        if np.max(gt_data[:, :, zz]) >0:
            gt_check_slice = np.multiply(FLAIR_data[:, :, zz], original_icv_data[:,:,zz].astype(float), (~original_csf_data[:,:,zz]).astype(float))
            gt_check_slice[gt_data[:, :, zz] > 0] = 10000
        else:
            gt_check_slice = np.zeros(gt_data[:, :, zz].shape)
        im4 = axes2[1, 1].imshow(np.rot90(gt_check_slice), cmap="jet")
        divider4 = make_axes_locatable(axes2[1, 1])
        cax4 = divider4.append_axes("right", size="7%", pad=0.05)
        cbar4 = plt.colorbar(im4, cax=cax4)

        plt.tight_layout()
        # Make space for title
        plt.subplots_adjust(top=0.95)

        fig2.savefig(save_path_jpg, dpi=100)
        plt.close()


def save_agemap_img(slice_age_map_all, patch_size, img_slice, gt_slice, save_path, original_icv_data, original_csf_data, slice_name):

    '''Show all age maps based on patch's size and saving the data'''
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.set_size_inches(16, 10)
    fig.suptitle('All Patches Gaussian Filtered', fontsize=16)

    axes[0, 0].set_title('Patch 1 x 1')
    im1 = axes[0, 0].imshow(np.rot90(slice_age_map_all[0, :, :]), cmap="jet", vmin=0, vmax=1)
    divider1 = make_axes_locatable(axes[0, 0])
    cax1 = divider1.append_axes("right", size="7%", pad=0.05)
    cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)

    if len(patch_size) > 1:
        axes[0, 1].set_title('Patch 2 x 2')
        im2 = axes[0, 1].imshow(np.rot90(slice_age_map_all[1, :, :]), cmap="jet", vmin=0, vmax=1)
        divider2 = make_axes_locatable(axes[0, 1])
        cax2 = divider2.append_axes("right", size="7%", pad=0.05)
        cbar2 = plt.colorbar(im2, ticks=[0, 0.5, 1], cax=cax2)

        if len(patch_size) > 2:
            axes[1, 0].set_title('Patch 4 x 4')
            im3 = axes[1, 0].imshow(np.rot90(slice_age_map_all[2, :, :]), cmap="jet", vmin=0,
                                    vmax=1)
            divider3 = make_axes_locatable(axes[1, 0])
            cax3 = divider3.append_axes("right", size="7%", pad=0.05)
            cbar3 = plt.colorbar(im3, ticks=[0, 0.5, 1], cax=cax3)

            if len(patch_size) > 3:
                axes[1, 1].set_title('Patch 8 x 8')
                im4 = axes[1, 1].imshow(np.rot90(slice_age_map_all[3, :, :]), cmap="jet", vmin=0,
                                        vmax=1)
                divider4 = make_axes_locatable(axes[1, 1])
                cax4 = divider4.append_axes("right", size="7%", pad=0.05)
                cbar4 = plt.colorbar(im4, ticks=[0, 0.5, 1], cax=cax4)

    axes[0, 2].set_title(slice_name)
    gt_check_slice = np.multiply(img_slice, original_icv_data, (~original_csf_data).astype(float))
    im5 = axes[0, 2].imshow(np.rot90(gt_check_slice), cmap="jet")
    divider5 = make_axes_locatable(axes[0, 2])
    cax5 = divider5.append_axes("right", size="7%", pad=0.05)
    cbar5 = plt.colorbar(im5, cax=cax5)

    axes[1, 2].set_title(slice_name + ' with WMH')
    gt_check_slice = np.multiply(img_slice, original_icv_data, (~original_csf_data).astype(float))
    gt_check_slice[gt_slice > 0] = 10000
    im6 = axes[1, 2].imshow(np.rot90(gt_check_slice), cmap='jet')
    divider6 = make_axes_locatable(axes[1, 2])
    cax6 = divider6.append_axes("right", size="7%", pad=0.05)
    cbar6 = plt.colorbar(im6, cax=cax6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    fig.savefig(save_path, dpi=100)
    plt.close()


def iam_Penalisation_gpu_compute(data = "",  patch_size=[1, 2, 4, 8], blending_weights=[0.65, 0.2, 0.1, 0.05], alpha=0.5, save_jpeg=True, save_mat=True,
                                 modality_data=[], icv_data=[], csf_data=[], gt_data=[],num_samples=[],
                                 original_icv_data=[], original_csf_data=[],num_mean_samples=0, dirOutput="", trsh=0.6):
    FLAIR_data = modality_data[0]
    T1w_data = modality_data[1]
    T2w_data = modality_data[2]

    T1w_max_agemap=[]
    T2w_max_agemap=[]

    [x_len, y_len, z_len] = FLAIR_data.shape
    alg1_combined_age_map_mri_mult = np.zeros((x_len, y_len, z_len))
    alg2_combined_age_map_mri_mult = np.zeros((x_len, y_len, z_len))
    T1w_agemap = np.zeros((2, x_len, y_len, z_len))
    T2w_agemap = np.zeros((2, x_len, y_len, z_len))

    '''Remove non-brain area'''
    mask_brain = np.multiply(csf_data, icv_data).astype(float)
    valid_FLAIR_data = np.multiply(mask_brain, FLAIR_data).astype(float)
    valid_T1w_data = np.multiply(mask_brain, T1w_data).astype(float)
    valid_T2w_data = np.multiply(mask_brain, T2w_data).astype(float)

    for zz in range(0, FLAIR_data.shape[2]):
        print('\n---> Slice number: ' + str(zz) + ' <---')

        '''Load Image Slices'''
        icv_slice = icv_data[:, :, zz]
        csf_slice = csf_data[:, :, zz]
        gt_slice = gt_data[:, :, zz]
        mask_slice = mask_brain[:,:,zz]
        valid_FLAIR_slice = valid_FLAIR_data[:,:,zz]
        valid_T1w_slice = valid_T1w_data[:,:,zz]
        valid_T2w_slice = valid_T2w_data[:, :, zz]

        FLAIR_slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))
        alg1_slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))
        alg2_slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))
        T1w_slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))
        T2w_slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))

        penalty_slice = np.multiply(mask_slice,FLAIR_data[:,:,zz]).astype(float)  ### PENALTY

        # Vol distance threshold
        vol_slice = np.count_nonzero(valid_FLAIR_slice) / (x_len * y_len)
        one_patch = timer()

        FLAIR_sources = []
        FLAIR_targets = []
        FLAIR_flag_valid = []
        agemap_cnt = 0
        idx_mapping = []

        print('WMH size : ', np.sum(gt_slice))

        for enum, xy in enumerate(range(0, len(patch_size))):
            if zz == 0:
                try:
                    dirOutData = dirOutput + '/' + data
                    os.makedirs(dirOutData + '/' + str(patch_size[xy]))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

            counter_y = int(y_len / patch_size[xy])
            counter_x = int(x_len / patch_size[xy])

            T1w_age_values_all = np.zeros(counter_x * counter_y)
            T2w_age_values_all = np.zeros(counter_x * counter_y)

            valid = 0
            idx_debug = 0


            if ((vol_slice >= 0.008 and vol_slice < 0.035) and (patch_size[xy] == 1 or patch_size[xy] == 2)) or \
                ((vol_slice >= 0.035 and vol_slice < 0.065) and (patch_size[xy] == 1 or patch_size[xy] == 2 or patch_size[xy] == 4)) or \
                (vol_slice > 0.065):

                valid = 1
                FLAIR_TRSH, T1w_TRSH = brain_vol_trsh(vol_slice, patch_size[xy])

                ## Creating grid-patch 'xy-by-xy'
                #  -- Column
                y_c = np.ceil(patch_size[xy] / 2)
                y_c_sources = np.zeros(int(y_len / patch_size[xy]))
                for iy in range(0, int(y_len / patch_size[xy])):
                    y_c_sources[iy] = (iy * patch_size[xy]) + y_c - 1

                # -- Row
                x_c = np.ceil(patch_size[xy] / 2)
                x_c_sources = np.zeros(int(x_len / patch_size[xy]))
                for ix in range(0, int(x_len / patch_size[xy])):
                    x_c_sources[ix] = (ix * patch_size[xy]) + x_c - 1

                ''' Extracting Source and Target Patches '''
                FLAIR_source_patch, icv_source_flag_valid, index_mapping = extract_source_patches([counter_x, counter_y], mask_slice, x_c_sources,
                                                                                                   y_c_sources, patch_size[xy], valid_FLAIR_slice)
                FLAIR_sources.append(FLAIR_source_patch)
                FLAIR_flag_valid.append(icv_source_flag_valid)
                idx_mapping.append(index_mapping)


                T1w_source_patch, icv_source_flag_valid, index_mapping = extract_source_patches([counter_x, counter_y], mask_slice, x_c_sources,
                                                                                                y_c_sources, patch_size[xy], valid_T1w_slice)
                T1w_age_values_valid = np.zeros(len(icv_source_flag_valid))

                T2w_source_patch, icv_source_flag_valid, index_mapping = extract_source_patches([counter_x, counter_y], mask_slice, x_c_sources,
                                                                                                y_c_sources, patch_size[xy], valid_T2w_slice)

                T2w_age_values_valid = np.zeros(len(icv_source_flag_valid))

                FLAIR_target_patches, T1w_target_patches, T2w_target_patches, idx_debug = extract_target_patches(x_len, y_len, FLAIR_TRSH, mask_slice, valid_FLAIR_slice,
                                                                                                                 valid_T1w_slice, valid_T2w_slice,patch_size[xy])
                FLAIR_targets.append(FLAIR_target_patches)

                '''Calculate T1 & T2 Age Values'''
                if len(T1w_target_patches) > 0:
                    agemap_cnt += 1
                    t_weight_alpha = 0.5
                    T1w_age_values_valid = calculate_age_value(T1w_source_patch, T1w_target_patches, num_samples, icv_source_flag_valid,
                                                               num_mean_samples, t_weight_alpha, T1w_age_values_valid)
                    T1w_age_values_valid_norm, unnormed_T1w = mapping_result(index_mapping, T1w_age_values_all, T1w_age_values_valid, True)
                    T1w_slice_age_map_all[xy, :, :] = age_value_mapping(np.reshape(T1w_age_values_valid_norm, [counter_x, counter_y]), patch_size[xy], icv_slice)
                    if patch_size[xy] == 1:
                        T1w_agemap[0, :, :, zz] = age_value_mapping(np.reshape(unnormed_T1w, [counter_x, counter_y]), patch_size[xy], icv_slice)
                    elif patch_size[xy] == 4:
                        T1w_agemap[1, :, :, zz] = age_value_mapping(np.reshape(unnormed_T1w, [counter_x, counter_y]), patch_size[xy], icv_slice)

                if len(T2w_target_patches) > 0:
                    T2w_age_values_valid = calculate_age_value(T2w_source_patch, T2w_target_patches,num_samples, icv_source_flag_valid,
                                                               num_mean_samples, t_weight_alpha, T2w_age_values_valid)
                    T2w_age_values_valid_norm, unnormed_T2w = mapping_result(index_mapping, T2w_age_values_all, T2w_age_values_valid, True)
                    T2w_slice_age_map_all[xy, :, :] = age_value_mapping(np.reshape(T2w_age_values_valid_norm, [counter_x, counter_y]), patch_size[xy], icv_slice)
                    if patch_size[xy] == 1:
                        T2w_agemap[0, :, :, zz] = age_value_mapping(np.reshape(unnormed_T2w, [counter_x, counter_y]), patch_size[xy], icv_slice)
                    elif patch_size[xy] == 4:
                        T2w_agemap[1, :, :, zz] = age_value_mapping(np.reshape(unnormed_T2w, [counter_x, counter_y]), patch_size[xy], icv_slice)

                print(str(patch_size[xy])+' size Sampling finished with: ' + str(idx_debug) + ' Target patches from: ' + str(x_len * y_len))

        if agemap_cnt>0:
            T1w_mean_agemap = np.max(T1w_slice_age_map_all[0:agemap_cnt,:,:], axis=0)
            T2w_mean_agemap = np.max(T2w_slice_age_map_all[0:agemap_cnt,:,:], axis=0)
        else:
            T1w_mean_agemap = T2w_mean_agemap = np.zeros([x_len, y_len])

        T1w_max_agemap.append(T1w_mean_agemap)
        T2w_max_agemap.append(T2w_mean_agemap)


        for enum, xy in enumerate(range(0, len(patch_size))):
            if ((vol_slice >= 0.008 and vol_slice < 0.035) and (patch_size[xy] == 1 or patch_size[xy] == 2)) or \
                ((vol_slice >= 0.035 and vol_slice < 0.065) and (patch_size[xy] == 1 or patch_size[xy] == 2 or patch_size[xy] == 4)) or \
                (vol_slice > 0.065):
                counter_y = int(y_len / patch_size[xy])
                counter_x = int(x_len / patch_size[xy])

                y_c = np.ceil(patch_size[xy] / 2)
                y_c_sources = np.zeros(int(y_len / patch_size[xy]))
                for iy in range(0, int(y_len / patch_size[xy])):
                    y_c_sources[iy] = (iy * patch_size[xy]) + y_c - 1

                # -- Row
                x_c = np.ceil(patch_size[xy] / 2)
                x_c_sources = np.zeros(int(x_len / patch_size[xy]))
                for ix in range(0, int(x_len / patch_size[xy])):
                    x_c_sources[ix] = (ix * patch_size[xy]) + x_c - 1

                ''' Extracting Source and Target Patches '''
                T1w_mean_agevals, dd, ee = extract_source_patches([counter_x, counter_y], mask_slice, x_c_sources,
                                                                    y_c_sources, patch_size[xy], T1w_mean_agemap)
                T2w_mean_agevals, dd, ee = extract_source_patches([counter_x, counter_y], mask_slice, x_c_sources,
                                                                  y_c_sources, patch_size[xy], T2w_mean_agemap)

                icv_source_flag_valid = FLAIR_flag_valid[enum]
                FLAIR_source_patch = FLAIR_sources[enum]
                FLAIR_target_patches = FLAIR_targets[enum]
                index_mapping = idx_mapping[enum]

                alg1_age_values_all = np.zeros(counter_x * counter_y)
                alg2_age_values_all = np.zeros(counter_x * counter_y)

                if len(FLAIR_target_patches)>0:
                    '''Algorithm 1'''
                    alg1_age_values_valid = np.zeros(len(icv_source_flag_valid))
                    alg1_age_values_valid = calculate_threshold_age_value(FLAIR_source_patch, FLAIR_target_patches, num_samples,
                                                                            icv_source_flag_valid, num_mean_samples, alpha, T1w_mean_agevals, T2w_mean_agevals,
                                                                            alg1_age_values_valid, trsh, 1)
                    alg1_age_values_valid = mapping_result(index_mapping, alg1_age_values_all, alg1_age_values_valid, False)
                    alg1_slice_age_map_all[xy, :, :] = age_value_mapping(np.reshape(alg1_age_values_valid, [counter_x, counter_y]), patch_size[xy],icv_slice)

                    '''Algorithm 2'''
                    alg2_age_values_valid = np.zeros(len(icv_source_flag_valid))

                    if patch_size[xy] ==1 :
                        alg2_trsh = trsh + 0.2
                    elif patch_size[xy] ==2:
                        alg2_trsh = trsh + 0.1
                    else:
                        alg2_trsh = trsh
                    alg2_age_values_valid = calculate_threshold_age_value(FLAIR_source_patch, FLAIR_target_patches, num_samples,
                                                                            icv_source_flag_valid, num_mean_samples, alpha, T1w_mean_agevals, T2w_mean_agevals,
                                                                            alg2_age_values_valid, alg2_trsh, 2)
                    alg2_age_values_valid = mapping_result(index_mapping, alg2_age_values_all, alg2_age_values_valid, False)
                    alg2_slice_age_map_all[xy, :, :] = age_value_mapping(np.reshape(alg2_age_values_valid, [counter_x, counter_y]), patch_size[xy],icv_slice)

                else:
                    alg1_slice_age_map_all[xy, :, :] = cv2.resize(np.zeros([counter_x, counter_y]), None, fx=patch_size[xy], fy=patch_size[xy], interpolation=cv2.INTER_CUBIC)
                    alg2_slice_age_map_all[xy, :, :] = cv2.resize(np.zeros([counter_x, counter_y]), None, fx=patch_size[xy], fy=patch_size[xy], interpolation=cv2.INTER_CUBIC)

        numba.cuda.profile_stop()
        alg1_slice_age_map_all = np.nan_to_num(alg1_slice_age_map_all)
        alg2_slice_age_map_all = np.nan_to_num(alg2_slice_age_map_all)

        '''Save age maps for each modality'''
        if save_jpeg:
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Patch/' + str(zz) + '_algorithm1_all.jpg'
            save_agemap_img(alg1_slice_age_map_all, patch_size, FLAIR_data[:, :, zz], gt_slice,
                            save_path, original_icv_data[:, :, zz], original_csf_data[:, :, zz], 'FLAIR')
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Patch/' + str(zz) + '_algorithm2_all.jpg'
            save_agemap_img(alg2_slice_age_map_all, patch_size, FLAIR_data[:, :, zz], gt_slice,
                            save_path, original_icv_data[:, :, zz], original_csf_data[:, :, zz], 'FLAIR')
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Patch/' + str(zz) + '_T1w_all.jpg'
            save_agemap_img(T1w_slice_age_map_all, patch_size, T1w_mean_agemap.astype(float), gt_slice,
                            save_path, original_icv_data[:, :, zz], original_csf_data[:, :, zz], 'T1w')
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Patch/' + str(zz) + '_T2w_all.jpg'
            save_agemap_img(T2w_slice_age_map_all, patch_size, T2w_mean_agemap.astype(float), gt_slice,
                            save_path, original_icv_data[:, :, zz], original_csf_data[:, :, zz], 'T2w')
        ## Save data
        if save_mat:
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Alg1_Result/Data/'
            sio.savemat(save_path + str(zz) + '_dat.mat', {'slice_age_map':alg1_slice_age_map_all})
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Alg2_Result/Data/'
            sio.savemat(save_path + str(zz) + '_dat.mat', {'slice_age_map':alg2_slice_age_map_all})

        alg1_combined_age_map_mri_mult[:, :, zz] = FLAIR_penalisation(patch_size, blending_weights, alg1_slice_age_map_all, penalty_slice, icv_slice)
        alg2_combined_age_map_mri_mult[:, :, zz] = FLAIR_penalisation(patch_size, blending_weights, alg2_slice_age_map_all, penalty_slice, icv_slice)

    ''' >>> Part 2 <<< '''
    ''' Penalty + Global Normalisation (GN) '''

    alg1_combined_age_map_mri_mult = normalisation(alg1_combined_age_map_mri_mult)
    alg2_combined_age_map_mri_mult = normalisation(alg2_combined_age_map_mri_mult)

    if save_jpeg:
        save_path = dirOutput + '/' + data + '/IAM_combined_python/Alg1_Result/Image/'
        save_result_img(valid_FLAIR_data, alg1_combined_age_map_mri_mult, T1w_max_agemap,
                        T2w_max_agemap, original_icv_data, original_csf_data, gt_data, save_path)
        save_path = dirOutput + '/' + data + '/IAM_combined_python/Alg2_Result/Image/'
        save_result_img(valid_FLAIR_data, alg2_combined_age_map_mri_mult, T1w_max_agemap,
                        T2w_max_agemap, original_icv_data, original_csf_data, gt_data, save_path)


    del valid_FLAIR_slice, valid_T1w_slice, valid_T2w_slice, icv_slice, csf_slice
    del FLAIR_data,T1w_data, T2w_data, icv_data, csf_data
    del valid_T1w_data, valid_T2w_data, valid_FLAIR_data
    del original_icv_data, original_csf_data
    del icv_source_flag_valid, index_mapping
    gc.collect()


    return alg1_combined_age_map_mri_mult, alg2_combined_age_map_mri_mult