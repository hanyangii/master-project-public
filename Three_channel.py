import matplotlib

matplotlib.use('Agg')

from numba import cuda

from IAM_lib import *

import numpy as np
import matplotlib.pyplot as plt

import numba, cv2, gc
import shutil

# Turn interactive plotting off
plt.ioff()


def save_result_img(rgb_result, lab_result, gray_result, gt_data, FLAIR_data, T1w_data, T2w_data, original_icv_data, original_csf_data, save_path):
    for zz in range(0, FLAIR_data.shape[2]):
        save_path_jpg = save_path + str(zz) + '_combined.jpg'
        fig2, axes2 = plt.subplots(2, 3)
        fig2.set_size_inches(16, 10)

        axes2[0, 0].set_title('RGB result')
        im1 = axes2[0, 0].imshow(np.rot90(rgb_result[:, :, zz]), cmap="jet", vmin=0, vmax=1)
        divider1 = make_axes_locatable(axes2[0, 0])
        cax1 = divider1.append_axes("right", size="7%", pad=0.05)
        cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)

        axes2[1, 0].set_title('FLAIR')
        gt_check_slice = np.multiply(FLAIR_data[:, :, zz], original_icv_data[:, :, zz],(~original_csf_data[:, :, zz]).astype(int))
        im2 = axes2[1, 0].imshow(np.rot90(gt_check_slice), cmap="gray")
        divider2 = make_axes_locatable(axes2[1, 0])
        cax2 = divider2.append_axes("right", size="7%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)

        axes2[0, 1].set_title('Lab result')
        im4 = axes2[0, 1].imshow(np.rot90(lab_result[:, :, zz]), cmap="jet", vmin=0, vmax=1)
        divider4 = make_axes_locatable(axes2[0, 1])
        cax4 = divider4.append_axes("right", size="7%", pad=0.05)
        cbar4 = plt.colorbar(im4, ticks=[0, 0.5, 1], cax=cax4)

        axes2[1, 1].set_title('T1w')
        gt_check_slice = np.multiply(T1w_data[:, :, zz], original_icv_data[:, :, zz], (~original_csf_data[:, :, zz]).astype(int))
        im5 = axes2[1, 1].imshow(np.rot90(gt_check_slice), cmap="gray")
        divider5 = make_axes_locatable(axes2[1, 1])
        cax5 = divider5.append_axes("right", size="7%", pad=0.05)
        cbar5 = plt.colorbar(im5, cax=cax5)

        axes2[0, 2].set_title('Gray channel result')
        im7 = axes2[0, 2].imshow(np.rot90(gray_result[:, :, zz]), cmap="jet", vmin=0, vmax=1)
        divider7 = make_axes_locatable(axes2[0, 2])
        cax7 = divider7.append_axes("right", size="7%", pad=0.05)
        cbar7 = plt.colorbar(im7, ticks=[0, 0.5, 1], cax=cax7)

        axes2[1, 2].set_title('T2w')
        gt_check_slice = np.multiply(T2w_data[:, :, zz], original_icv_data[:, :, zz], (~original_csf_data[:, :, zz]).astype(int))
        im8 = axes2[1, 2].imshow(np.rot90(gt_check_slice), cmap="gray")
        divider8 = make_axes_locatable(axes2[1, 2])
        cax8 = divider8.append_axes("right", size="7%", pad=0.05)
        cbar8 = plt.colorbar(im8, cax=cax8)

        plt.tight_layout()
        # Make space for title
        plt.subplots_adjust(top=0.95)

        fig2.savefig(save_path_jpg, dpi=100)
        plt.close()


def save_agemap_img(slice_age_map_all, patch_size, img_slice, gt_slice, save_path, original_icv_data, original_csf_data, slice_name):
    ''' >>> Part 0 <<<'''
    slice_shape = img_slice.shape
    if slice_shape[0] == 3:

        img_slice = np.dstack([img_slice[0],img_slice[1],img_slice[2]])

        if slice_name == 'Lab':
            # Normalise Lab colour channel for Visualisation
            img_slice[:,:, 0] = np.multiply(img_slice[:,:,0]/100, original_icv_data, (~original_csf_data).astype(float))
            img_slice[:,:, 1] = np.multiply((img_slice[:,:,1]+128)/255,original_icv_data, (~original_csf_data).astype(float))
            img_slice[:,:, 2] = np.multiply((img_slice[:,:,2]+128)/255,original_icv_data, (~original_csf_data).astype(float))
            gt_check_slice = img_slice[:, :, 0].copy()
            gt_check_slice[gt_slice > 0] = 25
        elif slice_name == 'RGB':
            img_slice[:, :, 0] = np.multiply(img_slice[:, :, 0], original_icv_data, (~original_csf_data).astype(float))
            img_slice[:, :, 1] = np.multiply(img_slice[:, :, 1], original_icv_data, (~original_csf_data).astype(float))
            img_slice[:, :, 2] = np.multiply(img_slice[:, :, 2], original_icv_data, (~original_csf_data).astype(float))
            gt_check_slice = img_slice[:, :, 0].copy()
            gt_check_slice[gt_slice > 0] = 5000
            img_slice = img_slice/255.0
    else:
        img_slice = np.multiply(img_slice, original_icv_data, (~original_csf_data).astype(float))
        gt_check_slice = img_slice.copy()
        gt_check_slice[gt_slice > 0] = 5000

    ''' Show all age maps based on patch's size and saving the data '''
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
            im3 = axes[1, 0].imshow(np.rot90(slice_age_map_all[2, :, :]), cmap="jet", vmin=0, vmax=1)
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
    im5 = axes[0, 2].imshow(np.rot90(img_slice))
    divider5 = make_axes_locatable(axes[0, 2])
    cax5 = divider5.append_axes("right", size="7%", pad=0.05)
    cbar5 = plt.colorbar(im5, cax=cax5)

    axes[1, 2].set_title(slice_name + ' with WMH')
    im6 = axes[1, 2].imshow(np.rot90(gt_check_slice), cmap='jet')
    divider6 = make_axes_locatable(axes[1, 2])
    cax6 = divider6.append_axes("right", size="7%", pad=0.05)
    cbar6 = plt.colorbar(im6, cax=cax6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    fig.savefig(save_path, dpi=100)
    plt.close()


def iam_colour_channel_gpu_compute(data = "",  patch_size=[1, 2, 4, 8], blending_weights=[0.65, 0.2, 0.1, 0.05], alpha=0.5, save_jpeg=True, save_mat=True,
                                   modality_data=[], icv_data=[], csf_data=[], nawm_data=[], gt_data=[],num_samples=[],
                                   original_icv_data=[], original_csf_data=[],num_mean_samples=0, dirOutput=""):

    FLAIR_data = modality_data[0]
    T1w_data = modality_data[1]
    T2w_data = modality_data[2]

    [x_len, y_len, z_len] = FLAIR_data.shape
    rgb_combined_age_map_mri_mult = np.zeros((x_len, y_len, z_len))
    lab_combined_age_map_mri_mult = np.zeros((x_len, y_len, z_len))
    gray_combined_age_map_mri_mult = np.zeros((x_len, y_len, z_len))


    '''Remove non-brain area'''
    mask_brain = np.multiply(csf_data, icv_data).astype(float)

    for zz in range(0, FLAIR_data.shape[2]):
        print('\n---> Slice number: ' + str(zz) + ' <---')

        '''Load Image Slices'''
        icv_slice = icv_data[:, :, zz]
        csf_slice = csf_data[:, :, zz]
        nawm_slice = nawm_data[:, :, zz]
        gt_slice = gt_data[:, :, zz]
        mask_slice = mask_brain[:, :, zz]

        rgb_slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))
        gray_slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))
        lab_slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))
        penalty_slice = np.multiply(mask_slice,FLAIR_data[:,:,zz]).astype(float)  ### PENALTY

        '''Make new colour channel images using three modalities'''
        original_gray_slice = modality_to_gray(FLAIR_data[:,:,zz], T1w_data[:,:,zz], T2w_data[:,:,zz], [0.5, 0.25, 0.25])
        original_rgb_slice = modality_to_rgb(FLAIR_data[:,:,zz], T1w_data[:,:,zz], T2w_data[:,:,zz])
        original_lab_slice = rgb_to_lab(original_rgb_slice)

        gray_slice = np.multiply(original_gray_slice, mask_slice).astype(float)
        rgb_slice = np.multiply(original_rgb_slice, mask_slice).astype(float)
        lab_slice = np.multiply(original_lab_slice, mask_slice).astype(float)

        # Vol distance threshold
        vol_slice = np.count_nonzero(gray_slice) / (x_len * y_len)

        for enum, xy in enumerate(range(0, len(patch_size))):
            print('>>> Processing patch-size: ' + str(patch_size[xy]) + ' <<<')
            ##Save patch data
            '''
            if zz == 0:
                try:
                    dirOutData = dirOutput + '/' + data
                    os.makedirs(dirOutData + '/' + str(patch_size[xy]))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            '''
            counter_y = int(y_len / patch_size[xy])
            counter_x = int(x_len / patch_size[xy])
            rgb_age_values_all = np.zeros(counter_x * counter_y)
            lab_age_values_all = np.zeros(counter_x * counter_y)
            gray_age_values_all = np.zeros(counter_x * counter_y)

            valid = 0
            idx_debug = 0

            if ((vol_slice >= 0.008 and vol_slice < 0.035) and (patch_size[xy] == 1 or patch_size[xy] == 2)) or \
                ((vol_slice >= 0.035 and vol_slice < 0.065) and (patch_size[xy] == 1 or patch_size[xy] == 2 or patch_size[xy] == 4)) or (vol_slice > 0.065):

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

                ''' Extract Source and Target Patches '''
                rgb_source_patch, icv_source_flag_valid, index_mapping = extract_source_patches([counter_x, counter_y], mask_slice, x_c_sources,
                                                                                                   y_c_sources, patch_size[xy], rgb_slice)
                rgb_age_values_valid = np.zeros(len(icv_source_flag_valid))
                gray_source_patch, icv_source_flag_valid, index_mapping = extract_source_patches([counter_x, counter_y], mask_slice, x_c_sources,
                                                                                                 y_c_sources, patch_size[xy], gray_slice)
                gray_age_values_valid = np.zeros(len(icv_source_flag_valid))
                lab_source_patch, icv_source_flag_valid, index_mapping = extract_source_patches([counter_x, counter_y], mask_slice, x_c_sources,
                                                                                                y_c_sources, patch_size[xy], lab_slice)
                lab_age_values_valid = np.zeros(len(icv_source_flag_valid))

                rgb_target_patches, lab_target_patches, gray_target_patches, idx_debug = extract_target_patches(x_len, y_len, FLAIR_TRSH, mask_slice, rgb_slice, lab_slice, gray_slice, patch_size[xy])

                '''Calculate Age Values'''
                if len(rgb_target_patches) > 0:
                    rgb_age_values_valid = calculate_age_value(rgb_source_patch, rgb_target_patches, num_samples,
                                                               icv_source_flag_valid, num_mean_samples,alpha, rgb_age_values_valid)
                    rgb_age_values_valid_norm = mapping_result(index_mapping, rgb_age_values_all, rgb_age_values_valid, False)
                    rgb_slice_age_map_all[xy, :, :] = age_value_mapping(np.reshape(rgb_age_values_valid_norm, [counter_x, counter_y]), patch_size[xy],icv_slice)

                    lab_age_values_valid = calculate_age_value(lab_source_patch, lab_target_patches,
                                                               num_samples,
                                                               icv_source_flag_valid, num_mean_samples,
                                                               alpha, lab_age_values_valid)
                    lab_age_values_valid_norm = mapping_result(index_mapping, lab_age_values_all, lab_age_values_valid, False)
                    lab_slice_age_map_all[xy, :, :] = age_value_mapping(np.reshape(lab_age_values_valid_norm, [counter_x, counter_y]), patch_size[xy], icv_slice)

                    gray_age_values_valid = calculate_age_value(gray_source_patch, gray_target_patches,
                                                               num_samples,
                                                               icv_source_flag_valid, num_mean_samples,
                                                               alpha, gray_age_values_valid)
                    gray_age_values_valid_norm = mapping_result(index_mapping, gray_age_values_all, gray_age_values_valid, False)
                    gray_slice_age_map_all[xy, :, :] = age_value_mapping(np.reshape(gray_age_values_valid_norm, [counter_x, counter_y]), patch_size[xy], icv_slice)

            numba.cuda.profile_stop()

            print('Sampling finished: ' + ' with: FLAIR ' + str(idx_debug))
            if not valid:
                rgb_slice_age_map_all[xy, :, :] = lab_slice_age_map_all[xy, :, :] = gray_slice_age_map_all[xy, :, :] \
                    = cv2.resize(np.zeros([counter_x, counter_y]), None, fx=patch_size[xy], fy=patch_size[xy], interpolation=cv2.INTER_CUBIC)

        rgb_slice_age_map_all = np.nan_to_num(rgb_slice_age_map_all)
        lab_slice_age_map_all = np.nan_to_num(lab_slice_age_map_all)
        gray_slice_age_map_all = np.nan_to_num(gray_slice_age_map_all)

        if save_jpeg:
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Patch/' + str(zz) + '_RGB_all.jpg'
            save_agemap_img(rgb_slice_age_map_all, patch_size, original_rgb_slice, gt_slice, save_path,
                            original_icv_data[:, :, zz], original_csf_data[:, :, zz], 'RGB')
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Patch/' + str(zz) + '_Lab_all.jpg'
            save_agemap_img(lab_slice_age_map_all, patch_size, original_lab_slice, gt_slice, save_path,
                            original_icv_data[:, :, zz], original_csf_data[:, :, zz], 'Lab')
            save_path = dirOutput + '/' + data + '/IAM_combined_python/Patch/' + str(zz) + '_Gray_all.jpg'
            save_agemap_img(gray_slice_age_map_all, patch_size, original_gray_slice, gt_slice, save_path,
                            original_icv_data[:, :, zz], original_csf_data[:, :, zz], 'Gray')

        ## Save data
        if save_mat:
            save_path = dirOutput + '/' + data + '/IAM_combined_python/ColorChannel_Result/Data/'
            sio.savemat(save_path + str(zz)+ '_dat.mat', {'rgb_slice_age_map': rgb_slice_age_map_all, 'lab_slice_age_map': lab_slice_age_map_all,
                                                          'gray_slice_age_map': gray_slice_age_map_all})

        penalty_rgb_age_map = FLAIR_penalisation(patch_size, blending_weights, rgb_slice_age_map_all, penalty_slice, icv_slice)
        rgb_combined_age_map_mri_mult[:, :, zz] = penalty_rgb_age_map
        penalty_lab_age_map = FLAIR_penalisation(patch_size,blending_weights,lab_slice_age_map_all,penalty_slice, icv_slice)
        lab_combined_age_map_mri_mult[:, :, zz] = penalty_lab_age_map
        penalty_gray_age_map = FLAIR_penalisation(patch_size, blending_weights, gray_slice_age_map_all,penalty_slice, icv_slice)
        gray_combined_age_map_mri_mult[:, :, zz] = penalty_gray_age_map

    ''' >>> Part 2 <<< '''
    ''' Penalty + Global Normalisation (GN) '''
    rgb_combined_age_map_mri_mult_normed = normalisation(rgb_combined_age_map_mri_mult)
    lab_combined_age_map_mri_mult_normed = normalisation(lab_combined_age_map_mri_mult)
    gray_combined_age_map_mri_mult_normed = normalisation(gray_combined_age_map_mri_mult)

    if save_jpeg:
        save_path = dirOutput + '/' + data + '/IAM_combined_python/ColorChannel_Result/Image/'
        save_result_img(rgb_combined_age_map_mri_mult_normed, rgb_combined_age_map_mri_mult_normed, gray_combined_age_map_mri_mult_normed,
                        gt_data, FLAIR_data, T1w_data, T2w_data, original_icv_data, original_csf_data, save_path)

    del icv_slice, csf_slice
    del FLAIR_data, icv_data, csf_data
    del original_icv_data, original_csf_data
    del icv_source_flag_valid, index_mapping
    gc.collect()

    return gray_combined_age_map_mri_mult_normed, rgb_combined_age_map_mri_mult_normed, lab_combined_age_map_mri_mult_normed

