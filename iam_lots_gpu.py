#!/usr/bin/python

from Three_channel import *
from Threshold_test import *
from iam_params import *

import sys

def dsc_evaluation(results,original_csf_data, original_icv_data, gt_map):

    approaches = []
    total_dsc = []

    for approach in results:
        result = results[approach]
        results[approach] = result[np.multiply(~original_csf_data, original_icv_data)>0]

    penalty_dsc = [0 for zero in range(len(results))]
    trsh = [tr / 10.0 for tr in range(1, 11)]

    for idd, dsc_trsh in enumerate(trsh):
        print_line = '\ntrsh %.2f : '%(dsc_trsh)
        for idx, approach in enumerate(results):
            penalty_dsc[idx] = calculate_dsc(gt_map, results[approach], dsc_trsh)
            print_line = print_line+approach+' - %.2f // '%(penalty_dsc[idx])
            if idd == 0: approaches.append(approach)
        total_dsc.append(penalty_dsc.copy())


    total_dsc = np.reshape(np.array(total_dsc),[len(results), 10])
    total_dsc = np.reshape(total_dsc,[1,-1])

    return total_dsc[0], approaches

def main():
    print('Check OpenCV version: ' + cv2.__version__ + '\n')
    print(cuda.current_context().get_memory_info())
    print('Initialisation is done..\n')

    ## NOTE: Put parameters in iam_params.py
    ## Parameters are loaded by "from iam_params import *" line


    total_dsc_avg = []

    num_mean_samples_all, num_samples_all = initial_check(output_filedir, csv_filename, patch_size, blending_weights,
                                                          delete_intermediary, num_samples_all_param)

    print('CSV data filename: ' + csv_filename)
    print('Patch size(s): ' + str(patch_size))

    for ii_s in range(0, len(num_samples_all)):
        num_samples = num_samples_all[ii_s]
        num_mean_samples = num_mean_samples_all[ii_s]
        dirOutput = output_filedir + '_' + str(num_samples) + 's' + str(num_mean_samples) + 'm'
        try:
            os.makedirs(dirOutput)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        with open(csv_filename, newline='') as csv_file:
            num_subjects = len(csv_file.readlines())
            print('Number of subject(s): ' + str(num_subjects))

        with open(csv_filename, newline='') as csv_file:
            reader = csv.reader(csv_file)
            timer_idx = 0
            elapsed_times_all = np.zeros((num_subjects))
            elapsed_times_patch_all = np.zeros((num_subjects, len(patch_size)))

            for row in reader:
                one_data = timer()
                data = row[0]

                results = {}

                # Get Data
                modality_data, icv_data, csf_data, gt_data, mri_nii = data_load(dirOutput, data, row, bin_tresh, colour_channel_algorithm, T_weighted_penalisation_algorithm)
                # Data Preprocessing
                original_icv_data, original_csf_data, csf_data, icv_data = preprocessing(icv_data, csf_data)
                gt_map = gt_data[np.multiply(~original_csf_data, original_icv_data) > 0]

                '''Colour channel approach'''
                if colour_channel_algorithm:
                    print(data, ' Colour Channel Approach')
                    gray_result, rgb_result, lab_result = iam_colour_channel_gpu_compute(data = data,
                                                   patch_size = patch_size,
                                                   blending_weights = blending_weights,
                                                   alpha = alpha,
                                                   save_jpeg = save_jpeg,
                                                   save_mat = save_mat,
                                                   modality_data = modality_data,
                                                   icv_data = icv_data,
                                                   csf_data = csf_data,
                                                   gt_data = gt_data,
                                                   num_samples = num_samples,
                                                   original_icv_data = original_icv_data,
                                                   original_csf_data = original_csf_data,
                                                   num_mean_samples = num_samples,
                                                   dirOutput=dirOutput)

                    results['RGB']=rgb_result
                    results['Lab']=lab_result
                    results['Gray']=gray_result

                    ## Save data
                    if save_mat:
                        save_path = dirOutput + '/' + data + '/IAM_GPU_nifti_python/'
                        combined_age_map_mri_GN_img = nib.Nifti1Image(rgb_result, mri_nii.affine)
                        nib.save(combined_age_map_mri_GN_img, str(save_path + '/IAM_rgb.nii.gz'))
                        combined_age_map_mri_GN_img = nib.Nifti1Image(lab_result, mri_nii.affine)
                        nib.save(combined_age_map_mri_GN_img, str(save_path + '/IAM_lab.nii.gz'))
                        combined_age_map_mri_GN_img = nib.Nifti1Image(gray_result, mri_nii.affine)
                        nib.save(combined_age_map_mri_GN_img, str(save_path + '/IAM_gray.nii.gz'))


                '''Penalisation approach'''
                if T_weighted_penalisation_algorithm:
                    print(data, ' Penalisation Approach')
                    alg1_result, alg2_result = iam_Penalisation_gpu_compute(data = data,
                                                   patch_size = patch_size,
                                                   blending_weights = blending_weights,
                                                   alpha = alpha,
                                                   save_jpeg = save_jpeg,
                                                   save_mat = save_mat,
                                                   modality_data = modality_data,
                                                   icv_data = icv_data,
                                                   csf_data = csf_data,
                                                   gt_data = gt_data,
                                                   num_samples = num_samples,
                                                   original_icv_data = original_icv_data,
                                                   original_csf_data = original_csf_data,
                                                   num_mean_samples = num_samples,
                                                   dirOutput=dirOutput,
                                                   trsh = Ttrsh)

                    results['Alg1']=alg1_result
                    results['Alg2']=alg2_result

                    ## Save data
                    if save_mat:
                        save_path = dirOutput + '/' + data + '/IAM_GPU_nifti_python/'
                        combined_age_map_mri_GN_img = nib.Nifti1Image(alg1_result, mri_nii.affine)
                        nib.save(combined_age_map_mri_GN_img, str(save_path + '/IAM_alg1.nii.gz'))
                        combined_age_map_mri_GN_img = nib.Nifti1Image(alg2_result, mri_nii.affine)
                        nib.save(combined_age_map_mri_GN_img, str(save_path + '/IAM_alg2.nii.gz'))

                elapsed_times_all[timer_idx] = timer() - one_data
                timer_idx += 1

                '''>>> Part 3 : Evaluation<<<'''

                total_dsc, approaches = dsc_evaluation(results, original_csf_data, original_icv_data, gt_map)
                total_dsc_avg.append(total_dsc)

    if delete_intermediary:
        dirOutDataCom = dirOutput + '/' + data + '/IAM_combined_python/'
        shutil.rmtree(dirOutDataCom, ignore_errors=True)

    '''Result Visualisation'''
    total_dsc_avg = np.mean(np.array(total_dsc_avg), 0)
    total_dsc_avg = np.reshape(total_dsc_avg,[10,len(approaches)])
    trsh = [tr / 10.0 for tr in range(1, 10)]
    print(total_dsc_avg)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> Final Average Result <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    for idd, dsc_trsh in enumerate(trsh):
        print_line = '\ntrsh %.2f : ' % (dsc_trsh)
        for idx, approach in enumerate(approaches):
            print_line = print_line + approach + ' - %.2f // ' % (total_dsc_avg[idd,idx])
        print(print_line)

    ##Graph Plot

    dsc_range = np.arange(0.1, 1.1, 0.1)
    linespec = ['ro-', 'g*-', 'b^-', 'kd-', 'm.-']

    
    fi = plt.figure()
    plt.ylim(ymin=0, ymax=0.5)

    for idx in range(len(approaches)):
        plt.plot(dsc_range, total_dsc_avg[:,idx], linespec[idx],label=approaches[idx])
    plt.legend()
    plt.title('DSC scores')
    fi.savefig(dirOutput + '/final_result_graph.jpg')

    ## Print the elapsed time information
    print('\n--\nSpeed statistics of this run..')
    print('mean elapsed time  : ' + str(np.mean(elapsed_times_all)) + ' seconds')
    print('std elapsed time   : ' + str(np.std(elapsed_times_all)) + ' seconds')
    print('median elapsed time : ' + str(np.median(elapsed_times_all)) + ' seconds')
    print('min elapsed time   : ' + str(np.min(elapsed_times_all)) + ' seconds')
    print('max elapsed time   : ' + str(np.max(elapsed_times_all)) + ' seconds')


if __name__ == "__main__":
    main()
