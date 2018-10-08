import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

import csv, os, errno

# Turn interactive plotting off
# plt.ioff()

'''Parameters'''
data_home_dir = 'W:\BRICIA\\resources\Cesca_MSS2_sample_50\\'
data_csv_dir = 'W:\BRICIA\\resources\Cesca_MSS2_sample_50\data_list.csv'
data_dir_list = os.listdir(data_home_dir)
data_dir_prefix = 'MSSB'
data_file_pattern = ['_1yr','_baseline']
nii_file_list = ['_FLAIR.nii.gz','_T1W.nii.gz','_T2W.nii.gz','_Final_mask.nii.gz']

def ICV_slice_processing(ICV_mark):
    mask_size = ICV_mark.shape
    for i in range(mask_size[2]):
        ICV_mark[:,:,i] = ndimage.binary_closing(ICV_mark[:,:,i], np.ones([10,10]))
        ICV_mark[:,:,i] = ndimage.binary_fill_holes(ICV_mark[:,:,i]).astype(int)
    return ICV_mark

def main():
    with open(data_csv_dir,'w') as csv_file:
        csv_writer = csv.writer(csv_file, dialect='excel')
        for data in data_dir_list:
            if data.startswith(data_dir_prefix):

                nii_dir = data_home_dir + data + '\\' + 'lots_iam_data\\'
                if not os.path.exists(nii_dir):
                    try:
                        os.makedirs(nii_dir)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise

                print('Processing data %s'%(data))
                for pattern in data_file_pattern:
                    data_dir = data_home_dir + data + '\\' + data + pattern
                    final_mask_dir = data_dir+nii_file_list[3]
                    if os.path.exists(final_mask_dir):
                        final_mask_nii = nib.load(final_mask_dir)
                        final_mask_data = np.squeeze(final_mask_nii.get_data())

                        '''Produce Each Mark'''

                        #Stroke
                        stroke_mark = np.zeros(final_mask_data.shape)
                        stroke_mark[final_mask_data == 6] = 1
                        stroke_mark[final_mask_data == 7] = 1

                        #WMH
                        WMH_mark = np.zeros(final_mask_data.shape)
                        WMH_mark[final_mask_data == 3] = 1
                        WMH_mark[final_mask_data == 4] = 1

                        #Gray Matter
                        GM_mark = np.zeros(final_mask_data.shape)
                        GM_mark[final_mask_data == 5] = 1

                        #ICV
                        ICV_mark = np.zeros(final_mask_data.shape)
                        ICV_mark[final_mask_data > 0] = 1
                        ICV_mark = ICV_slice_processing(ICV_mark)

                        #CSF
                        nonCSF_mark = np.zeros(final_mask_data.shape)
                        nonCSF_mark[final_mask_data == 2] = 1
                        nonCSF_mark[final_mask_data == 5] = 1
                        nonCSF_mark[stroke_mark == 1] = 1
                        nonCSF_mark[WMH_mark == 1] = 1
                        CSF_mark = ICV_mark.copy()
                        CSF_mark[nonCSF_mark==1] = 0

                        
                        ICV_nii = nib.Nifti1Image(ICV_mark, final_mask_nii.affine)
                        nib.save(ICV_nii, str(nii_dir + data+ pattern +'_ICV.nii.gz'))
                        CSF_nii = nib.Nifti1Image(CSF_mark, final_mask_nii.affine)
                        nib.save(CSF_nii, str(nii_dir + data+ pattern + '_CSF.nii.gz'))
                        WMH_nii = nib.Nifti1Image(WMH_mark, final_mask_nii.affine)
                        nib.save(WMH_nii, str(nii_dir + data+ pattern + '_WMH.nii.gz'))

                        '''Save ICV and CSF marks as nii files'''

                        '''Write CSV file'''
                        #FLAIR, T1w, T2w, ICV, CSF, WMH_label
                        csv_writer.writerow((data+pattern,
                                             data_dir+'_str'+nii_file_list[0],
                                            data_dir+'_str'+nii_file_list[1],
                                            data_dir+'_str'+nii_file_list[2],
                                            nii_dir + data + pattern + '_ICV.nii.gz',
                                            nii_dir + data + pattern + '_CSF.nii.gz',
                                            nii_dir + data + pattern + '_WMH.nii.gz'))
                    else:
                        print(data+pattern+' final mask does not exist...')

if __name__ == "__main__":
    main()
