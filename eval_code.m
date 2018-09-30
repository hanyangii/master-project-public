DATA={'002_S_0413_2013';
'002_S_0413_2014';
'002_S_4799_2013';
'002_S_4799_2014';
'009_S_0751_2013';
'009_S_0751_2014';
'018_S_2133_2012';
'018_S_2133_2014';
'024_S_2239_2013';
'024_S_2239_2014';
'031_S_4005_2014';
'035_S_2061_2013';
'035_S_2061_2014';
'035_S_2074_2013';
'035_S_2074_2014';
'035_S_4114_2013';
'035_S_4114_2014';
'041_S_4004_2013';
'098_S_2079_2013';
'098_S_2079_2014';
'137_S_0301_2013';
'137_S_0301_2014';
'137_S_0668_2013';
'137_S_0668_2014';
'137_S_0722_2013';
'137_S_0722_2014';
'137_S_0800_2013';
'137_S_0800_2014';
'137_S_1414_2013';
'137_S_1414_2014' 
}
dir_dat_original = '/mnt/Storage/ADNI_20x3_2015/';
dir_dat_lga005 = '/mnt/Storage/ToolboxResults/niftiimages_LST/';
dir_iam_gpu = '/mnt/Storage/ADNI_20x3_2015/IAM_GPU_MRI_CUDAv7_512s64m/';

sub_gt_all = zeros(length(DATA), 1);
sub_res_tool = zeros(length(DATA), 1);
sub_res_dm = zeros(length(DATA), 1);
sub_res_iam = zeros(length(DATA), 1);

Y = cell(length(DATA),1);
X_lga = cell(length(DATA),1);
X_iam_gpu = cell(length(DATA),1);


sourcePath = '/mnt/Storage/Results2016';
addpath(genpath(sourcePath));

for ii = 1 : length(DATA)
    %fprintf('MRI data name: %s\n', DATA{ii});
    
    icv_data = load_series([dir_dat_original, DATA{ii}, '/ICV_cerebrum_cleaned'], []);
    wmh_data = load_series([dir_dat_original, DATA{ii}, '/WMH_final_nifti_v2'], []); 
    
    % Load results
    iam_gpu_data_postpro = load_series([dir_iam_gpu, DATA{ii}, ...
        '/IAM_GPU_nifti_postprocessed/IAM_GPU_result_def_postprocessed'], []);
        
    lga_data = load_series([dir_dat_lga005, DATA{ii}, '/ples_lga_0.05_rmFLAIR_registered.nii'], []);
    
    mask_hdr = load_series([dir_dat_original, DATA{ii}, '/WMH_final_nifti_v2'],0);
    vox = [mask_hdr.hdr.dime.pixdim(2),mask_hdr.hdr.dime.pixdim(3),mask_hdr.hdr.dime.pixdim(4)];
    
    % Get volume
    wmh_data = wmh_data .* icv_data;
    wmh_vol = sum(wmh_data(:)) * prod(vox);
    if wmh_vol <= 4500
        disp(strcat(DATA(ii),' Small'));
    elseif wmh_vol <= 13000 & wmh_vol > 4500
        disp(strcat(DATA(ii),' Medium'));
    elseif wmh_vol > 13000
        disp(strcat(DATA(ii),' Large'));
    else
        disp(strcat(DATA(ii),' ERROR'));
    end
end
