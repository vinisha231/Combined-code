import skimage.metrics as ski_metrics
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import os

# Define directories, these input values are garbage
true_img_dir = '/gscratch/uwb/CT_images/recons2024/900views'
input_type_img_dir = '/gscratch/uwb/limitedangles/limitedangletest'
model_img_dir = '/gscratch/uwb/bodhik/CT-CNN-Code/refactor_main/unet_nD_data_python3.12.4/flt_dir'
model_name = 'Unet'
#--------------------INPUT ABOVE THIS LINE------------------------------

#calculates PSNR and SSIM values for a single model given its output flt directory and the target output flt directory

# Collect images
true_img_list = sorted(glob(os.path.join(true_img_dir, '*.flt')))#'000036[2-3]*.flt')))
model_img_list = sorted(glob(os.path.join(unet_img_dir, '*.flt')))#'000036[2-3]*.flt')))
input_type_img_list = sorted(glob(os.path.join(input_type_img_dir, '*.flt')))#'000036[2-3]*.flt')))

psnr_values_model = []
ssim_values_model = []

# Process and plot images
for idx, (true_file,model_file,input_type_file) in enumerate(zip(true_img_list, model_img_list, input_type_img_list)):
    
    true = np.fromfile(true_file, dtype='f').reshape(512, 512)
    max_val = np.amax(true)
    print("true maxval: ", max_val)

    input_type = np.fromfile(input_type_file, dtype='f').reshape(512, 512)
    print("input_type  maxval: ", np.amax(input_type))
	
    # Unet
    recon_model = np.fromfile(model_file, dtype='f').reshape(512, 512)
    print("recon_model maxval: ", np.amax(recon_model))

    psnr_values_model.append(ski_metrics.peak_signal_noise_ratio(true, recon_model, data_range=max_val))
    ssim_values_model.append(ski_metrics.structural_similarity(true, recon_model, data_range=max_val))

# Calculate mean and standard deviation for PSNR values
mean_psnr_model = np.mean(psnr_values_model)
std_psnr_model = np.std(psnr_values_model)

# Calculate mean and standard deviation for SSIM values
mean_ssim_model = np.mean(ssim_values_model)
std_ssim_model = np.std(ssim_values_model)

print(f"{model_name} == Mean PSNR: {mean_psnr_model:1.4f}, Std PSNR: {std_psnr_model:1.4f}")
print(f"{model_name} == Mean SSIM: {mean_ssim_model:1.4f}, Std SSIM: {std_ssim_model:1.4f}")
