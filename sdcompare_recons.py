import skimage.metrics as ski_metrics
import skimage as ski
from statistics import mean
from glob import glob
import numpy as np
import os
#import pdb


true_img_dir = '/mmfs1/gscratch/uwb/CT_images/recons2024/900views'
sv60_img_dir = '/mmfs1/gscratch/uwb/CT_images/recons2024/60views'
dncnn_img_dir = 'new_dncnn_data/reconstructed_images3'
transformer_img_dir = 'transformer_data'
unet_img_dir = 'new_unet_data'

true_img_list = sorted(glob(true_img_dir + '/*.flt'))
sv60_img_list = sorted(glob(sv60_img_dir + '/*.flt'))
dncnn_img_list = sorted(glob(dncnn_img_dir + '/*.flt'))
transformer_img_list = sorted(glob(transformer_img_dir + '/*.flt'))
unet_img_list = sorted(glob(unet_img_dir + '/*.flt'))

# Debugging: Print the lengths of the lists
print(f"Number of true images: {len(true_img_list)}")
print(f"Number of SV60 images: {len(sv60_img_list)}")
print(f"Number of DnCNN images: {len(dncnn_img_list)}")
print(f"Number of Transformer images: {len(transformer_img_list)}")
print(f"Number of Unet images: {len(unet_img_list)}")

psnr_values_sv60, psnr_values_dncnn, psnr_values_transformer, psnr_values_unet = [], [], [], []
ssim_values_sv60, ssim_values_dncnn, ssim_values_transformer, ssim_values_unet = [], [], [], []

for (true_file, sv60_file, dncnn_file, transformer_file,unet_file) in zip(true_img_list, sv60_img_list, dncnn_img_list, transformer_img_list, unet_img_list):
    print(true_file)
    true = np.fromfile(true_file, dtype='f')
    max_val = np.amax(true)
    true = true.reshape(512, 512)

    # sv60
    recon_sv60 = np.fromfile(sv60_file, dtype='f')
    recon_sv60 = recon_sv60.reshape(512, 512)
    psnr_values_sv60.append(ski_metrics.peak_signal_noise_ratio(true, recon_sv60, data_range=max_val))
    ssim_values_sv60.append(ski_metrics.structural_similarity(true, recon_sv60, data_range=max_val))

    # dncnn
    recon_dncnn = np.fromfile(dncnn_file, dtype='f')
    recon_dncnn = recon_dncnn.reshape(512, 512)
    psnr_values_dncnn.append(ski_metrics.peak_signal_noise_ratio(true, recon_dncnn, data_range=max_val))
    ssim_values_dncnn.append(ski_metrics.structural_similarity(true, recon_dncnn, data_range=max_val))

    # transformer
    recon_transformer = np.fromfile(transformer_file, dtype='f')
    recon_transformer = recon_transformer.reshape(512, 512)
      psnr_values_transformer.append(ski_metrics.peak_signal_noise_ratio(true, recon_transformer, data_range=max_val))
    ssim_values_transformer.append(ski_metrics.structural_similarity(true, recon_transformer, data_range=max_val))

    #unet
    recon_unet = np.fromfile(unet_file,dtype='f')
    recon_unet = recon_unet.reshape(512,512)
    psnr_values_unet.append(ski.metrics.peak_signal_noise_ratio(true,recon_unet,data_range=max_val))
    ssim_values_unet.append(ski.metrics.structural_similarity(true,recon_unet,data_range=max_val))

   # pdb.set_trace()

# Calculate mean and standard deviation for PSNR values
mean_psnr_sv60 = np.mean(psnr_values_sv60)
std_psnr_sv60 = np.std(psnr_values_sv60)
mean_psnr_dncnn = np.mean(psnr_values_dncnn)
std_psnr_dncnn = np.std(psnr_values_dncnn)
mean_psnr_transformer = np.mean(psnr_values_transformer)
std_psnr_transformer = np.std(psnr_values_transformer)
mean_psnr_unet = np.mean(psnr_values_unet)
std_psnr_unet = np.std(psnr_values_unet)

# Calculate mean and standard deviation for SSIM values
mean_ssim_sv60 = np.mean(ssim_values_sv60)
std_ssim_sv60 = np.std(ssim_values_sv60)
mean_ssim_dncnn = np.mean(ssim_values_dncnn)
std_ssim_dncnn = np.std(ssim_values_dncnn)
mean_ssim_transformer = np.mean(ssim_values_transformer)
std_ssim_transformer = np.std(ssim_values_transformer)
mean_ssim_unet = np.mean(ssim_values_unet)
std_ssim_unet = np.std(ssim_values_unet)


print(f"SV60 - Mean PSNR: {mean_psnr_sv60}, Std PSNR: {std_psnr_sv60}")
print(f"SV60 - Mean SSIM: {mean_ssim_sv60}, Std SSIM: {std_ssim_sv60}")
print(f"DnCNN - Mean PSNR: {mean_psnr_dncnn}, Std PSNR: {std_psnr_dncnn}")
print(f"DnCNN - Mean SSIM: {mean_ssim_dncnn}, Std SSIM: {std_ssim_dncnn}")
print(f"Transformer - Mean PSNR: {mean_psnr_transformer}, Std PSNR: {std_psnr_transformer}")
print(f"Transformer - Mean SSIM: {mean_ssim_transformer}, Std SSIM: {std_ssim_transformer}")
print(f"Unet - Mean PSNR: {mean_psnr_unet}, Std PSNR: {std_psnr_unet}")
print(f"Unet - Mean SSIM: {mean_ssim_unet}, Std SSIM: {std_ssim_unet}")

                                                                                            
