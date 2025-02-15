import skimage.metrics as ski_metrics
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import os

# Define directories
true_img_dir = '/mmfs1/gscratch/uwb/CT_images/recons2024/900views'
sv60_img_dir = '/mmfs1/gscratch/uwb/CT_images/recons2024/60views'
#limang_img_dir = '/mmfs1/gscratch/uwb/limitedangles/limitedangletest'
dncnn_img_dir = '/mmfs1/gscratch/uwb/CT_code/output3/reconstructed_images3'
transformer_img_dir = '/mmfs1/gscratch/uwb/CT_code/output1/reconstructed_images1'
unet_img_dir = '/mmfs1/gscratch/uwb/CT_code/output2/reconstructed_images2'
#SARTdncnn_img_dir = '/mmfs1/gscratch/uwb/plugandplay/DNCNN_pnp_Output'
#SARTtransformer_img_dir = '/mmfs1/gscratch/uwb/plugandplay/Transformer_pnp_Output'
#SARTunet_img_dir = '/mmfs1/gscratch/uwb/plugandplay/Unet_pnp_Output'

# Collect images
true_img_list = sorted(glob(os.path.join(true_img_dir, '*.flt')))
sv60_img_list = sorted(glob(os.path.join(sv60_img_dir, '*.flt')))
dncnn_img_list = sorted(glob(os.path.join(dncnn_img_dir, '*.flt')))
transformer_img_list = sorted(glob(os.path.join(transformer_img_dir, '*.flt')))
unet_img_list = sorted(glob(os.path.join(unet_img_dir, '*.flt')))
#SARTdncnn_img_list = sorted(glob(os.path.join(SARTdncnn_img_dir, '*.flt')))
#SARTtransformer_img_list = sorted(glob(os.path.join(SARTtransformer_img_dir, '*.flt')))
#SARTunet_img_list = sorted(glob(os.path.join(SARTunet_img_dir, '*.flt')))

# Debugging: Print the lengths of the lists
print(f"Number of true images: {len(true_img_list)}")
print(f"Number of SV60 images: {len(sv60_img_list)}")
print(f"Number of DnCNN images: {len(dncnn_img_list)}")
print(f"Number of Transformer images: {len(transformer_img_list)}")
print(f"Number of Unet images: {len(unet_img_list)}")
#print(f"Number of SART DnCNN images: {len(SARTdncnn_img_list)}")
#print(f"Number of SART Transformer images: {len(SARTtransformer_img_list)}")
#print(f"Number of SART Unet images: {len(SARTunet_img_list)}")

psnr_values_sv60, psnr_values_dncnn, psnr_values_transformer, psnr_values_unet = [], [], [], []
ssim_values_sv60, ssim_values_dncnn, ssim_values_transformer, ssim_values_unet = [], [], [], []

#psnr_values_SARTdncnn, psnr_values_SARTtransformer, psnr_values_SARTunet = [], [], []
#ssim_values_SARTdncnn, ssim_values_SARTtransformer, ssim_values_SARTunet = [], [], []

# Process and plot images
#for idx, (true_file, sv60_file, dncnn_file, transformer_file, unet_file, SARTdncnn_file, SARTtransformer_file, SARTunet_file) in enumerate(
#        zip(true_img_list, sv60_img_list, dncnn_img_list, transformer_img_list, unet_img_list, SARTdncnn_img_list, SARTtransformer_img_list, SARTunet_img_list)):
for idx,(true_file, sv60_file, dncnn_file, transformer_file, unet_file) in enumerate(
         zip(true_img_list, sv60_img_list, dncnn_img_list, transformer_img_list, unet_img_list)):
    print(true_file)
    # Skip if idx >= 8, to limit the first 8 images for each set
    # if idx >= 8:
    #     break
    true = np.fromfile(true_file, dtype='f').reshape(512, 512)
    max_val = np.amax(true)


    # SV60
    recon_sv60 = np.fromfile(sv60_file, dtype='f').reshape(512, 512)
    psnr_values_sv60.append(ski_metrics.peak_signal_noise_ratio(true, recon_sv60, data_range=max_val))
    ssim_values_sv60.append(ski_metrics.structural_similarity(true, recon_sv60, data_range=max_val))

    # DnCNN
    recon_dncnn = np.fromfile(dncnn_file, dtype='f').reshape(512, 512)
    psnr_values_dncnn.append(ski_metrics.peak_signal_noise_ratio(true, recon_dncnn, data_range=max_val))
    ssim_values_dncnn.append(ski_metrics.structural_similarity(true, recon_dncnn, data_range=max_val))

    # Transformer
    recon_transformer = np.fromfile(transformer_file, dtype='f').reshape(512, 512)
    psnr_values_transformer.append(ski_metrics.peak_signal_noise_ratio(true, recon_transformer, data_range=max_val))
    ssim_values_transformer.append(ski_metrics.structural_similarity(true, recon_transformer, data_range=max_val))

    # Unet
    recon_unet = np.fromfile(unet_file, dtype='f').reshape(512, 512)
    psnr_values_unet.append(ski_metrics.peak_signal_noise_ratio(true, recon_unet, data_range=max_val))
    ssim_values_unet.append(ski_metrics.structural_similarity(true, recon_unet, data_range=max_val))

    # SART DnCNN
    #recon_SARTdncnn = np.fromfile(SARTdncnn_file, dtype='f').reshape(512, 512)
    #psnr_values_SARTdncnn.append(ski_metrics.peak_signal_noise_ratio(true, recon_SARTdncnn, data_range=max_val))
    #ssim_values_SARTdncnn.append(ski_metrics.structural_similarity(true, recon_SARTdncnn, data_range=max_val))

    # SART Transformer
    #recon_SARTtransformer = np.fromfile(SARTtransformer_file, dtype='f').reshape(512, 512)
    #psnr_values_SARTtransformer.append(ski_metrics.peak_signal_noise_ratio(true, recon_SARTtransformer, data_range=max_val))
    #ssim_values_SARTtransformer.append(ski_metrics.structural_similarity(true, recon_SARTtransformer, data_range=max_val))

    # SART Unet
    #recon_SARTunet = np.fromfile(SARTunet_file, dtype='f').reshape(512, 512)
    #psnr_values_SARTunet.append(ski_metrics.peak_signal_noise_ratio(true, recon_SARTunet, data_range=max_val))
    #ssim_values_SARTunet.append(ski_metrics.structural_similarity(true, recon_SARTunet, data_range=max_val))

    # Plot images
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    axes = axes.flatten()

    plot_idx = 0
    axes[plot_idx].imshow(true, cmap='gray')
    axes[plot_idx].set_title(f"True Image {idx+1}")
    axes[plot_idx].axis('off')

    axes[plot_idx + 4].imshow(recon_sv60, cmap='gray')
    axes[plot_idx + 4].set_title(f"SV60 Recon {idx+1}")
    axes[plot_idx + 4].axis('off')

    axes[plot_idx + 1].imshow(recon_dncnn, cmap='gray')
    axes[plot_idx + 1].set_title(f"DnCNN Recon {idx+1}")
    axes[plot_idx + 1].axis('off')

    axes[plot_idx + 2].imshow(recon_transformer, cmap='gray')
    axes[plot_idx + 2].set_title(f"Transformer Recon {idx+1}")
    axes[plot_idx + 2].axis('off')

    axes[plot_idx + 3].imshow(recon_unet, cmap='gray')
    axes[plot_idx + 3].set_title(f"Unet Recon {idx+1}")
    axes[plot_idx + 3].axis('off')

    #axes[plot_idx + 5].imshow(recon_SARTdncnn, cmap='gray')
    #axes[plot_idx + 5].set_title(f"SART DnCNN Recon {idx+1}")
    #axes[plot_idx + 5].axis('off')

    #axes[plot_idx + 6].imshow(recon_SARTtransformer, cmap='gray')
    #axes[plot_idx + 6].set_title(f"SART Transformer Recon {idx+1}")
    #axes[plot_idx + 6].axis('off')

    #axes[plot_idx + 7].imshow(recon_SARTunet, cmap='gray')
    #axes[plot_idx + 7].set_title(f"SART Unet Recon {idx+1}")
    #axes[plot_idx + 7].axis('off')

    # Save the figure
    plt.savefig(f'/mmfs1/gscratch/uwb/CT_code/psnrssim/combined_image_{idx+1}.png', bbox_inches='tight')

    plt.close(fig)

# Calculate mean and standard deviation for PSNR values
mean_psnr_sv60 = np.mean(psnr_values_sv60)
std_psnr_sv60 = np.std(psnr_values_sv60)
mean_psnr_dncnn = np.mean(psnr_values_dncnn)
std_psnr_dncnn = np.std(psnr_values_dncnn)
mean_psnr_transformer = np.mean(psnr_values_transformer)
std_psnr_transformer = np.std(psnr_values_transformer)
mean_psnr_unet = np.mean(psnr_values_unet)
std_psnr_unet = np.std(psnr_values_unet)
#mean_psnr_SARTdncnn = np.mean(psnr_values_SARTdncnn)
#std_psnr_SARTdncnn = np.std(psnr_values_SARTdncnn)
#mean_psnr_SARTtransformer = np.mean(psnr_values_SARTtransformer)
#std_psnr_SARTtransformer = np.std(psnr_values_SARTtransformer)
#mean_psnr_SARTunet = np.mean(psnr_values_SARTunet)
#std_psnr_SARTunet = np.std(psnr_values_SARTunet)

# Calculate mean and standard deviation for SSIM values
mean_ssim_sv60 = np.mean(ssim_values_sv60)
std_ssim_sv60 = np.std(ssim_values_sv60)
mean_ssim_dncnn = np.mean(ssim_values_dncnn)
std_ssim_dncnn = np.std(ssim_values_dncnn)
mean_ssim_transformer = np.mean(ssim_values_transformer)
std_ssim_transformer = np.std(ssim_values_transformer)
mean_ssim_unet = np.mean(ssim_values_unet)
std_ssim_unet = np.std(ssim_values_unet)
# Print the calculated mean and std for PSNR and SSIM
print(f"Mean PSNR (SV60): {mean_psnr_sv60}, Std PSNR (SV60): {std_psnr_sv60}")
print(f"Mean PSNR (DnCNN): {mean_psnr_dncnn}, Std PSNR (DnCNN): {std_psnr_dncnn}")
print(f"Mean PSNR (Transformer): {mean_psnr_transformer}, Std PSNR (Transformer): {std_psnr_transformer}")
print(f"Mean PSNR (Unet): {mean_psnr_unet}, Std PSNR (Unet): {std_psnr_unet}")

print(f"Mean SSIM (SV60): {mean_ssim_sv60}, Std SSIM (SV60): {std_ssim_sv60}")
print(f"Mean SSIM (DnCNN): {mean_ssim_dncnn}, Std SSIM (DnCNN): {std_ssim_dncnn}")
print(f"Mean SSIM (Transformer): {mean_ssim_transformer}, Std SSIM (Transformer): {std_ssim_transformer}")
print(f"Mean SSIM (Unet): {mean_ssim_unet}, Std SSIM (Unet): {std_ssim_unet}")
