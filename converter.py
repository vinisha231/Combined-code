import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def read_flt_file(file_path, shape):
    with open(file_path, 'rb') as f:
        data = f.read()
        return np.array(struct.unpack('f' * (len(data) // 4), data)).reshape(shape)

def save_as_png(file_path, image_data):
    plt.imshow(image_data, cmap='gray')
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Define input and output directories and image shape
input_dir = '/mmfs1/gscratch/uwb/vdhaya/output'
output_dir = '/mmfs1/gscratch/uwb/vdhaya/output/png'
os.makedirs(output_dir, exist_ok=True)
image_shape = (512, 512)

# Process each .flt file in the input directory
flt_files = sorted(glob(os.path.join(input_dir, '*.flt')))[:100]
for i, file_path in enumerate(flt_files):
    image_data = read_flt_file(file_path, image_shape)
    output_file_path = os.path.join(output_dir, f'denoised_image_{i + 1}.png')
    save_as_png(output_file_path, image_data)
    print(f"Saved {output_file_path}")
