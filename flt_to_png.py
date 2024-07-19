import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def flt_to_png(flt_file_path, png_file_path):
    width, height = 512, 512

    num_elements = width * height

    data = np.fromfile(flt_file_path, dtype=np.float32)

    if data.size != num_elements:
        raise ValueError(f"Expected {num_elements} elements, but got {data.size}.")

    data = data.reshape((height, width))

    max_data = np.max(data)
    min_data = np.min(data)

    if max_data-min_data == 0:
        print("no variation")

    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    image_data = normalized_data.astype(np.uint8)

    print(max_data)
    print(min_data)

    img = Image.fromarray(image_data, 'L')
    img.save(png_file_path)
flt_file_path = '/mmfs1/gscratch/uwb/bkphill2/output/original_images/original_12.flt'
png_file_path = '/mmfs1/gscratch/uwb/bkphill2/Combined-code/png_output/outputog_12.png'
flt_to_png(flt_file_path, png_file_path)
