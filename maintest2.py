import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Subtract
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from PIL import Image
import Transformer 
import Unet 
import dncnn
import shard 
import testshard 
# Import necessary functions and classes

#BEGIN DATA IMPLEMENTATION
def load_flt_file(file_path, shape=(512, 512), add_channel=True): #Defines the flt load function with image size
    with open(file_path, 'rb') as file: #Opens the directory containing the flts
        data = np.fromfile(file, dtype=np.float32) #Stores the files as floats
        img = data.reshape(shape) #Reshapes the data to be used
        if add_channel: #Add channel set true so we do
            img = img[:, :, np.newaxis] #Adds a channel of depth to flt for use
    return img

# Load and preprocess images
def load_images_from_directory(directory, target_size=(512, 512)): #Function to load the images
    images = []                                                             #Creates empty array to store images
    for filename in os.listdir(directory): #For each file in the directory do
        if filename.endswith(".flt"): #If the file is an flt
            file_path = os.path.join(directory, filename) #Stores the name of the directory and file path to load
            img = load_flt_file(file_path, shape=target_size) #Loads the flts using previous function and stores as 512x512
            img = img / 255.0 #Normalizes to values between [0,1]
            images.append(img) #Adds the grabbed image to back of array
    return np.array(images)

# Directories
clean_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views' #Sets the data directories
dirty_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views'

clean_images = load_images_from_directory(clean_dir) #Performs the load image function on data directories
dirty_images = load_images_from_directory(dirty_dir)

X_train, X_test, y_train, y_test = train_test_split(dirty_images, clean_images, test_size=0.2, random_state=42)
#END DATA IMPLEMENTATION
print("Enter 1 for Transformers")
print("Enter 2 for Unet")
print("Enter 3 for DnCNN")
number = int(input("Enter preference:"))
#BEGIN IMAGE SAVING
model1 = Transformer.load_model('transformer_model.keras')
model2 = Unet.load_model('unet_model.h5')
model3 = dncnn.load_model('dncnn_model.weights.h5')
# Predict on a test set
predicted_images1 = model1.predict(X_test)
predicted_images2 = model2.predict(X_test)
predicted_images3 = model3.predict(X_test)
# Save original and reconstructed images for comparison
output_dir_original = '/mmfs1/home/bkphill2/output/original_images' #Saves the originals and reconstructed images to directory (Commented in other git code)
output_dir_reconstructed1 = '/mmfs1/home/bkphill2/output/reconstructed_images1'
output_dir_reconstructed2 = '/mmfs1/home/bkphill2/output/reconstructed_images2'
output_dir_reconstructed3 = '/mmfs1/home/bkphill2/output/reconstructed_images3'

def save_as_flt(data, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Open the file in write-binary (wb) mode
    with open(file_path, 'wb') as file:
    # Write the data as floating point data
        file.write(data.astype(np.float32).tobytes())

for i in range(len(X_test)):
    original_img = X_test[i] * 255  # Scale if necessary, but keep as float if saving as .flt
    reconstructed_img1 = predicted_images1[i] * 255  # Scale if necessary, but keep as float
    reconstructed_img2 = predicted_images2[i] * 255	
    reconstructed_img3 = predicted_images3[i] * 255
    save_as_flt(original_img, os.path.join(output_dir_original, f'original_{i}.flt'))
    save_as_flt(reconstructed_img1, os.path.join(output_dir_reconstructed1, f'reconstructed1_{i}.flt'))
    save_as_flt(reconstructed_img2, os.path.join(output_dir_reconstructed2, f'reconstructed2_{i}.flt'))
    save_as_flt(reconstructed_img3, os.path.join(output_dir_reconstructed3, f'reconstructed3_{i}.flt'))

print(f"Saved original images to {output_dir_original}")
print(f"Saved reconstructed images to {output_dir_reconstructed}")
#END IMAGE SAVING
