import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Subtract
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from PIL import Image

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
  
  
