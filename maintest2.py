import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Import necessary functions and classes
import Transformer 
import Unet 
import dncnn 

# BEGIN DATA IMPLEMENTATION

def load_flt_file(file_path, shape=(512, 512), add_channel=True):
    """Load and reshape .flt files."""
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32)
        img = data.reshape(shape)
        if add_channel:
            img = img[:, :, np.newaxis]
    return img

def load_images_from_directory(directory, target_size=(512, 512)):
    """Load and normalize images from a directory."""
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".flt"):
            file_path = os.path.join(directory, filename)
            img = load_flt_file(file_path, shape=target_size)
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
    return np.array(images)

# Directories
clean_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views'
dirty_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views'

clean_images = load_images_from_directory(clean_dir)
dirty_images = load_images_from_directory(dirty_dir)

X_train, X_test, y_train, y_test = train_test_split(dirty_images, clean_images, test_size=0.2, random_state=42)
# END DATA IMPLEMENTATION

def save_as_flt(data, file_path):
    """Save data as a .flt file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        file.write(data.astype(np.float32).tobytes())

def load_model_by_choice(choice):
    """Load the model based on user choice."""
    models = {
	1: Transformer.load_model('transformer_model.keras'),
        2: Unet.load_model('unet_model.h5'),
        3: dncnn.load_model('dncnn_model.weights.h5')
    }
    return models.get(choice, None)

# Model Selection
print("Enter 1 for Transformers")
print("Enter 2 for Unet")
print("Enter 3 for DnCNN")
number = int(input("Enter preference:"))

model = load_model_by_choice(number)
if model is None:
    raise ValueError("Invalid model choice. Please enter 1, 2, or 3.")

# Predict and Save
predicted_images = model.predict(X_test)

output_dir_original = '/mmfs1/gscratch/uwb/vdhaya/output/original_images'
output_dir_reconstructed = f'/mmfs1/gscratch/uwb/vdhaya/output/reconstructed_images{number}'

for i in range(len(X_test)):
    original_img = X_test[i] * 255  # Scale back if necessary
    reconstructed_img = predicted_images[i] * 255
    save_as_flt(original_img, os.path.join(output_dir_original, f'original_{i}.flt'))
    save_as_flt(reconstructed_img, os.path.join(output_dir_reconstructed, f'reconstructed_{i}.flt'))

print(f"Saved original images to {output_dir_original}")
print(f"Saved reconstructed images to {output_dir_reconstructed}")
