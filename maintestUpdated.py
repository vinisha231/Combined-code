import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Import necessary functions and classes
from Transformer import build_transformer_model
from  Unet import unet_model
from dncc import DnCNN

#No annoying messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (alternative method)

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
            images.append(img)
    return np.array(images)

# Directories
clean_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views'
dirty_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views'

clean_dir_test = '/mmfs1/gscratch/uwb/CT_images/recons2024/900views'
dirty_dir_test = '/mmfs1/gscratch/uwb/CT_images/recons2024/60views'

clean_images = load_images_from_directory(clean_dir)
dirty_images = load_images_from_directory(dirty_dir)
clean_images_test = load_images_from_directory(clean_dir_test)
dirty_images_test = load_images_from_directory(dirty_dir_test)

X_train, X_test, y_train, y_test = train_test_split(dirty_images, clean_images, test_size=0.2, random_state=42)
# END DATA IMPLEMENTATION

def save_as_flt(data, file_path):
    """Save data as a .flt file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        file.write(data.astype(np.float16).tobytes())


def load_model_by_choice(choice):
    """Load the model based on user choice."""
    model = load_model(f'model_{choice}.keras')
    return model

def pick_model(choice):
    if choice == 1:
        model = build_transformer_model(num_patches=4096, projection_dim=64, num_heads=3, transformer_layers=6, num_classes=4)
    elif choice == 2:
        model = unet_model()
    else:
        model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)
    return model

def train_selected_model(choice, X_train, y_train):
    model = pick_model(choice)
    if model is not None:
        print(model.output_shape)
        assert model.output_shape == (None, 512, 512, 1), "Mismatch in model output shape."
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=8)
    else:
        print("Model loading failed. Training aborted.")
    model.save(f'model_{choice}.keras')

# Model Selection
print("Enter 1 for Transformers")
print("Enter 2 for Unet")
print("Enter 3 for DnCNN")
number = int(input("Enter preference:"))

train_selected_model(number, X_train, y_train)

model = load_model_by_choice(number)
if model is None:
    raise ValueError("Invalid model choice. Please enter 1, 2, or 3.")

# Predict and Save
predicted_images = model.predict(dirty_images_test)

output_dir_original = '/mmfs1/gscratch/uwb/bkphill2/output/original_images'
output_dir_reconstructed = f'/mmfs1/gscratch/uwb/bkphill2/output/reconstructed_images{number}'

for i in range(len(clean_images_test)):
    original_img = X_test[i]
    reconstructed_img = predicted_images[i]
    save_as_flt(original_img, os.path.join(output_dir_original, f'original_{i:04}.flt'))
    save_as_flt(reconstructed_img, os.path.join(output_dir_reconstructed, f'reconstructed_{i:04}.flt'))

print(f"Saved original images to {output_dir_original}")
print(f"Saved reconstructed images to {output_dir_reconstructed}")
