import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import pdb
# Import necessary functions and classes
from Transformer import build_transformer_model
from  Unet import unet_model
from dncc import DnCNN
from Combined_Loss import combined_loss

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
# Memory growth must be set before GPUs have been initialized
        print(e)

#No annoying messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (alternative method)

# BEGIN DATA IMPLEMENTATION

def ssim_loss(y_true, y_pred): #Defining a function of ssim loss image to image
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) #return the loss value

def mse_loss(y_true, y_pred): #Defining a mean squared error loss
    return tf.reduce_mean(tf.square(y_true - y_pred)) #Returning loss

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred, alpha = 0.35, beta = 0.65): #Define a mixed loss with proportions alpha and beta
    return alpha * ssim_loss(y_true, y_pred) + beta * mse_loss(y_true, y_pred) #Return the sum of the weighted losses =1

def load_flt_file(file_path, shape=(512, 512), add_channel=True, normalize=True):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32)
        img = data.reshape(shape)

     #   if normalize:
        # Normalize data to [0, 1] range
          #  min_val = img.min()
           # max_val = img.max()
            #if max_val - min_val > 0:  # Prevent division by zero
             #   img = (img - min_val) / (max_val - min_val)
           # else:
            #    img = np.zeros(shape)  # Or set to a default value within the target rang

        if add_channel:
            img = img[:, :, np.newaxis]
    return img

def load_images_from_directory(directory, target_size=(512, 512, 1)):
    """Load and normalize images from a directory."""
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".flt"):
            file_path = os.path.join(directory, filename)
            img = load_flt_file(file_path, shape=target_size)
            images.append(img)
    return np.array(images)

# Directories
clean_dir_train = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views'
dirty_dir_train = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views'

clean_dir_test = '/mmfs1/gscratch/uwb/CT_images/recons2024/900views'
dirty_dir_test = '/mmfs1/gscratch/uwb/CT_images/recons2024/60views'

# END DATA IMPLEMENTATION

def save_as_flt(data, file_path):
    """Save data as a .flt file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        file.write(data.astype(np.float32).tobytes())


def load_model_by_choice(choice):
    """Load the model based on user choice."""
    model = load_model(f'model_{choice}.keras', custom_objects={'combined_loss': combined_loss})
    return model

def pick_model(choice):
    if choice == 1:
        model = build_transformer_model(num_patches=4096, projection_dim=64, num_heads=3, transformer_layers=6, num_classes=4)
    elif choice == 2:
        model = unet_model()
    else:
        model = DnCNN(depth=14, filters=64, image_channels=1, use_bn=True)
    return model
def train_selected_model(choice, X_train, y_train):
    model = pick_model(choice)
    if model is not None:
        print(model.output_shape)
        assert model.output_shape == (None, 512, 512, 1), "Mismatch in model output shape."
        model.compile(optimizer='AdamW', loss=combined_loss, metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=8)
    else:
        print("Model loading failed. Training aborted.")
    model.save(f'model_{choice}.keras')

test_or_train = input("Would you like to test or train?")
if test_or_train == "train":
    training = True
elif test_or_train == "test":
    training = False

# Handle the case where all values are the same (e.g., an image of a single color)
# Model Selection
print("Enter 1 for Transformers")
print("Enter 2 for Unet")
print("Enter 3 for DnCNN")
number = int(input("Enter preference:"))

if training:
    clean_images_train = load_images_from_directory(clean_dir_train)
    dirty_images_train = load_images_from_directory(dirty_dir_train)
    X_train, y_train = dirty_images_train, clean_images_train
    train_selected_model(number, X_train, y_train)

clean_images_test = load_images_from_directory(clean_dir_test)
dirty_images_test = load_images_from_directory(dirty_dir_test)
X_test, y_test = dirty_images_test, clean_images_test

model = load_model_by_choice(number)
if model is None:
    raise ValueError("Invalid model choice. Please enter 1, 2, or 3.")

# Predict and Save
pdb.set_trace()
predicted_images = model.predict(dirty_images_test)

output_dir_original = '/mmfs1/gscratch/uwb/bkphill2/output/original_images'
output_dir_reconstructed = f'/mmfs1/gscratch/uwb/bkphill2/output/reconstructed_images{number}'

for i in range(len(clean_images_test)):
    original_img = dirty_images_test[i]
    reconstructed_img = predicted_images[i]
    save_as_flt(original_img, os.path.join(output_dir_original, f'original_{i:04}.flt'))
    save_as_flt(reconstructed_img, os.path.join(output_dir_reconstructed, f'reconstructed_{i:04}.flt'))

print(f"Saved original images to {output_dir_original}")
print(f"Saved reconstructed images to {output_dir_reconstructed}")

