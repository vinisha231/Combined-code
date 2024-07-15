import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Subtract
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import struct
  
# Function to define the DnCNN model
def DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True):
    input_layer = Input(shape=(None, None, image_channels), name='input')
    x = layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(input_layer)

    for i in range(depth - 2):
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        if use_bnorm:
            x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

    output_layer = layers.Conv2D(filters=image_channels, kernel_size=3, padding='same')(x)
    output_layer = Subtract()([input_layer, output_layer])
    model = models.Model(inputs=input_layer, outputs=output_layer, name='DnCNN')
    return model

# Function to load .flt files
def load_flt_file(file_path, shape):
    with open(file_path, 'rb') as f:
        data = f.read()
    image = np.array(struct.unpack('f' * (len(data) // 4), data)).reshape(shape)
    return image

# Function to load and resize real image data
def load_real_data(noisy_dir, normal_dir, shape, limit=100):
    noisy_paths = sorted(glob(os.path.join(noisy_dir, '*.flt')))[:limit]
    normal_paths = sorted(glob(os.path.join(normal_dir, '*.flt')))[:limit]
    noisy_images = []
    normal_images = []
    
    for noisy_path, normal_path in zip(noisy_paths, normal_paths):
        noisy_image = load_flt_file(noisy_path, shape)
        normal_image = load_flt_file(normal_path, shape)
        noisy_image = np.expand_dims(noisy_image, axis=-1)  # Add channel dimension
        normal_image = np.expand_dims(normal_image, axis=-1)  # Add channel dimension
        noisy_image = tf.image.resize(noisy_image, shape)
        normal_image = tf.image.resize(normal_image, shape)
        noisy_images.append(noisy_image)
        normal_images.append(normal_image)
    
    noisy_images = np.array(noisy_images)
    normal_images = np.array(normal_images)
    return noisy_images, normal_images

# Load and preprocess a grayscale image from .flt or .png file
def preprocess_image(image_path, shape):
    if image_path.endswith('.flt'):
        image = load_flt_file(image_path, shape)
    else:
	raise ValueError(f"Unsupported file format: {image_path}")
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = tf.image.resize(image, shape)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to save .flt files
def save_flt_file(file_path, data):
    with open(file_path, 'wb') as f:
        f.write(struct.pack('f' * data.size, *data.flatten()))

# Function to save .png files
def save_png_file(file_path, data):
    tf.keras.preprocessing.image.save_img(file_path, data, scale=False)

# Function to display images
def display_images(original, noisy, denoised):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original[0, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy[0, :, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised[0, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.show()

# Define your image directory and size
noisy_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views'
normal_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views'
output_dir = '/mmfs1/gscratch/uwb/vdhaya/output'
os.makedirs(output_dir, exist_ok=True)
image_shape = (512, 512)  # Define the shape of the .flt images

# Load real data
X_train, y_train = load_real_data(noisy_dir, normal_dir, image_shape, limit=100)

# Check the shapes of the loaded data
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

# Create the model for grayscale images
model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=32, verbose=1)
# Save the model architecture and weights
model.save('dncnn_model.h5')
model.save_weights('dncnn_model.weights.h5')

# Print the model summary
model.summary()

# Get list of image paths (first 100 .flt files)
image_paths = sorted(glob(os.path.join(noisy_dir, '*.flt')))[:100]

# Process each image in the directory
for i, image_path in enumerate(image_paths):
    print(f'Processing image: {image_path}')

    # Preprocess the test image
    image = preprocess_image(image_path, image_shape)

    # Denoise the grayscale image using the DnCNN model
    predicted_noise = model.predict(image)
    denoised_image = image - predicted_noise

    # Save the denoised image to the specified folder
    output_image_path = os.path.join(output_dir, f'denoised_image_{i + 1}.flt')
    save_flt_file(output_image_path, denoised_image[0, :, :, 0])

    print(f"Denoised image saved to: {output_image_path}")

    # Optionally display the images
    display_images(image, image, denoised_image)
