import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Subtract
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import pdb

# Function to define the DnCNN model
def DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True):
    input_layer = Input(shape=(None, None, image_channels), name='input')
    x = layers.Conv2D(filters=filters, kernel_size=3, padding='same',    activation='relu')(input_layer)

    for i in range(depth - 2):
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        if use_bnorm:
            x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

    output_layer = layers.Conv2D(filters=image_channels, kernel_size=3, padding='same')(x)
    output_layer = Subtract()([input_layer, output_layer])
    model = models.Model(inputs=input_layer, outputs=output_layer, name='DnCNN')
    return model

# Function to load real image data
def load_real_data(image_dir):
    image_paths = glob(os.path.join(image_dir, '*.png'))
    images = []
    for path in image_paths:
        image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.0
        images.append(image)
    images = np.array(images)
    noise = np.random.normal(loc=0.0, scale=0.1, size=images.shape).astype('float32')
    noisy_images = np.clip(images + noise, 0., 1.)
    return noisy_images, images

# Function to add noise to an image
def add_noise(image, noise_factor=0.1):
    noisy_image = image + noise_factor * np.random.normal(loc=0.0, scale=0.1, size=image.shape)
    noisy_image = np.clip(noisy_image, 0., 1.)
    return noisy_image

# Load and preprocess a grayscale image
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
    image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define your image directory and size
image_dir = '/mmfs1/gscratch/uwb/vdhaya/'
# image_size = 32
pdb.set_trace()
# Load real data
X_train, y_train = load_real_data(image_dir)

# Check the shapes of the loaded data
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

# Create the model for grayscale images
model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=1, verbose=1)

# Save the model architecture and weights
model.save('dncnn_model.h5')
model.save_weights('dncnn_model.weights.h5')

# Print the model summary
model.summary()

# Specify the correct path to your grayscale image file
image_path = '/mmfs1/gscratch/uwb/CT_images/train/images/00003299_img.png'

# Ensure the grayscale image file exists at the specified path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The specified image file does not exist: {image_path}")

# Preprocess and add noise to the test image
image = preprocess_image(image_path)
noisy_image = add_noise(image)

# Denoise the grayscale image using the DnCNN model
predicted_noise = model.predict(noisy_image)
denoised_image = Subtract()([noisy_image, predicted_noise])
# Display the images
def display_images(original, noisy, denoised):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original[0, :, :, 0], cmap='gray')  # Ensure to plot back in the [0, 1] range for grayscale images
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy[0, :, :, 0], cmap='gray')  # Ensure to plot back in the [0, 1] range for grayscale images
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised[0, :, :, 0], cmap='gray')  # Ensure to plot back in the [0, 1] range for grayscale images
    plt.axis('off')

    # Save the denoised image to the specified folder
    output_folder = '/mmfs1/gscratch/uwb/vdhaya/images'
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, 'predicted_image1.png')
    pdb.set_trace()
    tf.keras.preprocessing.image.save_img(output_image_path, 255.0 * .           predicted_noise[0], scale=False)
    print(f"Denoised image saved to: {output_image_path}")

    plt.show()
display_images(image, noisy_image, denoised_image)
