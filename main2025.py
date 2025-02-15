import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import pdb
from Transformer import build_transformer_model, TransformerBlock
from Unet import unet_model
from dncnn2 import DnCNN
# from Combined_Loss import combined_loss

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred, alpha=0.35, beta=0.65):
    return alpha * ssim_loss(y_true, y_pred) + beta * mse_loss(y_true, y_pred)

def load_flt_file(file_path, shape=(512, 512), add_channel=True, normalize=False):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32)
        img = data.reshape(shape)

        if normalize:
            min_val = img.min()
            max_val = img.max()
            if max_val - min_val > 0:
                img = (img - min_val) / (max_val - min_val)
            else:
                img = np.zeros(shape)

        if add_channel:
            img = img[:, :, np.newaxis]
    return img

def load_images_from_directory(directory, target_size=(512, 512, 1)):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".flt"):
            file_path = os.path.join(directory, filename)
            print('Loading ' + file_path + '\n')
            img = load_flt_file(file_path, shape=target_size)
            images.append(img)
    return np.array(images)

clean_dir_train = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views'
dirty_dir_train = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views'
clean_dir_test = '/mmfs1/gscratch/uwb/CT_images/recons2024/900views'
dirty_dir_test = '/mmfs1/gscratch/uwb/CT_images/recons2024/60views'

def save_as_flt(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        file.write(data.astype(np.float32).tobytes())

def load_model_by_choice(choice):
    model = load_model(f'model_{choice}.keras', custom_objects={'combined_loss': 'mse'})
    return model

def pick_model(choice):
    if choice == 1:
        model = build_transformer_model(num_patches=4096, projection_dim=64, num_heads=3, transformer_layers=6, num_classes=4)
    elif choice == 2:
        model = unet_model()
    elif choice == 3:
        model = DnCNN(depth=17, filters=64, image_channels=1, use_bn=True)
    else:
        print("Invalid choice")
        model = None
    model.summary()
 #   pdb.set_trace()
    return model

def train_selected_model(choice, X_train, y_train):
    model = pick_model(choice)
    if model is not None:
        print(model.output_shape)
        print("Model output shape:", model.output_shape)
        assert model.output_shape == (None, 512, 512, 1), "Mismatch in model output shape."
 #       if choice == 3:
 #           model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
 #           model.fit(X_train, y_train, epochs=250, batch_size=8, verbose=1)
 #           model.save('dncnn_model.h5')
 #           model.save_weights('dncnn_model.weights.h5')

 #       else:
        model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=2)
        model.save(f'model_{choice}.keras')
    else:
        print("Model loading failed. Training aborted.")

test_or_train = input("Would you like to test or train? ")
training = True if test_or_train == "train" else False

print("Enter 1 for Transformers")
print("Enter 2 for Unet")
print("Enter 3 for DnCNN")
number = int(input("Enter preference: "))

if training:
    clean_images_train = load_images_from_directory(clean_dir_train)
    dirty_images_train = load_images_from_directory(dirty_dir_train)
    np.savez('train_data',clean_images_train=clean_images_train,dirty_images_train=dirty_images_train)
    #pdb.set_trace()
    train_data = np.load('train_data.npz')
    X_train, y_train = train_data['dirty_images_train'], train_data['clean_images_train']
    train_selected_model(number, train_data['dirty_images_train'], train_data['clean_images_train'])

X_test = load_images_from_directory(clean_dir_test)
y_test = load_images_from_directory(dirty_dir_test)
np.savez('test_data', X_test=X_test, y_test=y_test)
X_test, y_test = X_test, y_test
#pdb.set_trace()
model = load_model_by_choice(number)
if model is None:
    raise ValueError("Invalid model choice. Please enter 1, 2, or 3.")

predicted_images = model.predict(y_test)
#predicted_images = model.predict(X_test) 

output_dir_original = 'output/original_images'
output_dir_reconstructed = f'output{number}/reconstructed_images{number}'
for i in range(len(X_test)):
    original_img = y_test[i]
    reconstructed_img = predicted_images[i]
    save_as_flt(original_img, os.path.join(output_dir_original, f'original_{i:04}.flt'))
    save_as_flt(reconstructed_img, os.path.join(output_dir_reconstructed, f'reconstructed_{i:04}.flt'))

print(f"Saved original images to {output_dir_original}")
print(f"Saved reconstructed images to {output_dir_reconstructed}")
