import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
#
#from tensorflow.keras.callbacks import TensorBoard

#tensorboard_callback = TensorBoard(log_dir='./logs')
#
from custom_loss_functions import combined_loss
from unet_model_library import unet_model

# Import necessary functions and classes
from Transformer import build_transformer_model
from dncnn2 import DnCNN
from shard3 import (
                    load_original_recon_dataset,
                    xy_dataset_from_xi_yi_datasets,
                    configure_for_performance,
                    fltbatch_plt, fltbatch_flt)

X_TRAIN_PATTERN = '/gscratch/uwb/CT_images/RECONS2024/60views/*.flt'
Y_TRAIN_PATTERN = '/gscratch/uwb/CT_images/RECONS2024/900views/*.flt'
X_TEST_PATTERN = '/gscratch/uwb/CT_images/recons2024/60views/*.flt'
Y_TEST_PATTERN = '/gscratch/uwb/CT_images/recons2024/900views/*.flt'
TRAIN_DATA_SIZE = 3629
TEST_DATA_START = 3629
BUFFER_SIZE = 1800
BATCH_SIZE = 80
REPLICA_BUFFER = 10
EPOCHS = 25
STEPS_PER_EPOCH = 45
VAL_DATA_RATIO = 0.2
FLT_DIR = 'flt_dir'#careful here with testing on the whole test set and dumping images into cwd
IMAGE_DIR = 'png_dir'

# BEGIN DATA IMPLEMENTATION

def load_model_by_choice(choice): ##seems redundant, change to a load call
    try:
        model = load_model(choice,custom_objects={'combined_loss':combined_loss})
        return model
    except IOError:
        print(f"Model {choice} not found.")
        return None

def pick_model(choice): #this is applicable, save this as an option to main
    if choice == '1':
        model = build_transformer_model(num_patches=4096, projection_dim=64, num_heads=3, transformer_layers=6, num_classes=4)
    elif choice == '2':
        model = unet_model()
    else:
        model = DnCNN(depth=17, filters=64, image_channels=1, use_bn=True)
    
    return model

def train_selected_model(number, flt_xy_dataset, dataSize, valDataRatio,
		 bufferSize, batchSize, epochs, steps_per_epoch):#add options for batch_size, epochs, etc
    
    choice = ['Transformer', 'Unet', 'DnCNN']

    strategy = tf.distribute.MirroredStrategy()
    #with tf.device('/GPU:0'):
    with strategy.scope():
        model = pick_model(number)
    model.compile(optimizer='AdamW', loss=combined_loss, metrics=['accuracy'])

    if model is not None:
        xy_dataset_PMD = configure_for_performance(flt_xy_dataset, bufferSize, batchSize)
        #print("SUCCESSFUL_CACHE")
        #^^^caches, shuffles, and batches dataset; be sure to watch bufferSize, 
                                                    #as this is the number of images stored in CPU memory 
        strategy.experimental_distribute_dataset(xy_dataset_PMD, 
		tf.distribute.InputOptions(experimental_per_replica_buffer_size=REPLICA_BUFFER))
        model.fit(xy_dataset_PMD, epochs=epochs, steps_per_epoch=steps_per_epoch)#, callbacks=[tensorboard_callback])
        model_index = (int(number) - 1) % 3
        model_string = f'NEW_{choice[model_index]}.keras'
        model.save(model_string)
        
        for x_batch, y_batch in xy_dataset_PMD.take(1):
            p_batch = model.predict_on_batch(x_batch)
            fltbatch_plt(x_batch[0:3], 3, 0, 'x', IMAGE_DIR)
            #fltbatch_flt(x_batch[0:3], 3, 0, 'x', FLT_DIR)
            fltbatch_plt(p_batch[0:3], 3, 0, 'p', IMAGE_DIR)
            #fltbatch_flt(p_batch[0:3], 3, 0, 'p', FLT_DIR)
            fltbatch_plt(y_batch[0:3], 3, 0, 'y', IMAGE_DIR)
            #fltbatch_flt(y_batch[0:3], 3, 0, 'y', FLT_DIR)
    else:
        print("Model loading failed. Training aborted.")

def test_model(model, flt_xy_dataset): #selecting the outputs for these actions for async run
    if model is not None:
        loss, accuracy = model.evaluate(flt_xy_dataset)
        print(f"Test loss: {loss}\n Test accuracy: {accuracy}")
    else:
        print("Model is not loaded or invalid.")

# User choice for training or testing   #I think this is ridiculous, make it a template that is editable
test_or_train = input("Would you like to (1) test or (2) train?\nEnter your choice as '1' or '2':") 


if test_or_train == '2':
    # Model Selection
    number = input("Select (1) Transformer (2) Unet (3) DnCNN:")
    #
    xi_dataset = load_original_recon_dataset(X_TRAIN_PATTERN)
    yi_dataset = load_original_recon_dataset(Y_TRAIN_PATTERN)
    flt_xy_dataset = xy_dataset_from_xi_yi_datasets(xi_dataset, yi_dataset)#get training data, then train on it
    train_selected_model(number, flt_xy_dataset, 
	TRAIN_DATA_SIZE, VAL_DATA_RATIO, BUFFER_SIZE, BATCH_SIZE, EPOCHS, STEPS_PER_EPOCH)

    print(f"Saved sample of 3 original, reconstructed, and target (x,p,y) images to {IMAGE_DIR},\nas well as .flt files to {FLT_DIR}")
    
elif test_or_train == '1':
    # Model Selection
    number = input("Enter valid model file <file_name>.keras.h5: " )
    model = load_model_by_choice(number)

    if model is None:
        raise ValueError("Only '1', '2', '3', or a valid *.h5 file are valid inputs")
    xi_dataset = load_original_recon_dataset(X_TEST_PATTERN)
    yi_dataset = load_original_recon_dataset(Y_TEST_PATTERN)
    flt_xy_dataset = xy_dataset_from_xi_yi_datasets(xi_dataset, yi_dataset) 
    # Test the model
    test_model(model, flt_xy_dataset)##add parameters for batch size etcetera
    
    # Predict and Save
    predicted_images = model.predict(dirty_images_test)###prediction should be slight, just for verifying functionality
    
    for x_batch, y_batch in flt_xy_dataset:
        p_batch = model.predict_on_batch(x_batch)
        size = (p_batch.shape)[0]
        start = TEST_DATA_START
        fltbatch_plt(x_batch, size, start, 'x', IMAGE_DIR)
        #fltbatch_flt(x_batch, size, start, 'x', FLT_DIR)
        fltbatch_plt(p_batch, size, start, 'p', IMAGE_DIR)
        #fltbatch_flt(p_batch, size, start, 'p', FLT_DIR)
        fltbatch_plt(y_batch, size, start, 'y', IMAGE_DIR)
        #fltbatch_flt(y_batch, size, start, 'y', FLT_DIR)

    print(f"Saved original, reconstructed, and target (x,p,y) images to {IMAGE_DIR},\nas well as .flt files to {FLT_DIR}")
