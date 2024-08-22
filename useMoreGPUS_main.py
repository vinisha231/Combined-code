import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
#
#from tensorflow.keras.callbacks import TensorBoard

#tensorboard_callback = TensorBoard(log_dir='./logs')
#
from Combined_Loss import combined_loss
from unet_noDropout_model_library import unet_model

# Import necessary functions and classes
from Transformer import build_transformer_model, TransformerBlock
from dncnn2 import DnCNN
from shard3 import (
                    load_original_recon_dataset,
                    xy_dataset_from_xi_yi_datasets,
                    fltbatch_plt, fltbatch_flt)
# __GETTING VARIABLES__-----------------------------------
X_TRAIN_PATTERN = '/gscratch/uwb/CT_images/RECONS2024/60views/*.flt'  #glob pattern train input data
Y_TRAIN_PATTERN = '/gscratch/uwb/CT_images/RECONS2024/900views/*.flt' #glob pattern train target output data
X_TEST_PATTERN = '/gscratch/uwb/CT_images/recons2024/60views/*.flt'   #glob pattern validation input data
Y_TEST_PATTERN = '/gscratch/uwb/CT_images/recons2024/900views/*.flt'  #glob pattern validation output data
IMG_DIR = 'unet_nD_data_python3.12.4/png_dir'                         #directory for image files
FLT_DIR = 'unet_nD_data_python3.12.4/flt_dir'                         #directory for float32 raw data files
MODEL_FILE = 'model_unet_nD_250epoch_python3.12.4.keras'              #filename to store trained model to

#tensorflow-type variables to set initially

data_size = tf.constant(value=3629, dtype=tf.float64)       #number of input tensors
test_data_start = data_size                                 #to prefix that number to test tensor output file names
replica_batch_size = tf.constant(value=32, dtype=tf.float64)#batch size on each gpu
epochs = tf.constant(value=250, dtype=tf.int64)             #epochs to train for
validation_split = tf.constant(value=0.1, dtype=tf.float64) #percent as a decimal of training data to use for validation

# ------------------- SPECIFY ALL NON-INTERACTIVE USER INPUT ABOVE THIS LINE ------------------------
# __INPUT PROCESSING__--------------------------------------
with tf.device('CPU:0'):
    #see how much data goes where with the first CPU

    logical_devices = tf.config.list_logical_devices('GPU')#will return same as physical devices 'GPU'
    gpu_count = tf.cast(len(logical_devices), dtype=tf.float64)
    global_batch_size = replica_batch_size * gpu_count#get batch size of total

    train_data_size = tf.cast( ###get size in global_batch_size sized portions
	data_size - tf.math.floormod(data_size, global_batch_size),
	dtype=tf.float64)
	#get number of batches possible

    buffer_size = tf.cast(global_batch_size*gpu_count, ###overbatch buffer
			  dtype=tf.int64)
			  #global_batch_size * gpu_count,
			  #dtype=tf.int64)
	#^^^for valid int64 type in configure_for_performance call
    global_batches_per_epoch = train_data_size / global_batch_size
	#can be a float value, but should not

	#get batched length
    validation_slice = tf.cast( ### lower bound validation slice size
	tf.math.floor(global_batches_per_epoch * validation_split),
	dtype=tf.int64)
	#get validation batches
    epoch_slice = (
   	tf.cast(tf.math.floor(global_batches_per_epoch),
		dtype=tf.int64) - (validation_slice))

    #ensure the validation slice is at least one global batch
    if (validation_slice == 0):
        validation_slice += 1
        epoch_slice -= 1

    #for valid type in .batch() call
    global_batch_size = tf.cast(global_batch_size, tf.int64)

#easy reference record keeping printout
tf.print("\n\ndata_size: ",data_size,
	 "\nreplica_batch_size: ",replica_batch_size,
	 "\ngpu_count: ",gpu_count,
	 "\nglobal_batch_size",global_batch_size,
	 "\nglobal_batches_per_epoch: ",global_batches_per_epoch,
	 "\ntrain_data_size",train_data_size,
	 "\nbuffer_size: ",buffer_size,
	 "\nepoch_slice: ",epoch_slice,
	 "\nvalidation_slice: ",validation_slice,"\n\n")

# __INTERNAL FUNCTIONS__--------------------------------

#calls the build method of one of the three models
def pick_model(choice):
    if choice == '1':
        model = build_transformer_model(num_patches=4096, projection_dim=64, num_heads=3, transformer_layers=6, num_classes=4)
    elif choice == '2':
        model = unet_model()
    else:
        model = DnCNN(depth=17, filters=64, image_channels=1, use_bn=True)
    
    return model

#train the selected model using multiple gpus
def train_selected_model(number, flt_xy_dataset,
        epoch_slice,validation_slice,
	buffer_size, global_batch_size, epochs):

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = pick_model(number)
        model.compile(optimizer='AdamW', 
		  loss=combined_loss, metrics=['mse'])

    if model is not None:
        # cache, shuffle, and batch data
        xy_PMD = flt_xy_dataset.cache().shuffle(buffer_size=buffer_size)
        xy_PMD = xy_PMD.batch(batch_size=global_batch_size)

        train_xy_PMD = xy_PMD.skip(validation_slice)#take shuffled non-validation batches
        val_xy_PMD = xy_PMD.take(validation_slice)#take what isn't in train_xy_PMD



        #distribute global batches into local batches on each gpu
        dist_train_xy_PMD = (
        strategy.experimental_distribute_dataset(train_xy_PMD,
		tf.distribute.InputOptions(
			experimental_fetch_to_device=True, #send data to gpu
			experimental_replication_mode=     #for each worker
		tf.distribute.InputReplicationMode.PER_WORKER,
			experimental_per_replica_buffer_size=4))) #give each worker a batch buffer

        dist_val_xy_PMD = (
        strategy.experimental_distribute_dataset(val_xy_PMD,
		tf.distribute.InputOptions(
			experimental_fetch_to_device=True,
			experimental_replication_mode=
		tf.distribute.InputReplicationMode.PER_WORKER,
			experimental_per_replica_buffer_size=4)))
        

        #fit model
        model.fit(
	    x=dist_train_xy_PMD, 
	    epochs=epochs,validation_data=dist_val_xy_PMD)

        #save model
        model_string = MODEL_FILE
        model.save(model_string)
        
        #save some test png on the evaluation set
        for x_batch, y_batch in val_xy_PMD.take(1):
            p_batch = model.predict_on_batch(x_batch)
            fltbatch_plt(x_batch[0:3], 3, 0, 'x', IMG_DIR)
            fltbatch_plt(p_batch[0:3], 3, 0, 'p', IMG_DIR)
            fltbatch_plt(y_batch[0:3], 3, 0, 'y', IMG_DIR)
    else:
        print("Model loading failed. Training aborted.")

#prints .evaluate() on a model
def test_model(model, flt_xy_dataset): #selecting the outputs for these actions for async run
    if model is not None:
        loss, accuracy = model.evaluate(flt_xy_dataset)
        print(f"Test loss: {loss}\n Test accuracy: {accuracy}")
    else:
        print("Model is not loaded or invalid.")

# User choice for training or testing
test_or_train = input("Would you like to (1) test or (2) train?\nEnter your choice as '1' or '2':") 


if test_or_train == '2':
    # Model Selection
    number = input("Select (1) Transformer (2) Unet (3) DnCNN:")
    #
    xi_dataset = load_original_recon_dataset(X_TRAIN_PATTERN)#gather train input into dataset
    yi_dataset = load_original_recon_dataset(Y_TRAIN_PATTERN)#^^     train targets into dataset
    flt_xy_dataset = xy_dataset_from_xi_yi_datasets(         #consolidate both into (x,y) dataset
	xi_dataset, yi_dataset)#get training data, then train on it

    train_selected_model(number, flt_xy_dataset,
        epoch_slice, validation_slice, buffer_size, 
	global_batch_size, epochs)

    print(f"Saved sample of 3 original, reconstructed, and target (x,p,y) images to {IMG_DIR},\nas well as .flt files to {FLT_DIR}")
    
elif test_or_train == '1':
    # Model Selection
    choice = input("Enter valid model file <file_name>.keras: " )
    model = load_model(choice,#irrelevant custom_objects should not affect loading
	custom_objects={'combined_loss':combined_loss,'TransformerBlock':TransformerBlock})

    #get an (x,y) dataset like before
    xi_dataset = load_original_recon_dataset(X_TEST_PATTERN)
    yi_dataset = load_original_recon_dataset(Y_TEST_PATTERN)
    flt_xy_dataset = xy_dataset_from_xi_yi_datasets(
	xi_dataset, yi_dataset) 
    # Test the model
    test_model(model, flt_xy_dataset)##add parameters for batch size etcetera
   
    #process the dataset given the loaded model; tested for Transformer and Unet 
    start = test_data_start
    for x_batch, y_batch in flt_xy_dataset:
        p_batch = model.predict_on_batch(x_batch)
        size = (p_batch.shape)[0]
        fltbatch_plt(p_batch, size, start, 'p', IMG_DIR)
        fltbatch_flt(p_batch, size, start, 'p', FLT_DIR)
        start += size

    print(f"Saved original, reconstructed, and target (x,p,y) images to {IMG_DIR},\nas well as .flt files to {FLT_DIR}")
