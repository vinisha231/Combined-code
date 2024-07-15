import tensorflow as tf
import os
import sys
import glob
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

#directory containing directories of tfrecord dataset directories

"""
get_basename(path): takes the last label, the file name, of a file name absolute path
	returns the file name
create_tfrecord_name(pathlike): appends the .tfrecord extension to a tf.string after isolating file name with get_basename(pathlike)
	returns the file name with extension .tfrecord
file_to_tensor(pathlike): reads a an original recon file into a tensor
	returns tf.float32 tensor
where_tensor_plt(where, tensor): plots a tf.float32 tensor using plt.imshow()
	creates subplot for tensor in open plt.figure with subplots to fill remaining
write_proto_to_tfrecord(proto, tfrecord): writes formatted tensor as 'tfrecord' named file
	writes proto of tensor to tfrecord file by name, does so assuming you are in home directory
tensor_to_serialized_example_proto(tensor): formats tensor for writing
	returns serialized example of tensor
tensor_label_to_serialized_example_proto(tensor, label): formats only tensor of tensor-label pairs
	same as ^^^, but returns label as well, maintaining data-label pairs form
parse_tfrecord(tfrecord): parses tfrecord of tensor back into tensor
	parses tensor from tfrecord
	returns the parsed tensor
save_original_files_to_tfrecord_recon_dataset(file_pattern):
	gets original files into tfrecords in a tfrecord directory
	returns absolute path to directory of saved dataset
load_tfrecord_recon_dataset(tfrecord_dir): loads tfrecords from tfrecord_dir into a dataset of tensors (without labels)
	loads tensors of tfrecords into dataset
	returns loaded dataset
load_original_recon_dataset(file_pattern): loads non-tfrecord float raw files into dataset of tf.float32 tensors
	loads original recon files into dataset
	returns loaded dataset
save_recon_dataset_as_tfrecords(tensor_label_dataset) saves tensor-label pair dataset as tfrecords in a tfrecord dir
	saves dataset to tfrecord directory
	returns directory absolute path
"""



HOME_DIRECTORY = '/home/NETID/bodhik/reudir/UWB_CT' #your home directory, to make all commands run smoothly
RELATIVE_ROOT_TFRECORDS_DIR = 'tfrecords_dir' #root to tfrecords_dir relative to the directory you are running the program in
FILE_PATTERN = '/data/CT_images/train/images/0000100[0-2]*.flt' #the file pattern. It expects 512,512,1 datapoints of type float32
TFRECORD_EXT = tf.constant('.tfrecord')#typical extension for data files

#file name extraction

def _get_basename(path):
	return os.path.basename(path.numpy().decode())
"""
	input: tf.string 'path' containing a path to a file
	output: the file name at the end of that path
"""

#encapsulator

@tf.function
def get_basename(path):
	return tf.py_function(_get_basename, [path], tf.string)
"""
	encapsulates the _get_basename function for use in a tensorflow function
"""

#tfrecord file name internal function

def create_tfrecord_name(pathlike):
	prename = get_basename(pathlike)
	name = tf.strings.join([prename, TFRECORD_EXT])
	return name
"""
	adds extension to file name
"""

#original files reading function

def file_to_tensor(pathlike):
	raw = tf.io.read_file(pathlike)
	flt = tf.io.decode_raw(raw, tf.float32)
	tensor = tf.reshape(flt, [512, 512, 1])
	return tensor
"""
	reads raw data to 512,512,1 RGB tf.float32 from a 'path'
	intended for use withfrom CT_images/train/images 0000????_img.flt
"""

#sequential plotting for tensors in a matplotlib.pyplot figure

def where_tensor_plt(where, tensor):
	plt.subplot(*where)
	plt.imshow(tensor, cmap='gray', vmin=0, vmax=1)
	plt.title(where[2])
	plt.axis("off")
	where[2] += 1
	return where
"""
	plots a 512,512,1 tf.float32 tensor as a heatmap at the position available in 'where', with where := (n,n,n) of integers
	would not work with parallelism
	when where=(a,b,c), it is printing in subplot c of a plt.figure(figsize=(a,b)), and  1<= c <= a*b
	it increments entry c
"""

#internal file writing

def _write_proto_to_tfrecord(proto, tfrecord):
	with tf.io.TFRecordWriter(tfrecord.numpy().decode('utf-8')) as writer:
		writer.write(proto.numpy())
	return proto, tfrecord
"""
	writes formatted tensor to a .tfrecord file of file name tfrecord
"""

#encapsulator

@tf.function
def write_proto_to_tfrecord(proto, tfrecord):
	return 	tf.py_function(func=_write_proto_to_tfrecord, inp=[proto, tfrecord], Tout=(tf.string, tf.string))

#internal formatting for tensor to write to .tfrecord file

def _tensor_to_serialized_example_proto(tensor):
	stensor = tf.io.serialize_tensor(tensor)

	feature = tf.train.Feature(bytes_list = tf.train.BytesList(value = [stensor.numpy()]))

	feature_proto = {'data': feature}

	example_proto = tf.train.Example(
		features = tf.train.Features(feature=feature_proto))

	serialized_example_proto = example_proto.SerializeToString()
	
	return serialized_example_proto

"""
	essentially performs multiple transformations to a standard data format
	many of these types are a nightmare to interact with
"""
#encapsulator

@tf.function
def tensor_to_serialized_example_proto(tensor):
	return tf.py_function(func=_tensor_to_serialized_example_proto, inp=[tensor], Tout=tf.string)

#encapsulator: for tuple dataset of tensor-label pairs

@tf.function
def tensor_label_to_serialized_example_proto(tensor, label):
	tensor_example_proto = ( 
	tf.py_function(func=_tensor_to_serialized_example_proto, inp=[tensor], Tout=tf.string))
	return tensor_example_proto, label


#reads tfrecord of tensor into tensor, is internal to reading a tensor dataset

#@tf.function >>> this was giving an aggregious error
@tf.autograph.experimental.do_not_convert
def parse_tfrecord(tfrecord):
	features_proto = {'data': tf.io.FixedLenFeature([], tf.string)}
	serialized_tensor = tf.io.parse_single_example(tfrecord, features_proto)
	tensor = tf.io.parse_tensor(serialized_tensor['data'], out_type=tf.float32)
	return tensor

"""
	defines the form of the data in the proto, extracts it, deserializes the tensor, and returns it
"""

#create a multi-tensor dataset from a file pattern; expects shape [512, 512, 1] images from files of raw float32 data
#dataset is stored in tfrecords
def save_original_files_to_tfrecord_recon_dataset(file_pattern):
	
	os.chdir(HOME_DIRECTORY)

	original_data_files = tf.data.Dataset.list_files(file_pattern)#file list dataset
	dataset = original_data_files.map(file_to_tensor, num_parallel_calls=tf.data.AUTOTUNE)#load tensor data

	os.makedirs(RELATIVE_ROOT_TFRECORDS_DIR, exist_ok=True)
	os.chdir(RELATIVE_ROOT_TFRECORDS_DIR)#change to root tfrecord dir
	PID = os.getpid()#get process id for unique directory name
	new_tfrecord_dir = f'{PID}tfrecord_dir'
	os.makedirs(new_tfrecord_dir, exist_ok=True)#create directory for tfrecord storage
	os.chdir(new_tfrecord_dir)#change to new tfrecord dir

	#shard it to tfrecord files, described element, x, wise

	serialized_example_proto_dataset = dataset.map(tensor_to_serialized_example_proto, 
						       num_parallel_calls=tf.data.AUTOTUNE)#create example of x
	data_files = original_data_files.map(create_tfrecord_name, 
					     num_parallel_calls=tf.data.AUTOTUNE)#parse name for new files
	sep_f = tf.data.Dataset.zip((serialized_example_proto_dataset, data_files))#pair x, labelx
	sep_f = sep_f.map(write_proto_to_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)#write x tfrecord

	for sf in sep_f:#checks whether files have been created; ensures lazy-executions are performed
		tf.io.gfile.exists(sf[1].numpy().decode())


	current_dir = os.getcwd()
	os.chdir(HOME_DIRECTORY)
	return current_dir #returns directory where files are stored in .tfrecord format

"""
	creates a subdirectory within the ROOT_TFRECORDS_DIR according to the process id of the python program running
	stores each tensor as an individual .tfrecord file, which could be suboptimal
	returns the full path to the directory that the tfrecords were stored to
"""

#read float32 image dataset from tfrecord directory


def load_tfrecord_recon_dataset(tfrecord_dir):	
	#read it from the tfrecord files, expecting shape [512, 512, 1] of float32
	#assumes storage of only tensor data in tfrecords
	#tfrecord_dir should be an absolute path

	file_pattern = f'{tfrecord_dir}/*.tfrecord'
	file_path_dataset = tf.data.Dataset.list_files(file_pattern)

	tfrecord_dataset = tf.data.TFRecordDataset(
				file_path_dataset,
				compression_type="",
				buffer_size=None,
				num_parallel_reads=tf.data.AUTOTUNE)

	parsed_dataset = tfrecord_dataset.map(parse_tfrecord,
					num_parallel_calls = tf.data.AUTOTUNE)
	#parsed_dataset = file_path_dataset.interleave(lambda x: 
	#	tf.data.TFRecordDataset(x).map(parse_tfrecord), 
	#	num_parallel_calls=tf.data.AUTOTUNE,
	#	deterministic=False)
	#^^^working, but unnecessarily tedious

	return parsed_dataset#shape [512,512,1], tf.float32

"""
	load the dataset from the directory of .tfrecord files
	tries to maximize parallelism, so data order is not preserved
"""

def load_original_recon_dataset(file_pattern):
	
	original_data_files = tf.data.Dataset.list_files(file_pattern)#file list dataset
	dataset = original_data_files.map(file_to_tensor, num_parallel_calls=tf.data.AUTOTUNE)#load tensor data
	tensor_label_dataset = (
	tf.data.Dataset.zip((dataset, original_data_files)))#pair x, labelx

	return tensor_label_dataset #dataset of tuples of tensor, label
	#the label is the original full file path to where the tensor was read from

def save_recon_dataset_as_tfrecords(tensor_label_dataset):
	os.chdir(HOME_DIRECTORY)
	#this does not change the label names to have the .tfrecord extension, nor
	#clips them from being absolute paths, that must all be done
	#mapping the function create_tfrecord_name to label portions
	#creates new directory for this dataset assuming program run from ~ directory
	os.makedirs(RELATIVE_ROOT_TFRECORDS_DIR, exist_ok=True)
	os.chdir(RELATIVE_ROOT_TFRECORDS_DIR)#change to root tfrecord dir
	PID = os.getpid()#get process id for unique directory name
	new_tfrecord_dir = f'{PID}tfrecord_dir'
	os.makedirs(new_tfrecord_dir, exist_ok=True)#create directory for tfrecord storage
	os.chdir(new_tfrecord_dir)#change to new tfrecord dir

	#shard it to tfrecord files, described element, x, wise

	serialized_example_proto_label_dataset = (
		tensor_label_dataset.map(tensor_label_to_serialized_example_proto, 
				num_parallel_calls=tf.data.AUTOTUNE))#create example of x
	written_dataset = (
		serialized_example_proto_label_dataset.map(write_proto_to_tfrecord, 
					num_parallel_calls=tf.data.AUTOTUNE))#write x tfrecord

	for tensor,label in written_dataset:
	#checks whether files have been created; ensures lazy-executions are performed
		tf.io.gfile.exists(label.numpy().decode())

	current_dir = os.getcwd()
	os.chdir(HOME_DIRECTORY)
	return current_dir #returns directory where files are stored in .tfrecord format

"""
	this assumes an input of a dataset of (tensor, label) tuples, where label is a tf.string
	and tensor is a 512,512,1 tensor of tf.float32
	creates a subdirectory within the ROOT_TFRECORDS_DIR according to the process id of the python program running
	stores each tensor in the dataset as an individual .tfrecord file, which could be suboptimal
	returns the full path to the directory that the tfrecords were stored to"""


#Begin Transformer Model Code
def create_patch_embeddings(inputs, patch_size=2, projection_dim=64): #Defines patch embedding function
        # Assuming inputs have shape (batch_size, height, width, channels)
        batch_size, height, width, channels = inputs.shape
        # Calculate the number of patches
        num_patches = (height // patch_size) * (width // patch_size)
        patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size)(inputs) #Convolves a filter that generates patches of the image
        # Reshape the output to have (num_patches, projection_dim)
        patches = layers.Reshape((num_patches, projection_dim))(patches)
              return patches

def positional_encoding(num_patches, projection_dim): #Def function for positional encoding the patches generated
        position = np.arange(num_patches)[:, np.newaxis] #Creates an array of the range up to num_patches, then makes it a 2d column vector
        div_term = np.exp(np.arange(0, projection_dim, 2) * -(np.log(10000.0) / projection_dim)) #This is the coeffient in the arg of the trig funcs based on pos
        pos_enc = np.zeros((num_patches, projection_dim)) #Creates an array of zeroes the same size as patches array
        pos_enc[:, 0::2] = np.sin(position * div_term) #Even indices are unique sine frequencies
        pos_enc[:, 1::2] = np.cos(position * div_term) #Odd indices are unique cos frequencies
        pos_enc = pos_enc[np.newaxis, ...]  # Shape: [1, num_patches, projection_dim]
        return tf.cast(pos_enc, dtype=tf.float32) #Casts as float for use in transformer

class TransformerBlock(layers.Layer):  #Creates the transformer block class
      def __init__(self, projection_dim, num_heads): #constructor  method with params of self, the dim of embedding space, and number of transformer heads
            super(TransformerBlock, self).__init__() #calls layers.layer parent class and init.
            self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim) #Attention layer
            self.ffn = tf.keras.Sequential([ #Dense forward projection/multilayer perceptron layer
                layers.Dense(projection_dim, activation="relu"),
                layers.Dense(projection_dim),
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6) #Normalizes perceptron outputs
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

      def call(self, inputs): #Defines method to add attention values to initial input vectors
            attn_output = self.attention(inputs, inputs) #creates a variable of the attention output
            out1 = self.layernorm1(inputs + attn_output) #adds to original and normalizes and stores as out1
            ffn_output = self.ffn(out1) #Runs this through feed forward
            return self.layernorm2(out1 + ffn_output) #Adds that output to output of attention layer

def build_transformer_model(num_patches, projection_dim, num_heads, transformer_layers, num_classes): #Function to build the transformer model
    inputs = layers.Input(shape=(512, 512, 1))  #Defines image shape as 512x512x1 tensor
    patches = create_patch_embeddings(inputs, patch_size=8, projection_dim=projection_dim) #Defines patches as patch embedding function called on images
    encoded_patches = patches + positional_encoding(num_patches, projection_dim)

    x = encoded_patches  #Variable storage of encoded patches
    for _ in range(transformer_layers): #For ever transformer layer
    x = TransformerBlock(projection_dim, num_heads)(x) #Iterate the transformer block on the patches

    # Use Conv2D and UpSampling2D layers to reconstruct the image
    x = layers.Reshape((64, 64, 64))(x) #Reshape as 64x64x64 tensor

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x) #Filter that image with 128 3x3 filters
    x = layers.UpSampling2D((2, 2))(x)  #Upsamples image to 128x128
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x) #Second filter pass
    x = layers.UpSampling2D((4, 4))(x)  #Gets us back to 512x512 image
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer to reconstruct the image

    return tf.keras.Model(inputs, outputs)  # Return the model object

num_patches = 4096 #Defines the number of patches used
projection_dim = 64 #This is the dimension of the embedding space of the patches (Where the patch vectors exist)
num_heads = 4 #Number of heads in the transformer
transformer_layers = 6 #How many transformer layers
num_classes = 4 #Not sure if this line is useful

model = build_transformer_model(num_patches, projection_dim, num_heads, transformer_layers, num_classes) #Builds the model with our attributes
model.compile(optimizer='adam', loss='mean_squared_error') #Uses mse loss


# Train model
model.fit(X_train, y_train, validation_split=0.1, epochs=250, batch_size=12)

# Save model
model.save('transformer_model.keras')
#End Transformer Model Code




#UNET MODEL CODE BEGIN
def ssim_loss(y_true, y_pred): #Defining a function of ssim loss image to image
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) #return the loss value

def mse_loss(y_true, y_pred): #Defining a mean squared error loss
    return tf.reduce_mean(tf.square(y_true - y_pred)) #Returning loss

def combined_loss(y_true, y_pred, alpha = 0.2, beta = 0.8): #Define a mixed loss with proportions alpha and beta
    return alpha * ssim_loss(y_true, y_pred) + beta * mse_loss(y_true, y_pred) #Return the sum of the weighted losses =1

def unet_model(input_size=(512, 512, 3)): #Defining the model
        inputs = tf.keras.Input(input_size)
            # Downsample
        ...
        c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs) #Initial convolutional layer
        c1 = layers.Dropout(0.1)(c1) #Drops 10% of neurons from layer 1
        c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) #Second convolution
        p1 = layers.MaxPooling2D((2, 2))(c1) #Max pools 2x2 regions

        c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) #Second Same as first section but filters x 2
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2) #Third same but filters x 2
        c3 = layers.Dropout(0.1)(c3)
        c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3) #Fourth same but filters x 2
        c4 = layers.Dropout(0.1)(c4)
        c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4) #Fourth has no maxpool and filters x 2
        c5 = layers.Dropout(0.1)(c5)
        c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) #Undoes previous convolve
        u6 = layers.concatenate([u6, c4], axis=3) #First skip connection
        c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6) #Filtering convolutional layer
        c6 = layers.Dropout(0.1)(c6) #Drop 10% neurons from layer 6
        c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6) #Second convolution and filter

        u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) #Same as first but half filters
        u7 = layers.concatenate([u7, c3], axis=3) #Second skip connection
        c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.1)(c7)
        c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #Same but half filters
        u8 = layers.concatenate([u8, c2], axis=3)
        c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) #Same but half filters
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c9) #Final 3 filter 1x1 kernel with sigmoid activation
        model = models.Model(inputs=inputs, outputs=[outputs]) #Compresses the function into a single variable "model"

        return model #Returns that variable
X_train, X_test, y_train, y_test = train_test_split(dirty_images, clean_images, test_size=0.2, random_state=42)

#defines model as the unet model
model = unet_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy']) #Compiles the model with adam optimizer and our mixed loss

# Train model
model.fit(X_train, y_train, validation_split=0.1, epochs=750, batch_size=16) #Trains the dirty images with clean images as target

# Save model
model.save('unet_model.h5') #Saves the model
#UNET MODEL CODE END




#DCNN Model Begins
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
X_train, y_train = load_real_data(noisy_dir, normal_dir, image_shape, limit=100)
model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=32, verbose=1)
# Save the model architecture and weights
model.save('dncnn_model.h5')
model.save_weights('dncnn_model.weights.h5')
#DNCNN Model Ends
