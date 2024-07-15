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
	returns the full path to the directory that the tfrecords were stored to
"""
