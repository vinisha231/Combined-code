import tensorflow as tf
import os
import sys
import glob
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

# FUNCTION LIST
"""
get_basename(path)
	_get_basename(path)

format_tfrecord_name(path)

file_to_tensor(path)

where_tensor_plt(where, tensor)

write_proto_to_tfrecord(proto, tfrecord)
	_write_proto_to_tfrecord(proto, tfrecord)

x_y_pair_to_serialized_example_proto(x,y)
	_x_y_pair_to_serialized_example_proto(x,y)
x_y_pair_identifier_to_serialized_example_proto(x,y, identifier)

parse_x_y_pair_from_tfrecord(tfrecord)

load_tfrecord_x_y_pair_recon_dataset(tfrecord_dir)

load_original_recon_dataset(file_pattern)

save_recon_dataset_as_tfrecords(x_y_pair_identifier_dataset)

get_i_of_yi_dataset_from_datasets(xi_dataset, yi_dataset)

xyi_dataset_from_xi_yi_datasets(xi_dataset, yi_dataset, 
		identifier_dataset_function=get_i_of_yi_dataset_from_datasets)
"""


# name: HOME_DIRECTORY
# purpose: the place where 'tfrecords_dir' directory will be placed,
# 	   and subsequent directories of *.tfrecord files representing
#	   datasets will be stored

HOME_DIRECTORY = '/home/NETID/bodhik/reudir/UWB_CT'

# name: ROOT_TFRECORDS_DIR
# purpose: directory name for the directory containing subdirectories
#	   representing datasets to be stored at
 
ROOT_TFRECORDS_DIR = 'tfrecords_dir'

TFRECORD_DIR = 'tfrecord_dir'

TFRECORD_EXT = tf.constant('.tfrecord')
#^^^ typical extension for data files

# name: _get_basename
# purpose: get 'last part' of 'absolute path', e.g., 
# 	   'file' of '/dir/file' 
# input: absolute path; as a char string
# output: file or directory name; as a char string

def _get_basename(path):
	return os.path.basename(path.numpy().decode())

#___ encapsulator for ^^^, returns a tf.string

@tf.function
def get_basename(path):
	return tf.py_function(_get_basename, [path], tf.string)

# name: format_tfrecord_name
# purpose: add '.tfrecord' to a shortened path, using ^^^, e.g.,
#	  bs.flt -> bs.flt.tfrecord
# input: path; as a char string
# output: 'get_basename(path).tfrecord' tf.string

def format_tfrecord_name(path):
	prename = get_basename(path)
	name = tf.strings.join([prename, TFRECORD_EXT])
	return name

# name: file_to_tensor
# purpose: read a file of 512*512 32-bit floats into a 512*512 
#	   tensor of type tf.float32
# input: absolute path to file as tf.string (?)
# output: tensor of shape 512*512*1 of tf.float32

def file_to_tensor(path):
	raw = tf.io.read_file(path)
	flt = tf.io.decode_raw(raw, tf.float32)
	tensor = tf.reshape(flt, [512, 512, 1])
	return tensor

# name: where_tensor_plt
# purpose: write heatmap subplot from 3x1D tensor of tf.float32
# input: array of 3 numbers, [x,y,z], where [x,y] are the row-col
# 	 dimensions of the plt.figure and 1<= z <=x*y
# output: it places the image of 'tensor' at position 'z'
# 	  of the figure, then increments 'z' of 'where' by 1

def where_tensor_plt(where, tensor):
	plt.subplot(*where)
	plt.imshow(tensor.numpy(), cmap='gray', vmin=0, vmax=1)
	plt.title(where[2])
	plt.axis("off")
	where[2] += 1
	return where

# name: _write_proto_to_tfrecord
# purpose: write proto to absolute path in 'tfrecord'
# input: 'proto' as serialized example proto, 'tfrecord' as tf.string
# output: returns inputs proto, tfrecord, and writes file
# TIP: beware of lazy execution

def _write_proto_to_tfrecord(proto, tfrecord):
	with tf.io.TFRecordWriter(tfrecord.numpy().decode('utf-8')) as writer:
		writer.write(proto.numpy())
	return proto, tfrecord

#___ encapsulator for ^^^, returns its inputs

@tf.function
def write_proto_to_tfrecord(proto, tfrecord):
	return 	tf.py_function(func=_write_proto_to_tfrecord, inp=[proto, tfrecord], Tout=(tf.string, tf.string))

# name: _x_y_pair_to_serialized_example_proto
# purpose: format (x,y) tuple of 512*512*1 tf.float tensors as a proto
# input: (x,y) tuple
# output: proto of (x,y)

def _x_y_pair_to_serialized_example_proto(x, y):
	serialized_x = tf.io.serialize_tensor(x)
	serialized_y = tf.io.serialize_tensor(y)

	feature_x = tf.train.Feature(bytes_list = (
		tf.train.BytesList(
			value = [serialized_x.numpy()])
			))

	feature_y = tf.train.Feature(bytes_list = (
		tf.train.BytesList(
			value = [serialized_y.numpy()])
			))
	
	features_proto = {
		'x': feature_x,
		'y': feature_y}

	example_proto = tf.train.Example(
		features = tf.train.Features(feature=features_proto))

	serialized_example_proto = example_proto.SerializeToString()
	
	return serialized_example_proto

#___ encapsulator for ^^^, returns proto

@tf.function
def x_y_pair_to_serialized_example_proto(x, y):
	return tf.py_function(
		func=_x_y_pair_to_serialized_example_proto, 
		inp=[x, y], Tout=tf.string)

#___ encapsulator for ^^^, handles the extra 'identifier' argument
#		by returning a (proto, identifier) tuple

@tf.function
def x_y_pair_identifier_to_serialized_example_proto(x, y, identifier):
	x_y_pair_example_proto = ( 
	tf.py_function(func=_x_y_pair_to_serialized_example_proto,
			 inp=[x,y], Tout=tf.string))
	return x_y_pair_example_proto, identifier


# name: parse_x_y_pair_from_tfrecord
# purpose: get (x,y) 512*512*1 tf.float tensors tuple from 
#	   tfrecord file
# input: absolute path 'tfrecord' to *.tfrecord file containing that
#	 (x,y) tensor tuple
# output: (x,y) tensor tuple
# TIP: order of (x,y) tuple extraction via methods in shard.py is
#      indeterminate


#@tf.function >>> this was giving an aggregious error
@tf.autograph.experimental.do_not_convert
def parse_x_y_pair_from_tfrecord(tfrecord):
	features_proto = {
		'x': tf.io.FixedLenFeature([], tf.string),
		'y': tf.io.FixedLenFeature([], tf.string)}

	serialized_tensor = (
		tf.io.parse_single_example(tfrecord, features_proto))
	x = tf.io.parse_tensor(
		serialized_tensor['x'], out_type=tf.float32)
	y = tf.io.parse_tensor(
		serialized_tensor['y'], out_type=tf.float32)
	return x,y

# name: load_tfrecord_x_y_pair_recon_dataset
# purpose: load an (x,y) 512*512*1 tf.float32 tensor tuple dataset
#	   from a directory of *.tfrecord files
# input: absolute path to that directory of '*.tfrecord' files
# output: dataset of (x,y) tuples

def load_tfrecord_x_y_pair_recon_dataset(tfrecord_dir):	
	#read it from the tfrecord files, expecting shape [512, 512, 1] of float32
	#assumes storage of only tensor data in tfrecords
	#tfrecord_dir should be an absolute path

	file_pattern = (
		f'{tfrecord_dir}/*{TFRECORD_EXT.numpy().decode()}')
	file_path_dataset = tf.data.Dataset.list_files(file_pattern)

	tfrecord_dataset = tf.data.TFRecordDataset(
				file_path_dataset,
				compression_type="",
				buffer_size=None,
				num_parallel_reads=tf.data.AUTOTUNE)

	parsed_dataset = tfrecord_dataset.map(
		parse_x_y_pair_from_tfrecord,
		num_parallel_calls = tf.data.AUTOTUNE)
	#parsed_dataset = file_path_dataset.interleave(lambda x: 
	#	tf.data.TFRecordDataset(x).map(parse_tfrecord), 
	#	num_parallel_calls=tf.data.AUTOTUNE,
	#	deterministic=False)
	#^^^working, but unnecessarily tedious

	return parsed_dataset#shape [512,512,1], tf.float32

# name: load_original_recon_dataset
# purpose: get all files matching the pattern of 'file_pattern'
#	   that contain 512*512*1 32-bit floats input without
#	   formatting (raw data) into a dataset of tensors, x,
#	   with shape=[512,512,1] and datatype tf.float32
# input: file_pattern as char string, including regexy things, like
#	 [0-9] for a digit 1-9 at that position, or '*' as a wildcard
# output: dataset of tuples (x, label), where x is a tensor, and
#	  'label' is the absolute path to the file x came from stored
#	  as a tf.string

def load_original_recon_dataset(file_pattern):
	
	original_data_files = (
		tf.data.Dataset.list_files(file_pattern))
		#^^^ file list dataset
	dataset = original_data_files.map(
		file_to_tensor, 
		num_parallel_calls=tf.data.AUTOTUNE)#load tensor data
	tensor_label_dataset = (
	tf.data.Dataset.zip((dataset, original_data_files)))
	#^^^ pair x, labelx

	return tensor_label_dataset

# name: save_recon_dataset_as_tfrecords
# purpose: save a dataset of tensor tuples (x,y)
#	   512*512*1 tf.float32 each, by unique identifiers
# input: dataset of (x, y, identifier) tuples, where
#	 EACH 'identifier' MUST BE A UNIQUE VALID FILENAME
# output: directory of 'identifier'.tfrecord files
# 	  comprising the dataset

def save_recon_dataset_as_tfrecords(x_y_pair_identifier_dataset):
	starting_directory = os.getcwd() 
	#^^^ save current directory for later
	os.chdir(HOME_DIRECTORY)
	os.makedirs(ROOT_TFRECORDS_DIR, exist_ok=True)
	os.chdir(ROOT_TFRECORDS_DIR)
	#^^^ change to root tfrecord dir
	PID = os.getpid()#get process id for unique directory name
	ver = 0

	while True: #makes sure directory is new
		new_tfrecord_dir = f'{PID}{TFRECORD_DIR}_{ver}'
		if not os.path.isdir(new_tfrecord_dir):
			break
		ver += 1	
		
	os.makedirs(new_tfrecord_dir, exist_ok=True)
	#^^^ create directory for tfrecord storage
	os.chdir(new_tfrecord_dir)#change to new tfrecord dir

	#___ write dataset to tfrecord files, (x,y) tuple-wise

	serialized_example_proto_identifier_dataset = (
		x_y_pair_identifier_dataset.map(
		x_y_pair_identifier_to_serialized_example_proto, 
		num_parallel_calls=tf.data.AUTOTUNE))
			#^^^ create example of each (x,y) tuple
	written_dataset = (
		serialized_example_proto_identifier_dataset.map(
			write_proto_to_tfrecord, 
			num_parallel_calls=tf.data.AUTOTUNE))
	#^^^ write (x,y) to tfrecord at 'identifier'

	for proto, identifier in written_dataset:
	#checks whether files have been created;
	#	ensures lazy-executions are performed
		tf.io.gfile.exists(identifier.numpy().decode())

	current_dir = os.getcwd()
	os.chdir(starting_directory) #return to starting directory
	return  current_dir
	#returns directory where files are stored in .tfrecord format


# name: get_i_of_yi_dataset_from_datasets
# purpose: select the identifier from second input dataset as identifier for combined xyi dataset
# inputs: xi_dataset, yi_dataset
# output: identifier_dataset; of tf.strings

def get_i_of_yi_dataset_from_datasets(xi_dataset, yi_dataset):
	i_of_yi_dataset = yi_dataset.map(
		lambda y, i: i, num_parallel_calls=tf.data.AUTOTUNE)
	return i_of_yi_dataset

# name: xyi_dataset_from_xi_yi_datasets
# purpose: collect tensor, tensorlabel, pairlabel data elements into a 3-tuple for storage
# inputs: x, identifier, y, identifier, and identifier selector argument
# output: x_y_pair_identifier dataset of tuples (x, y, identifier)

def xyi_dataset_from_xi_yi_datasets(
		xi_dataset, yi_dataset, 
		identifier_dataset_function=get_i_of_yi_dataset_from_datasets):

	x_dataset = xi_dataset.map(
		lambda x, i: x, num_parallel_calls=tf.data.AUTOTUNE)
	y_dataset = yi_dataset.map(
		lambda y, i: y, num_parallel_calls=tf.data.AUTOTUNE)
	
	identifier_dataset = identifier_dataset_function(xi_dataset, yi_dataset)

	xyi_dataset = tf.data.Dataset.zip(
			(x_dataset, y_dataset, identifier_dataset))
	
	return xyi_dataset
