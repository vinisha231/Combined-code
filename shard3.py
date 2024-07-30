import re
import tensorflow as tf
import os
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# FUNCTION LIST
"""
get_basename(path)
    _get_basename(path)

file_to_tensor(path)

where_tensor_plt(where, tensor)

load_original_recon_dataset(file_pattern)

get_i_of_yi_dataset_from_datasets(xi_dataset, yi_dataset)

xyi_dataset_from_xi_yi_datasets(xi_dataset, yi_dataset)

xy_dataset_from_xi_yi_datasets(xi_dataset, yi_dataset)  

normalize_float32_tensor_0_1(tensor)
"""


# name: HOME_DIRECTORY
# purpose: the place where 'tfrecords_dir' directory will be placed,
#      and subsequent directories of *.tfrecord files representing
#      datasets will be stored

HOME_DIRECTORY = '/gscratch/uwb/bodhik/CT-CNN-Code'

# name: _get_basename
# purpose: get 'last part' of 'absolute path', e.g., 
#      'file' of '/dir/file' 
# input: absolute path; as a char string
# output: file or directory name; as a char string

def _get_basename(path):
    return os.path.basename(path.numpy().decode())

#___ encapsulator for ^^^, returns a tf.string

@tf.function
def get_basename(path):
    return tf.py_function(_get_basename, [path], tf.string)

# name: file_to_tensor
# purpose: read a file of 512*512 32-bit floats into a 512*512 
#      tensor of type tf.float32
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
#    dimensions of the plt.figure and 1<= z <=x*y
# output: it places the image of 'tensor' at position 'z'
#     of the figure, then increments 'z' of 'where' by 1

def where_tensor_plt(where, tensor):
    plt.subplot(*where)
    plt.imshow(tensor, cmap='gray', vmin=0, vmax=1)
    plt.title(where[2])
    plt.axis("off")
    where[2] += 1
    return where

def basename_numeric_prefix(filename):
    basename = os.path.basename(filename)
    match = re.match(r'^(\d+)_', basename)
    if match:
        return float(match.group(1))
    else:
        return float('inf')


# name: load_original_recon_dataset
# purpose: get all files matching the pattern of 'file_pattern'
#      that contain 512*512*1 32-bit floats input without
#      formatting (raw data) into a dataset of tensors, x,
#      with shape=[512,512,1] and datatype tf.float32
# input: file_pattern as char string, including regexy things, like
#    [0-9] for a digit 1-9 at that position, or '*' as a wildcard
# output: dataset of tuples (x, label), where x is a tensor, and
#     'label' is the absolute path to the file x came from stored
#     as a tf.string

def load_original_recon_dataset(file_pattern):
    
    file_list = glob.glob(file_pattern)
    sorted_file_list = sorted(file_list, key=basename_numeric_prefix)

    original_data_files = tf.data.Dataset.from_tensor_slices(sorted_file_list)
        #^^^ file list dataset
    dataset = original_data_files.map(
    file_to_tensor, 
    num_parallel_calls=tf.data.AUTOTUNE)#load tensor data
    tensor_label_dataset = (
    tf.data.Dataset.zip((dataset, original_data_files)))
    #^^^ pair x, labelx

    return tensor_label_dataset

# name: get_i_of_yi_dataset_from_datasets
# purpose: select the identifier from second input dataset as identifier for combined xyi dataset
# inputs: xi_dataset, yi_dataset
# output: identifier_dataset; of tf.strings

def get_i_of_yi_dataset_from_datasets(xi_dataset, yi_dataset):
    i_of_yi_dataset = yi_dataset.map(
    get_i_of_xi_dataset_from_datasets,
        lambda y, i: i, num_parallel_calls=tf.data.AUTOTUNE)
    return i_of_yi_dataset

# name: xyi_dataset_from_xi_yi_datasets
# purpose: collect tensor, tensorlabel, pairlabel data elements into a 3-tuple for storage
# inputs: x, identifier, y, identifier
# output: x_y_pair_identifier dataset of tuples (x, y, identifier)

def xyi_dataset_from_xi_yi_datasets(
        xi_dataset, yi_dataset, 
        identifier_dataset_function=get_i_of_yi_dataset_from_datasets):

    x_dataset = xi_dataset.map(
        lambda x, i: x, num_parallel_calls=tf.data.AUTOTUNE)
    y_dataset = yi_dataset.map(
        lambda y, i: y, num_parallel_calls=tf.data.AUTOTUNE)
    i_dataset = yi_dataset.map(
        lambda y, i: i, num_parallel_calls=tf.data.AUTOTUNE)

    xyi_dataset = tf.data.Dataset.zip(
            (x_dataset, y_dataset, i_dataset))
    
    return xyi_dataset

# name: xy_dataset_from_xi_yi_datasets
# repeats the functionality of above, but without doing anything with identifiers

def xy_dataset_from_xi_yi_datasets(xi_dataset, yi_dataset):
    x_dataset = xi_dataset.map(
    lambda x, i: x, num_parallel_calls=tf.data.AUTOTUNE)
    y_dataset = yi_dataset.map(
    lambda y, i: y, num_parallel_calls=tf.data.AUTOTUNE)
    
    xy_dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
    
    return xy_dataset

def normalize_float32_tensor_0_1(tensor):
    return (tensor - tf.reduce_min(tensor)) / (tf.reduce_max(tensor) - tf.reduce_min(tensor)) 

def configure_for_performance(dataset, bufferSize, batchSize):
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=bufferSize)
    dataset = dataset.batch(batchSize).repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def fltbatch_plt(fltBatch, fltBatchSize, baseNum, name, imageOutDir):
    for i in range(fltBatchSize):
        fltname = f'{imageOutDir}/{i+baseNum:0{8}}_{name}.png'
        plt.figure(figsize=[1,1], dpi=600)
        plt.subplot(1,1,1)
        plt.imshow(fltBatch[i], cmap='gray', vmin=0, vmax=1)
        plt.title("")
        plt.axis("off")
        plt.savefig(fltname, dpi=600)
        plt.close()

def fltbatch_flt(fltBatch, fltBatchSize, baseNum, name, fltOutDir):
    for i in range(fltBatchSize):
        fltname = f'{fltOutDir}/{i+baseNum:0{8}}_{name}.flt'
        serialized_tensor = tf.io.serialize_tensor(fltBatch[i])
        with open(fltname, 'wb') as f:
            f.write(serialized_tensor.numpy())
