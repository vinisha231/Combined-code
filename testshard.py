import tensorflow as tf
import matplotlib
matplotlib.use("qtagg")
import os
import sys
import matplotlib.pyplot as plt
from shard import (get_basename, save_original_files_to_tfrecord_recon_dataset, load_tfrecord_recon_dataset,
		  load_original_recon_dataset, save_recon_dataset_as_tfrecords)

data_origin = '/data/CT_images/train/images/0000200[0-2]*.flt'
stored_dataset_dir = save_original_files_to_tfrecord_recon_dataset(data_origin)
tf.print(stored_dataset_dir)
loaded_dataset = load_tfrecord_recon_dataset(stored_dataset_dir)

data_origin2 = '/data/CT_images/train/images/0000200[3-4]*.flt'
loaded_dataset2 = load_original_recon_dataset(data_origin2)
#crops the name to the filename
loaded_dataset2 = loaded_dataset2.map(lambda x, pathlikex: (x, get_basename(pathlikex)), num_parallel_calls=tf.data.AUTOTUNE)
#stores with original filename
stored_dataset_dir2 = save_recon_dataset_as_tfrecords(loaded_dataset2)#saves alternate dataset to same directory
tf.print(stored_dataset_dir2)

#where = [1,3,1]
#plt.figure(figsize=where[0:2])
#for tensor in loaded_dataset:
#	where_tensor_plt(where, tensor)
#plt.show()
	 
