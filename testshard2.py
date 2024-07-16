import tensorflow as tf
import matplotlib
matplotlib.use("qtagg")
import os
import sys
import matplotlib.pyplot as plt
from shard import (
	get_basename, format_tfrecord_name,
	where_tensor_plt, 
	load_tfrecord_x_y_pair_recon_dataset,
	load_original_recon_dataset, 
	save_recon_dataset_as_tfrecords,
	get_i_of_yi_dataset_from_datasets,
	xyi_dataset_from_xi_yi_datasets)

def editname(x, name):
	return x, format_tfrecord_name(name)

data_origin1 = (
	'/data/CT_images/RECONS2024/60views/0000200[0-2]*.flt')
loaded_dataset1 = (
	load_original_recon_dataset(data_origin1))#load input data

where = [1,3,1]#print out tensors to verify them
plt.figure(figsize=where[0:2])
for tensor, label in loaded_dataset1:
	where_tensor_plt(where, tensor)
plt.show()

data_origin2 = '/data/CT_images/RECONS2024/900views/0000200[0-2]*.flt'
loaded_dataset2= (
	load_original_recon_dataset(data_origin2))#load label data
loaded_dataset2 = loaded_dataset2.map(
	editname, num_parallel_calls=tf.data.AUTOTUNE)#format title

for e in loaded_dataset2.take(1):
	tf.print("NAME:",e[1])#verify formatted titles

#nolabel_x = loaded_dataset1.map(#create x, y, identifier datasets
#	lambda x, label: x, num_parallel_calls=tf.data.AUTOTUNE)
#nolabel_y = loaded_dataset2.map(
#	lambda y, label: y, num_parallel_calls=tf.data.AUTOTUNE)
#label_y = loaded_dataset2.map(
#	lambda y, label: label, num_parallel_calls=tf.data.AUTOTUNE) 

x_y_pair_identifier_dataset = (
	xyi_dataset_from_xi_yi_datasets(
		loaded_dataset1, loaded_dataset2,
		get_i_of_yi_dataset_from_datasets))

#x_y_pair_identifier_dataset = tf.data.Dataset.zip(
	#combine data elements to (x,y,identifier) tuples
#	(nolabel_x, nolabel_y, label_y))

where = [1,2,1]#print the content of a data entry in ^^^
plt.figure(figsize=where[0:2])
for e in x_y_pair_identifier_dataset.take(1):
	where_tensor_plt(where, e[0])
	where_tensor_plt(where, e[1])
	tf.print("label of xyi element ^^^:",e[2])
plt.show()

dataset_directory = save_recon_dataset_as_tfrecords(
	x_y_pair_identifier_dataset)
tf.print(dataset_directory)#save to dataset directory and print where

retrieved_xyi_dataset = (#retrieve from that directory
	load_tfrecord_x_y_pair_recon_dataset(dataset_directory))

where = [1,2,1]#print the content of a data entry in ^^^
plt.figure(figsize=where[0:2])
for e in retrieved_xyi_dataset.take(1):
	where_tensor_plt(where, e[0])
	where_tensor_plt(where, e[1])
	tf.print("a retrieved xyi element ^^^")
plt.show()
