import tensorflow as tf
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
import os
import sys
from shard import (
	load_original_recon_dataset,
	get_basename, where_tensor_plt,
	save_recon_dataset_as_tfrecords)

batchSize = 32

def poisson_noise_to_normalized_flt_x_y_pairs(fltTensor, 
	fltTensorLabel):

	poissonTensor = tf.random.poisson(lam=0.0025,
				shape=tf.shape(fltTensor),
				dtype=tf.float32, seed=29)
	combinedTensor = fltTensor + poissonTensor
	combinedTensor = tf.clip_by_value(combinedTensor, 0.0, 1.0)
	return combinedTensor, fltTensorLabel

def random_uniform_brightness_shift_x_y_pairs(fltTensor, 
	fltTensorLabel):

	ur_delta = tf.random.uniform([], minval=-0.06, maxval=-0.06, 
				dtype=tf.float32, seed=14)
	shiftedTensor = tf.image.adjust_brightness(fltTensor,
						delta=ur_delta)
	shiftedTensor = tf.clip_by_value(shiftedTensor, 0.0, 1.0)
	
	shiftedTensorLabel = tf.image.adjust_brightness(
		fltTensorLabel, delta=ur_delta)
	shiftedTensorLabel = tf.clip_by_value(shiftedTensor, 0.0, 1.0)

	return shiftedTensor, shiftedTensorLabel

def random_uniform_rotation_x_y_pairs(fltTensor, fltTensorLabel):
	ur_delta = tf.random.uniform([],
		minval = 0, maxval = 4.0, dtype=tf.float32, seed=9)
	ur_delta = tf.math.ceil(ur_delta)
	k_rotations = tf.cast(ur_delta, dtype=tf.int32)
	rotatedTensor = tf.image.rot90(fltTensor, k=k_rotations)
	rotatedTensorLabel = (
		tf.image.rot90(fltTensorLabel, k=k_rotations))
	return rotatedTensor, rotatedTensorLabel

def random_noise_brightness_rotation_x_y_pairs(fltTensor,
	fltTensorLabel):

	flt, fltTensorLabel = (
		random_uniform_rotation_x_y_pairs(fltTensor,
		fltTensorLabel))#identical rotation
	flt, fltTensorLabel = (
		random_uniform_brightness_shift_x_y_pairs(fltTensor,
		fltTensorLabel))#identical brightness shift

	fltTensor, fltTensorLabel = (
		poisson_noise_to_normalized_flt_x_y_pairs(fltTensor, 
		fltTensorLabel)) #adds noise to fltTensor only
	return fltTensor, fltTensorLabel

def configure_for_performance(dataset):
	dataset.cache()
	dataset.shuffle(buffer_size=320)
	dataset.batch(batchSize)
	dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	return dataset


fltTensor_file_pattern = (
	'/data/CT_images/RECONS2024/60views/00000*.flt')
fltTensorLabel_file_pattern = (
	'/data/CT_images/RECONS2024/900views/00000*.flt')
fltTensor_dataset = load_original_recon_dataset(
				fltTensor_file_pattern)
fltTensor_dataset = fltTensor_dataset.map(
	lambda t, l: t, num_parallel_calls=tf.data.AUTOTUNE)
fltTensorLabel_dataset = load_original_recon_dataset(
				fltTensorLabel_file_pattern)
fltTensorLabel_dataset = fltTensorLabel_dataset.map(
	lambda t, l: t, num_parallel_calls=tf.data.AUTOTUNE)

flt_x_y_pair_dataset = tf.data.Dataset.zip(
	(fltTensor_dataset, fltTensorLabel_dataset))

flt_x_y_pair_dataset_PMD = configure_for_performance(
		flt_x_y_pair_dataset)



#where = [2,4,1]
#plt.figure(figsize=where[0:2])
#for x_y_pair in flt_x_y_pair_dataset:
#	where_tensor_plt(where, x_y_pair[0])
#	where_tensor_plt(where, x_y_pair[1])
#plt.show()


