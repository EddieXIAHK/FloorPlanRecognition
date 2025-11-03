import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from skimage.transform import resize as skresize
from matplotlib import pyplot as plt
from rgb_ind_convertor import *

import os
import sys
import glob
import time

def load_raw_images(path):
	paths = path.split('\t')

	image = imageio.imread(paths[0])
	wall  = imageio.imread(paths[1])
	if len(wall.shape) == 3:
		wall = wall[:, :, 0]
	close = imageio.imread(paths[2])
	if len(close.shape) == 3:
		close = close[:, :, 0]
	room  = imageio.imread(paths[3])
	close_wall = imageio.imread(paths[4])
	if len(close_wall.shape) == 3:
		close_wall = close_wall[:, :, 0]

	# NOTE: resize will preserve data type
	image = (skresize(image, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
	wall = (skresize(wall, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
	close = (skresize(close, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
	close_wall = (skresize(close_wall, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
	room = (skresize(room, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)

	room_ind = rgb2ind(room)

	# make sure the dtype is uint8
	image = image.astype(np.uint8)
	wall = wall.astype(np.uint8)
	close = close.astype(np.uint8)
	close_wall = close_wall.astype(np.uint8)
	room_ind = room_ind.astype(np.uint8)

	return image, wall, close, room_ind, close_wall

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_record(paths, name='dataset.tfrecords'):
	writer = tf.io.TFRecordWriter(name)

	for i in range(len(paths)):
		# Load the image
		image, wall, close, room_ind, close_wall = load_raw_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(image.tobytes()),
					'wall': _bytes_feature(wall.tobytes()),
					'close': _bytes_feature(close.tobytes()),
					'room': _bytes_feature(room_ind.tobytes()),
					'close_wall': _bytes_feature(close_wall.tobytes())}

		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())

	writer.close()

def read_record(data_path, batch_size=1, size=512):
	"""Read TFRecord using TF2's tf.data API"""
	feature_description = {
		'image': tf.io.FixedLenFeature([], tf.string),
		'wall': tf.io.FixedLenFeature([], tf.string),
		'close': tf.io.FixedLenFeature([], tf.string),
		'room': tf.io.FixedLenFeature([], tf.string),
		'close_wall': tf.io.FixedLenFeature([], tf.string)
	}

	def parse_function(serialized_example):
		# Parse the input tf.train.Example proto
		features = tf.io.parse_single_example(serialized_example, feature_description)

		# Convert from string back to numbers
		image = tf.io.decode_raw(features['image'], tf.uint8)
		wall = tf.io.decode_raw(features['wall'], tf.uint8)
		close = tf.io.decode_raw(features['close'], tf.uint8)
		room = tf.io.decode_raw(features['room'], tf.uint8)
		close_wall = tf.io.decode_raw(features['close_wall'], tf.uint8)

		# Cast data
		image = tf.cast(image, dtype=tf.float32)
		wall = tf.cast(wall, dtype=tf.float32)
		close = tf.cast(close, dtype=tf.float32)
		close_wall = tf.cast(close_wall, dtype=tf.float32)

		# Reshape image data into the original shape
		image = tf.reshape(image, [size, size, 3])
		wall = tf.reshape(wall, [size, size, 1])
		close = tf.reshape(close, [size, size, 1])
		room = tf.reshape(room, [size, size])
		close_wall = tf.reshape(close_wall, [size, size, 1])

		# Normalize
		image = tf.divide(image, 255.0)
		wall = tf.divide(wall, 255.0)
		close = tf.divide(close, 255.0)
		close_wall = tf.divide(close_wall, 255.0)

		# Generate one hot room label
		room_one_hot = tf.one_hot(room, 9, axis=-1)

		return image, wall, close, room_one_hot, close_wall

	# Create dataset
	dataset = tf.data.TFRecordDataset(data_path)
	dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size=batch_size*128)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)

	return dataset

# ------------------------------------------------------------------------------------------------------------------------------------- *
# Following are only for segmentation task, merge all label into one

def load_seg_raw_images(path):
	paths = path.split('\t')

	image = imageio.imread(paths[0])
	close = imageio.imread(paths[2])
	if len(close.shape) == 3:
		close = close[:, :, 0]
	room  = imageio.imread(paths[3])
	close_wall = imageio.imread(paths[4])
	if len(close_wall.shape) == 3:
		close_wall = close_wall[:, :, 0]

	# NOTE: resize will preserve range
	image = (skresize(image, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
	close = (skresize(close, (512, 512), preserve_range=True, anti_aliasing=True) / 255.0).astype(np.float32)
	close_wall = (skresize(close_wall, (512, 512), preserve_range=True, anti_aliasing=True) / 255.0).astype(np.float32)
	room = (skresize(room, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)

	room_ind = rgb2ind(room)

	# merge result
	d_ind = (close>0.5).astype(np.uint8)
	cw_ind = (close_wall>0.5).astype(np.uint8)
	room_ind[cw_ind==1] = 10
	room_ind[d_ind==1] = 9

	# make sure the dtype is uint8
	image = image.astype(np.uint8)
	room_ind = room_ind.astype(np.uint8)

	return image, room_ind

def write_seg_record(paths, name='dataset.tfrecords'):
	writer = tf.io.TFRecordWriter(name)

	for i in range(len(paths)):
		# Load the image
		image, room_ind = load_seg_raw_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(image.tobytes()),
					'label': _bytes_feature(room_ind.tobytes())}

		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())

	writer.close()

def read_seg_record(data_path, batch_size=1, size=512):
	"""Read segmentation TFRecord using TF2's tf.data API"""
	feature_description = {
		'image': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.string)
	}

	def parse_function(serialized_example):
		features = tf.io.parse_single_example(serialized_example, feature_description)

		# Convert from string back to numbers
		image = tf.io.decode_raw(features['image'], tf.uint8)
		label = tf.io.decode_raw(features['label'], tf.uint8)

		# Cast data
		image = tf.cast(image, dtype=tf.float32)

		# Reshape image data into the original shape
		image = tf.reshape(image, [size, size, 3])
		label = tf.reshape(label, [size, size])

		# Normalize
		image = tf.divide(image, 255.0)

		# Generate one hot room label
		label_one_hot = tf.one_hot(label, 11, axis=-1)

		return image, label_one_hot

	# Create dataset
	dataset = tf.data.TFRecordDataset(data_path)
	dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size=batch_size*128)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)

	return dataset

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- *
# ------------------------------------------------------------------------------------------------------------------------------------- *
# Following are only for multi-task network. Two labels(boundary and room.)

def load_bd_rm_images(path):
	paths = path.split('\t')

	image = imageio.imread(paths[0])
	close = imageio.imread(paths[2])
	if len(close.shape) == 3:
		close = close[:, :, 0]
	room  = imageio.imread(paths[3])
	close_wall = imageio.imread(paths[4])
	if len(close_wall.shape) == 3:
		close_wall = close_wall[:, :, 0]

	# NOTE: resize will preserve range
	image = (skresize(image, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
	close = (skresize(close, (512, 512), preserve_range=True, anti_aliasing=True) / 255.0).astype(np.float32)
	close_wall = (skresize(close_wall, (512, 512), preserve_range=True, anti_aliasing=True) / 255.0).astype(np.float32)
	room = (skresize(room, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)

	room_ind = rgb2ind(room)

	# merge result
	d_ind = (close>0.5).astype(np.uint8)
	cw_ind = (close_wall>0.5).astype(np.uint8)

	cw_ind[cw_ind==1] = 2
	cw_ind[d_ind==1] = 1

	# make sure the dtype is uint8
	image = image.astype(np.uint8)
	room_ind = room_ind.astype(np.uint8)
	cw_ind = cw_ind.astype(np.uint8)

	return image, cw_ind, room_ind, d_ind

def write_bd_rm_record(paths, name='dataset.tfrecords'):
	writer = tf.io.TFRecordWriter(name)

	for i in range(len(paths)):
		# Load the image
		image, cw_ind, room_ind, d_ind = load_bd_rm_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(image.tobytes()),
					'boundary': _bytes_feature(cw_ind.tobytes()),
					'room': _bytes_feature(room_ind.tobytes()),
					'door': _bytes_feature(d_ind.tobytes())}

		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())

	writer.close()

def read_bd_rm_record(data_path, batch_size=1, size=512):
	"""Read boundary+room TFRecord using TF2's tf.data API"""
	feature_description = {
		'image': tf.io.FixedLenFeature([], tf.string),
		'boundary': tf.io.FixedLenFeature([], tf.string),
		'room': tf.io.FixedLenFeature([], tf.string),
		'door': tf.io.FixedLenFeature([], tf.string)
	}

	def parse_function(serialized_example):
		features = tf.io.parse_single_example(serialized_example, feature_description)

		# Convert from string back to numbers
		image = tf.io.decode_raw(features['image'], tf.uint8)
		boundary = tf.io.decode_raw(features['boundary'], tf.uint8)
		room = tf.io.decode_raw(features['room'], tf.uint8)
		door = tf.io.decode_raw(features['door'], tf.uint8)

		# Cast data
		image = tf.cast(image, dtype=tf.float32)

		# Reshape image data into the original shape
		image = tf.reshape(image, [size, size, 3])
		boundary = tf.reshape(boundary, [size, size])
		room = tf.reshape(room, [size, size])
		door = tf.reshape(door, [size, size])

		# Normalize
		image = tf.divide(image, 255.0)

		# Generate one hot labels
		label_boundary = tf.one_hot(boundary, 3, axis=-1)
		label_room = tf.one_hot(room, 9, axis=-1)

		return image, label_boundary, label_room, door

	# Create dataset
	dataset = tf.data.TFRecordDataset(data_path)
	dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size=batch_size*128)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)

	return dataset
