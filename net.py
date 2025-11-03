import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import sys
import glob
import time
import random

import imageio.v2 as imageio
from skimage.transform import resize as skresize

sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import fast_hist
from tf_record import read_bd_rm_record

# No CUDA environment variable needed for DirectML
# GPU selection is handled automatically by TensorFlow with DirectML

def data_loader_bd_rm_from_tfrecord(batch_size=1):
	"""Load boundary+room data using TF2's tf.data API"""
	paths = open('./dataset/r3d_train.txt', 'r').read().splitlines()

	dataset = read_bd_rm_record('./dataset/r3d.tfrecords', batch_size=batch_size, size=512)

	num_batch = len(paths) // batch_size

	return dataset, num_batch


class DeepFloorplanModel(keras.Model):
	"""
	TensorFlow 2 / Keras implementation of DeepFloorplan multi-task network.

	Architecture:
	- FNet: VGG16-based feature extraction (shared encoder)
	- CWNet: Boundary detection decoder (walls, doors, windows)
	- RNet: Room type decoder with attention guided by CWNet features
	"""

	def __init__(self, num_room_classes=9, num_boundary_classes=3, **kwargs):
		super(DeepFloorplanModel, self).__init__(**kwargs)

		self.num_room_classes = num_room_classes
		self.num_boundary_classes = num_boundary_classes

		# Feature extraction network (VGG16-style encoder)
		self._build_feature_extractor()

		# Boundary detection decoder (CWNet)
		self._build_boundary_decoder()

		# Room type decoder with attention (RNet)
		self._build_room_decoder()

	def _build_feature_extractor(self):
		"""Build VGG16-style feature extraction network (FNet)"""
		# Block 1
		self.conv1_1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='FNet/conv1_1')
		self.conv1_2 = layers.Conv2D(64, 3, padding='same', activation='relu', name='FNet/conv1_2')
		self.pool1 = layers.MaxPooling2D(2, 2, name='FNet/pool1')

		# Block 2
		self.conv2_1 = layers.Conv2D(128, 3, padding='same', activation='relu', name='FNet/conv2_1')
		self.conv2_2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='FNet/conv2_2')
		self.pool2 = layers.MaxPooling2D(2, 2, name='FNet/pool2')

		# Block 3
		self.conv3_1 = layers.Conv2D(256, 3, padding='same', activation='relu', name='FNet/conv3_1')
		self.conv3_2 = layers.Conv2D(256, 3, padding='same', activation='relu', name='FNet/conv3_2')
		self.conv3_3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='FNet/conv3_3')
		self.pool3 = layers.MaxPooling2D(2, 2, name='FNet/pool3')

		# Block 4
		self.conv4_1 = layers.Conv2D(512, 3, padding='same', activation='relu', name='FNet/conv4_1')
		self.conv4_2 = layers.Conv2D(512, 3, padding='same', activation='relu', name='FNet/conv4_2')
		self.conv4_3 = layers.Conv2D(512, 3, padding='same', activation='relu', name='FNet/conv4_3')
		self.pool4 = layers.MaxPooling2D(2, 2, name='FNet/pool4')

		# Block 5
		self.conv5_1 = layers.Conv2D(512, 3, padding='same', activation='relu', name='FNet/conv5_1')
		self.conv5_2 = layers.Conv2D(512, 3, padding='same', activation='relu', name='FNet/conv5_2')
		self.conv5_3 = layers.Conv2D(512, 3, padding='same', activation='relu', name='FNet/conv5_3')
		self.pool5 = layers.MaxPooling2D(2, 2, name='FNet/pool5')

	def _build_boundary_decoder(self):
		"""Build boundary detection decoder (CWNet)"""
		# Upsample layers
		self.cw_up2_deconv = layers.Conv2DTranspose(256, 4, strides=2, padding='same', name='CWNet/up2_deconv')
		self.cw_up2_conv = layers.Conv2D(256, 1, padding='same', activation='linear', name='CWNet/up2_conv')
		self.cw_up2_relu = layers.ReLU(name='CWNet/up2_relu')
		self.cw_up2_final = layers.Conv2D(256, 3, padding='same', activation='relu', name='CWNet/up2_final')

		self.cw_up4_deconv = layers.Conv2DTranspose(128, 4, strides=2, padding='same', name='CWNet/up4_deconv')
		self.cw_up4_conv = layers.Conv2D(128, 1, padding='same', activation='linear', name='CWNet/up4_conv')
		self.cw_up4_relu = layers.ReLU(name='CWNet/up4_relu')
		self.cw_up4_final = layers.Conv2D(128, 3, padding='same', activation='relu', name='CWNet/up4_final')

		self.cw_up8_deconv = layers.Conv2DTranspose(64, 4, strides=2, padding='same', name='CWNet/up8_deconv')
		self.cw_up8_conv = layers.Conv2D(64, 1, padding='same', activation='linear', name='CWNet/up8_conv')
		self.cw_up8_relu = layers.ReLU(name='CWNet/up8_relu')
		self.cw_up8_final = layers.Conv2D(64, 3, padding='same', activation='relu', name='CWNet/up8_final')

		self.cw_up16_deconv = layers.Conv2DTranspose(32, 4, strides=2, padding='same', name='CWNet/up16_deconv')
		self.cw_up16_conv = layers.Conv2D(32, 1, padding='same', activation='linear', name='CWNet/up16_conv')
		self.cw_up16_relu = layers.ReLU(name='CWNet/up16_relu')
		self.cw_up16_final = layers.Conv2D(32, 3, padding='same', activation='relu', name='CWNet/up16_final')

		# Final prediction layers
		self.cw_logits_conv = layers.Conv2D(self.num_boundary_classes, 1, padding='same', activation='linear', name='CWNet/logits_conv')
		self.cw_logits_upsample = layers.UpSampling2D(2, interpolation='bilinear', name='CWNet/logits_upsample')

	def _build_room_decoder(self):
		"""Build room type decoder with attention (RNet)"""
		# Upsample layers with attention
		self.r_up2_deconv = layers.Conv2DTranspose(256, 4, strides=2, padding='same', name='RNet/up2_deconv')
		self.r_up2_conv = layers.Conv2D(256, 1, padding='same', activation='linear', name='RNet/up2_conv')
		self.r_up2_relu = layers.ReLU(name='RNet/up2_relu')
		self.r_up2_combine = layers.Conv2D(256, 3, padding='same', activation='relu', name='RNet/up2_combine')
		# Attention layers for up2
		self.r_up2_attn_conv1 = layers.Conv2D(256, 3, padding='same', activation='relu', name='RNet/up2_attn_conv1')
		self.r_up2_attn_conv2 = layers.Conv2D(256, 3, padding='same', activation='relu', name='RNet/up2_attn_conv2')
		self.r_up2_attn_final = layers.Conv2D(256, 1, padding='same', activation='linear', name='RNet/up2_attn_final')
		self.r_up2_attn_gate = layers.Conv2D(256, 1, padding='same', activation='linear', name='RNet/up2_attn_gate')
		self.r_up2_context_final = layers.Conv2D(256, 3, padding='same', activation='relu', name='RNet/up2_context_final')

		self.r_up4_deconv = layers.Conv2DTranspose(128, 4, strides=2, padding='same', name='RNet/up4_deconv')
		self.r_up4_conv = layers.Conv2D(128, 1, padding='same', activation='linear', name='RNet/up4_conv')
		self.r_up4_relu = layers.ReLU(name='RNet/up4_relu')
		self.r_up4_combine = layers.Conv2D(128, 3, padding='same', activation='relu', name='RNet/up4_combine')
		# Attention layers for up4
		self.r_up4_attn_conv1 = layers.Conv2D(128, 3, padding='same', activation='relu', name='RNet/up4_attn_conv1')
		self.r_up4_attn_conv2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='RNet/up4_attn_conv2')
		self.r_up4_attn_final = layers.Conv2D(128, 1, padding='same', activation='linear', name='RNet/up4_attn_final')
		self.r_up4_attn_gate = layers.Conv2D(128, 1, padding='same', activation='linear', name='RNet/up4_attn_gate')
		self.r_up4_context_final = layers.Conv2D(128, 3, padding='same', activation='relu', name='RNet/up4_context_final')

		self.r_up8_deconv = layers.Conv2DTranspose(64, 4, strides=2, padding='same', name='RNet/up8_deconv')
		self.r_up8_conv = layers.Conv2D(64, 1, padding='same', activation='linear', name='RNet/up8_conv')
		self.r_up8_relu = layers.ReLU(name='RNet/up8_relu')
		self.r_up8_combine = layers.Conv2D(64, 3, padding='same', activation='relu', name='RNet/up8_combine')
		# Attention layers for up8
		self.r_up8_attn_conv1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='RNet/up8_attn_conv1')
		self.r_up8_attn_conv2 = layers.Conv2D(64, 3, padding='same', activation='relu', name='RNet/up8_attn_conv2')
		self.r_up8_attn_final = layers.Conv2D(64, 1, padding='same', activation='linear', name='RNet/up8_attn_final')
		self.r_up8_attn_gate = layers.Conv2D(64, 1, padding='same', activation='linear', name='RNet/up8_attn_gate')
		self.r_up8_context_final = layers.Conv2D(64, 3, padding='same', activation='relu', name='RNet/up8_context_final')

		self.r_up16_deconv = layers.Conv2DTranspose(32, 4, strides=2, padding='same', name='RNet/up16_deconv')
		self.r_up16_conv = layers.Conv2D(32, 1, padding='same', activation='linear', name='RNet/up16_conv')
		self.r_up16_relu = layers.ReLU(name='RNet/up16_relu')
		self.r_up16_combine = layers.Conv2D(32, 3, padding='same', activation='relu', name='RNet/up16_combine')
		# Attention layers for up16
		self.r_up16_attn_conv1 = layers.Conv2D(32, 3, padding='same', activation='relu', name='RNet/up16_attn_conv1')
		self.r_up16_attn_conv2 = layers.Conv2D(32, 3, padding='same', activation='relu', name='RNet/up16_attn_conv2')
		self.r_up16_attn_final = layers.Conv2D(32, 1, padding='same', activation='linear', name='RNet/up16_attn_final')
		self.r_up16_attn_gate = layers.Conv2D(32, 1, padding='same', activation='linear', name='RNet/up16_attn_gate')
		self.r_up16_context_final = layers.Conv2D(32, 3, padding='same', activation='relu', name='RNet/up16_context_final')

		# Final prediction layers
		self.r_logits_conv = layers.Conv2D(self.num_room_classes, 1, padding='same', activation='linear', name='RNet/logits_conv')
		self.r_logits_upsample = layers.UpSampling2D(2, interpolation='bilinear', name='RNet/logits_upsample')

	def _apply_attention(self, boundary_features, room_features, attn_conv1, attn_conv2, attn_final, attn_gate, context_final):
		"""
		Apply room-boundary-guided attention mechanism.
		Uses boundary features to guide room segmentation.
		"""
		# Compute attention map from boundary features
		attn = attn_conv1(boundary_features)
		attn = attn_conv2(attn)
		attn = attn_final(attn)
		attn = tf.nn.sigmoid(attn)

		# Apply attention to room features
		gated = attn_gate(room_features)
		attended = attn * gated

		# Combine with original features
		combined = tf.concat([room_features, attended], axis=-1)
		output = context_final(combined)

		return output

	def call(self, inputs, training=False):
		"""
		Forward pass of the network.

		Args:
			inputs: Input images [batch, height, width, 3]
			training: Whether in training mode

		Returns:
			logits_room: Room type logits [batch, height, width, num_room_classes]
			logits_boundary: Boundary logits [batch, height, width, num_boundary_classes]
		"""
		# Feature extraction (shared encoder - FNet)
		x = self.conv1_1(inputs)
		x = self.conv1_2(x)
		pool1 = self.pool1(x)

		x = self.conv2_1(pool1)
		x = self.conv2_2(x)
		pool2 = self.pool2(x)

		x = self.conv3_1(pool2)
		x = self.conv3_2(x)
		x = self.conv3_3(x)
		pool3 = self.pool3(x)

		x = self.conv4_1(pool3)
		x = self.conv4_2(x)
		x = self.conv4_3(x)
		pool4 = self.pool4(x)

		x = self.conv5_1(pool4)
		x = self.conv5_2(x)
		x = self.conv5_3(x)
		pool5 = self.pool5(x)

		# Boundary detection decoder (CWNet)
		cw_up2 = self.cw_up2_deconv(pool5)
		cw_up2 = cw_up2 + self.cw_up2_conv(pool4)
		cw_up2 = self.cw_up2_relu(cw_up2)
		cw_up2 = self.cw_up2_final(cw_up2)

		cw_up4 = self.cw_up4_deconv(cw_up2)
		cw_up4 = cw_up4 + self.cw_up4_conv(pool3)
		cw_up4 = self.cw_up4_relu(cw_up4)
		cw_up4 = self.cw_up4_final(cw_up4)

		cw_up8 = self.cw_up8_deconv(cw_up4)
		cw_up8 = cw_up8 + self.cw_up8_conv(pool2)
		cw_up8 = self.cw_up8_relu(cw_up8)
		cw_up8 = self.cw_up8_final(cw_up8)

		cw_up16 = self.cw_up16_deconv(cw_up8)
		cw_up16 = cw_up16 + self.cw_up16_conv(pool1)
		cw_up16 = self.cw_up16_relu(cw_up16)
		cw_up16 = self.cw_up16_final(cw_up16)

		logits_boundary = self.cw_logits_conv(cw_up16)
		logits_boundary = self.cw_logits_upsample(logits_boundary)

		# Room type decoder with attention (RNet)
		r_up2 = self.r_up2_deconv(pool5)
		r_up2 = r_up2 + self.r_up2_conv(pool4)
		r_up2 = self.r_up2_relu(r_up2)
		r_up2 = self.r_up2_combine(r_up2)
		# Apply attention from boundary features
		r_up2 = self._apply_attention(cw_up2, r_up2,
									   self.r_up2_attn_conv1, self.r_up2_attn_conv2,
									   self.r_up2_attn_final, self.r_up2_attn_gate,
									   self.r_up2_context_final)

		r_up4 = self.r_up4_deconv(r_up2)
		r_up4 = r_up4 + self.r_up4_conv(pool3)
		r_up4 = self.r_up4_relu(r_up4)
		r_up4 = self.r_up4_combine(r_up4)
		r_up4 = self._apply_attention(cw_up4, r_up4,
									   self.r_up4_attn_conv1, self.r_up4_attn_conv2,
									   self.r_up4_attn_final, self.r_up4_attn_gate,
									   self.r_up4_context_final)

		r_up8 = self.r_up8_deconv(r_up4)
		r_up8 = r_up8 + self.r_up8_conv(pool2)
		r_up8 = self.r_up8_relu(r_up8)
		r_up8 = self.r_up8_combine(r_up8)
		r_up8 = self._apply_attention(cw_up8, r_up8,
									   self.r_up8_attn_conv1, self.r_up8_attn_conv2,
									   self.r_up8_attn_final, self.r_up8_attn_gate,
									   self.r_up8_context_final)

		r_up16 = self.r_up16_deconv(r_up8)
		r_up16 = r_up16 + self.r_up16_conv(pool1)
		r_up16 = self.r_up16_relu(r_up16)
		r_up16 = self.r_up16_combine(r_up16)
		r_up16 = self._apply_attention(cw_up16, r_up16,
									    self.r_up16_attn_conv1, self.r_up16_attn_conv2,
									    self.r_up16_attn_final, self.r_up16_attn_gate,
									    self.r_up16_context_final)

		logits_room = self.r_logits_conv(r_up16)
		logits_room = self.r_logits_upsample(logits_room)

		return logits_room, logits_boundary
