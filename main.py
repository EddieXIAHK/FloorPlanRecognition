import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import random

from net import *

# No CUDA environment variable needed for DirectML
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 8964

# input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--phase', type=str, default='Test',
					help='Train/Test network.')

class MODEL:
	"""Main model class for training and evaluation"""
	def __init__(self):
		print('Initializing DeepFloorplan model...')
		self.log_dir = 'pretrained'
		self.eval_file = './dataset/r3d_test.txt'
		self.loss_type = 'balanced'

		# Create the model
		self.model = DeepFloorplanModel(num_room_classes=9, num_boundary_classes=3)

		# Build model by calling it once
		dummy_input = tf.zeros([1, 512, 512, 3])
		_ = self.model(dummy_input)

		print(f'Model created with {self.model.count_params():,} parameters')

	def convert_one_hot_to_image(self, one_hot, dtype='float'):
		"""Convert one-hot encoded predictions to class indices"""
		im = tf.argmax(one_hot, axis=-1)
		if dtype == 'int':
			im = tf.cast(im, dtype=tf.uint8)
		else:
			im = tf.cast(im, dtype=tf.float32)
		im = tf.expand_dims(im, axis=-1)
		return im

	def cross_two_tasks_weight(self, y1, y2):
		"""Compute dynamic weights for multi-task learning"""
		p1 = tf.reduce_sum(y1)
		p2 = tf.reduce_sum(y2)

		w1 = p2 / (p1 + p2 + 1e-8)
		w2 = p1 / (p1 + p2 + 1e-8)

		return w1, w2

	def balanced_entropy(self, logits, labels):
		"""
		Balanced cross-entropy loss for handling class imbalance.
		Weights each class inversely proportional to its frequency.
		"""
		eps = 1e-6
		predictions = tf.nn.softmax(logits, axis=-1)
		predictions = tf.clip_by_value(predictions, eps, 1-eps)
		log_predictions = tf.math.log(predictions)

		num_classes = labels.shape.as_list()[-1]
		ind = tf.argmax(labels, axis=-1, output_type=tf.int32)

		total = tf.reduce_sum(labels)

		# Compute per-class pixel counts
		m_c = []  # masks
		n_c = []  # counts
		for c in range(num_classes):
			mask = tf.cast(tf.equal(ind, c), dtype=tf.int32)
			m_c.append(mask)
			n_c.append(tf.cast(tf.reduce_sum(mask), dtype=tf.float32))

		# Compute weights (inverse frequency)
		c = []
		for i in range(num_classes):
			c.append(total - n_c[i])
		tc = tf.add_n(c)

		# Compute weighted loss
		loss = 0.
		for i in range(num_classes):
			w = c[i] / (tc + eps)
			m_c_one_hot = tf.one_hot(i * m_c[i], num_classes, axis=-1)
			y_c = m_c_one_hot * labels

			loss += w * tf.reduce_mean(-tf.reduce_sum(y_c * log_predictions, axis=-1))

		return loss / num_classes

	@tf.function
	def train_step(self, images, labels_room, labels_boundary, optimizer):
		"""Single training step with GradientTape"""
		with tf.GradientTape() as tape:
			# Forward pass
			logits_room, logits_boundary = self.model(images, training=True)

			# Compute losses
			if self.loss_type == 'balanced':
				loss_room = self.balanced_entropy(logits_room, labels_room)
				loss_boundary = self.balanced_entropy(logits_boundary, labels_boundary)
			else:
				loss_room = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(logits=logits_room, labels=labels_room))
				loss_boundary = tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(logits=logits_boundary, labels=labels_boundary))

			# Dynamic task weighting
			w1, w2 = self.cross_two_tasks_weight(labels_room, labels_boundary)
			total_loss = w1 * loss_room + w2 * loss_boundary

		# Compute gradients and update weights
		gradients = tape.gradient(total_loss, self.model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return total_loss, loss_room, loss_boundary

	def train(self, dataset, num_batch, max_step=40000):
		"""Train the model using TF2 eager execution"""
		max_ep = max_step // num_batch
		print(f'max_step = {max_step}, max_ep = {max_ep}, num_batch = {num_batch}')

		# Create optimizer
		optimizer = keras.optimizers.Adam(learning_rate=1e-4)

		# Create checkpoint manager
		checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.model)
		checkpoint_manager = tf.train.CheckpointManager(
			checkpoint, directory=self.log_dir, max_to_keep=10)

		# Restore latest checkpoint if exists
		if checkpoint_manager.latest_checkpoint:
			checkpoint.restore(checkpoint_manager.latest_checkpoint)
			print(f'Restored from {checkpoint_manager.latest_checkpoint}')
		else:
			print('Starting training from scratch')

		print("Start Training!")
		total_times = 0
		step = 0

		for ep in range(max_ep):
			print(f'\n=== Epoch {ep}/{max_ep} ===')

			for batch_idx, (images, label_boundaries, label_rooms, label_doors) in enumerate(dataset.take(num_batch)):
				tic = time.time()

				# Training step
				total_loss, loss_room, loss_boundary = self.train_step(
					images, label_rooms, label_boundaries, optimizer)

				duration = time.time() - tic
				total_times += duration

				# Log progress
				if step % 10 == 0:
					print(f'Step {step}: loss = {total_loss:.3f} (room: {loss_room:.3f}, boundary: {loss_boundary:.3f}); '
						  f'{1.0/duration:.2f} steps/sec, executed {int(total_times/60)} minutes')

				step += 1

			# Save checkpoint every 2 epochs
			if ep % 2 == 0:
				save_path = checkpoint_manager.save()
				print(f'Saved checkpoint: {save_path}')
				self.evaluate(epoch=ep)

		# Final save and evaluation
		save_path = checkpoint_manager.save()
		print(f'Final checkpoint saved: {save_path}')
		self.evaluate(epoch=max_ep)

	def infer(self, save_dir='out', resize=True, merge=True):
		"""Run inference on test set"""
		print(f"Generating test set predictions from {self.eval_file}... will save to [./{save_dir}]")

		room_dir = os.path.join(save_dir, 'room')
		close_wall_dir = os.path.join(save_dir, 'boundary')

		os.makedirs(save_dir, exist_ok=True)
		os.makedirs(room_dir, exist_ok=True)
		os.makedirs(close_wall_dir, exist_ok=True)

		# Load latest checkpoint
		checkpoint = tf.train.Checkpoint(model=self.model)
		checkpoint.restore(tf.train.latest_checkpoint(self.log_dir)).expect_partial()
		print(f'Restored model from {tf.train.latest_checkpoint(self.log_dir)}')

		# Infer one by one
		paths = open(self.eval_file, 'r').read().splitlines()
		paths = [p.split('\t')[0] for p in paths]

		for p in paths:
			im = imageio.imread(p)
			im_x = (skresize(im, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.float32) / 255.
			im_x = np.expand_dims(im_x, axis=0)

			# Run inference
			logits_room, logits_boundary = self.model(im_x, training=False)

			# Convert to class indices
			rooms = tf.argmax(logits_room, axis=-1).numpy()[0]
			close_walls = tf.argmax(logits_boundary, axis=-1).numpy()[0]

			if resize:
				# Convert to RGB and resize back
				out1_rgb = ind2rgb(rooms)
				out1_rgb = (skresize(out1_rgb, (im.shape[0], im.shape[1]), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
				out2_rgb = ind2rgb(close_walls, color_map=floorplan_boundary_map)
				out2_rgb = (skresize(out2_rgb, (im.shape[0], im.shape[1]), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
			else:
				out1_rgb = ind2rgb(rooms)
				out2_rgb = ind2rgb(close_walls, color_map=floorplan_boundary_map)

			if merge:
				out1 = rooms.copy()
				out2 = close_walls.copy()
				out1[out2==2] = 10
				out1[out2==1] = 9
				out3_rgb = ind2rgb(out1, color_map=floorplan_fuse_map)

			name = p.split('/')[-1]
			save_path1 = os.path.join(room_dir, name.split('.jpg')[0]+'_rooms.png')
			save_path2 = os.path.join(close_wall_dir, name.split('.jpg')[0]+'_bd_rm.png')
			save_path3 = os.path.join(save_dir, name.split('.jpg')[0]+'_rooms.png')

			imageio.imwrite(save_path1, out1_rgb.astype(np.uint8))
			imageio.imwrite(save_path2, out2_rgb.astype(np.uint8))
			if merge:
				imageio.imwrite(save_path3, out3_rgb.astype(np.uint8))

			print(f'Saved prediction: {name}')

	def evaluate(self, epoch=0, num_of_classes=11):
		"""Evaluate model on test set"""
		print(f'\n=== Evaluating at epoch {epoch} ===')

		paths = open(self.eval_file, 'r').read().splitlines()
		image_paths = [p.split('\t')[0] for p in paths]
		gt2_paths = [p.split('\t')[2] for p in paths]  # doors
		gt3_paths = [p.split('\t')[3] for p in paths]  # rooms
		gt4_paths = [p.split('\t')[-1] for p in paths]  # close wall

		n = len(paths)

		hist = np.zeros((num_of_classes, num_of_classes))
		for i in range(n):
			im = imageio.imread(image_paths[i])

			# Read ground truth
			dd = imageio.imread(gt2_paths[i])
			if len(dd.shape) == 3:
				dd = dd[:, :, 0]
			rr = imageio.imread(gt3_paths[i])
			cw = imageio.imread(gt4_paths[i])
			if len(cw.shape) == 3:
				cw = cw[:, :, 0]

			# Preprocess
			im = (skresize(im, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.float32) / 255.
			im = np.expand_dims(im, axis=0)

			# Merge label
			rr = (skresize(rr, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
			rr_ind = rgb2ind(rr)
			cw = (skresize(cw, (512, 512), preserve_range=True, anti_aliasing=True) / 255.0).astype(np.float32)
			dd = (skresize(dd, (512, 512), preserve_range=True, anti_aliasing=True) / 255.0).astype(np.float32)
			cw = (cw>0.5).astype(np.uint8)
			dd = (dd>0.5).astype(np.uint8)
			rr_ind[cw==1] = 10
			rr_ind[dd==1] = 9

			# Merge prediction
			logits_room, logits_boundary = self.model(im, training=False)
			rm_ind = tf.argmax(logits_room, axis=-1).numpy()[0]
			bd_ind = tf.argmax(logits_boundary, axis=-1).numpy()[0]
			rm_ind[bd_ind==2] = 10
			rm_ind[bd_ind==1] = 9

			hist += fast_hist(rm_ind.flatten(), rr_ind.flatten(), num_of_classes)

		overall_acc = np.diag(hist).sum() / hist.sum()
		mean_acc = np.diag(hist) / (hist.sum(1) + 1e-6)
		mean_acc9 = (np.nansum(mean_acc[:7])+mean_acc[-2]+mean_acc[-1]) / 9.

		with open(f'EVAL_{self.log_dir}', 'a') as file:
			file.write(f'Model at epoch {epoch}: overall accuracy = {overall_acc:.4f}, mean_acc = {mean_acc9:.4f}\n')
			for i in range(mean_acc.shape[0]):
				if i not in [7, 8]:  # ignore class 7 & 8
					file.write(f'\t\tepoch {epoch}: {i}th label: accuracy = {mean_acc[i]:.4f}\n')

		print(f'Epoch {epoch}: Overall accuracy = {overall_acc:.4f}, Mean accuracy (9 classes) = {mean_acc9:.4f}')

def main(args):
	# Set random seeds
	tf.random.set_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	model = MODEL()

	if args.phase.lower() == 'train':
		dataset, num_batch = data_loader_bd_rm_from_tfrecord(batch_size=1)

		# START TRAINING
		tic = time.time()
		model.train(dataset, num_batch)
		toc = time.time()
		print(f'Total training + evaluation time = {(toc-tic)/60:.2f} minutes')

	elif args.phase.lower() == 'test':
		model.infer()

	else:
		print(f'Unknown phase: {args.phase}')

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
