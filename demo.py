import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from skimage.transform import resize as skresize
from matplotlib import pyplot as plt

# Add utils to path for post-processing
sys.path.append('./utils/')

# Disable TensorFlow 2 behavior to use TF1 compatibility mode
tf.compat.v1.disable_v2_behavior()

# DirectML doesn't use CUDA environment variables
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='Input image path.')
parser.add_argument('--save_dxf', action='store_true',
                    help='Export results to DXF format.')
parser.add_argument('--dxf_output', type=str, default='./output/floorplan.dxf',
                    help='Output DXF file path.')
parser.add_argument('--scale', type=float, default=0.1,
                    help='Scale factor for DXF export (e.g., 0.05 for pixels to meters).')
parser.add_argument('--apply_postprocess', action='store_true',
                    help='Apply post-processing before DXF export for cleaner vectors.')
parser.add_argument('--no_display', action='store_true',
                    help='Skip matplotlib display (useful for batch processing).')

# color map
floorplan_map = {
	0: [255,255,255], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
	3: [224,255,192], # livingroom/kitchen/dining room
	4: [255,224,128], # bedroom
	5: [255,160, 96], # hall
	6: [255,224,224], # balcony
	7: [255,255,255], # not used
	8: [255,255,255], # not used
	9: [255, 60,128], # door & window
	10:[  0,  0,  0]  # wall
}

def ind2rgb(ind_im, color_map=floorplan_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im

def main(args):
	# load input
	im = imageio.imread(args.im_path)

	# Convert to RGB if needed (handle grayscale or RGBA images)
	if len(im.shape) == 2:
		# Grayscale to RGB
		im = np.stack([im, im, im], axis=-1)
	elif im.shape[2] == 4:
		# RGBA to RGB (drop alpha channel)
		im = im[:, :, :3]

	im = im.astype(np.float32)
	im = (skresize(im, (512, 512), preserve_range=True, anti_aliasing=True)).astype(np.float32) / 255.

	# create tensorflow session
	with tf.compat.v1.Session() as sess:

		# initialize
		sess.run(tf.group(tf.compat.v1.global_variables_initializer(),
					tf.compat.v1.local_variables_initializer()))

		# restore pretrained model
		saver = tf.compat.v1.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
		saver.restore(sess, './pretrained/pretrained_r3d')

		# get default graph
		graph = tf.compat.v1.get_default_graph()

		# restore inputs & outpus tensor
		x = graph.get_tensor_by_name('inputs:0')
		room_type_logit = graph.get_tensor_by_name('Cast:0')
		room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

		# infer results
		[room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
										feed_dict={x:im.reshape(1,512,512,3)})
		room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

		# merge results
		floorplan = room_type.copy()
		floorplan[room_boundary==1] = 9
		floorplan[room_boundary==2] = 10

		# Apply post-processing if requested (for cleaner vectors)
		if args.apply_postprocess or args.save_dxf:
			print("Applying post-processing to refine predictions...")
			from export_dxf import apply_post_processing
			floorplan = apply_post_processing(room_type, room_boundary)

		# Export to DXF if requested
		if args.save_dxf:
			from export_dxf import export_to_dxf

			# Create output directory if needed
			output_dir = os.path.dirname(args.dxf_output)
			if output_dir and not os.path.exists(output_dir):
				os.makedirs(output_dir)

			print(f"\nExporting to DXF format...")
			print(f"  Scale: {args.scale}")
			print(f"  Output: {args.dxf_output}")

			export_to_dxf(floorplan, args.dxf_output, scale=args.scale)

		# Visualization
		if not args.no_display:
			floorplan_rgb = ind2rgb(floorplan)

			# plot results
			plt.subplot(121)
			plt.imshow(im)
			plt.title('Input Floor Plan')
			plt.subplot(122)
			plt.imshow(floorplan_rgb/255.)
			plt.title('Recognition Result')
			plt.show()

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
