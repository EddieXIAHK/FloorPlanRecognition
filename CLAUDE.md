# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DeepFloorplan is a deep learning project for floor plan recognition using a multi-task network with room-boundary-guided attention (ICCV 2019). The system recognizes room types and boundaries in floor plan images using a VGG16-based encoder-decoder architecture with dual task heads.

## Technology Stack

- **Python 2.7** (legacy codebase)
- **TensorFlow 1.10.1** (TensorFlow 1.x API)
- **GPU**: CUDA 9.0, tested on Nvidia Titan Xp
- **Dependencies**: numpy, scipy, Pillow, matplotlib, OpenCV 3.1.0

## Common Commands

### Demo/Inference
Run inference on a single image using pretrained model:
```bash
python demo.py --im_path=./demo/45719584.jpg
```

### Training
Train the multi-task network:
```bash
python main.py --phase=Train
```

### Testing
Generate predictions for test set (saves to `./out` directory):
```bash
python main.py --phase=Test
```

### Evaluation
Compute accuracy and IoU metrics on a dataset:
```bash
python scores.py --dataset=R3D
```
or
```bash
python scores.py --dataset=R2V
```

### Post-processing
Refine network predictions using morphological operations:
```bash
python postprocess.py
```
or
```bash
python postprocess.py --result_dir=./[result_folder_path]
```

## Architecture Overview

### Multi-Task Network Design

The network has three main components:

1. **FNet (Feature Extraction Network)**: VGG16-based encoder shared by both task branches
   - 5 convolutional blocks (conv1-conv5) with max pooling
   - Can be initialized with pretrained VGG16 weights
   - Defined in `net.py` starting at line 277

2. **CWNet (Close Wall Network)**: Decoder for boundary detection (walls, doors, windows)
   - 4 upsampling stages (up2, up4, up8, up16)
   - Outputs 3-class segmentation: background, door/window, wall
   - Defined in `net.py` starting at line 312

3. **RNet (Room Network)**: Decoder for room type classification
   - 4 upsampling stages with non-local context attention modules
   - Uses boundary features from CWNet via `_non_local_context()` to guide room segmentation
   - Outputs 9-class segmentation for room types
   - Defined in `net.py` starting at line 335

### Key Architectural Features

- **Room-boundary-guided attention**: RNet uses boundary predictions from CWNet at multiple scales through non-local context modules (`_non_local_context()` in `net.py:221-268`)
- **Skip connections**: Encoder features are combined with upsampled decoder features
- **Balanced loss**: Custom balanced cross-entropy loss (`balanced_entropy()` in `main.py:46-80`) to handle class imbalance
- **Cross-task weighting**: Dynamic loss weighting between room and boundary tasks (`cross_two_tasks_weight()` in `main.py:37-44`)

## Label Encoding

### Room Types (9 classes)
0. Background
1. Closet
2. Bathroom/Washroom
3. Living room/Kitchen/Dining room
4. Bedroom
5. Hall
6. Balcony
7. Not used
8. Not used

### Boundary Types (3 classes)
0. Background
1. Door & Window (opening)
2. Wall line

### Fused Output (11 classes)
Combines room types (0-8) + door/window (9) + wall (10)

Color mappings are defined in `utils/rgb_ind_convertor.py`.

## Data Format

### Dataset Structure
- Training/test splits defined in `dataset/r3d_train.txt` and `dataset/r3d_test.txt` (R3D dataset) or `dataset/r2v_train.txt` and `dataset/r2v_test.txt` (R2V dataset)
- Each line format: `<image_path>\t<wall_label>\t<door_label>\t<room_label>\t<close_wall_label>`
- Annotations are PNG images with specific suffixes:
  - `_wall.png`: wall annotations
  - `_close.png`: door & window annotations
  - `_room.png`: room type annotations
  - `_close_wall.png`: combined wall + door/window
  - `_multi.png`: all labels combined

### TFRecord Format
For faster training data loading, use TFRecord format:
- Create TFRecords using `utils/create_tfrecord.py`
- Training code expects TFRecords at `../dataset/r3d.tfrecords`
- Data loader: `data_loader_bd_rm_from_tfrecord()` in `net.py:22-29`

## Important Implementation Details

### TensorFlow 1.x Specifics
- Uses legacy `tf.contrib.slim` for VGG16
- Uses deprecated APIs: `scipy.misc.imread/imresize/imsave`, `xrange`, `print` statements
- Session-based execution (not eager execution)
- Queue-based data loading with coordinators

### GPU Configuration
- GPU ID is hardcoded in `net.py:20` (`GPU_ID = '0'`)
- Change `os.environ['CUDA_VISIBLE_DEVICES']` to use different GPU
- Model uses device placement: `/device:GPU:0`

### Model Checkpoints
- Pretrained model location: `./pretrained/pretrained_r3d.meta` and `./pretrained/pretrained_r3d`
- Training saves checkpoints to `self.log_dir` (default: `pretrained/`)
- Checkpoints saved every 2 epochs during training (`main.py:155-156`)

### Network Outputs
- Room type logits: `logits_r` (shape: [N, H, W, 9])
- Boundary logits: `logits_cw` (shape: [N, H, W, 3])
- Predictions are softmax-converted and argmax-reduced to class indices
- Results are resized back to original image dimensions

### Post-Processing Pipeline
The `postprocess.py` script refines predictions:
1. Separate fused predictions into room and boundary masks
2. Fill broken boundary lines using morphological closing (`fill_break_line()`)
3. Fill holes using flood fill algorithm (`flood_fill()`)
4. Refine room regions: one connected component = one room label (`refine_room_region()`)
5. Merge boundary predictions back into final output

## File Organization

- `net.py`: Core network architecture (Network base class)
- `main.py`: Training/testing entry point (MODEL class extends Network)
- `demo.py`: Single-image inference demo
- `scores.py`: Evaluation metrics (accuracy, IoU)
- `postprocess.py`: Post-processing for prediction refinement
- `utils/rgb_ind_convertor.py`: Color map definitions and RGB↔index conversion
- `utils/util.py`: Helper functions (histogram, flood fill, morphological ops)
- `utils/tf_record.py`: TFRecord reading/writing utilities
- `utils/create_tfrecord.py`: Script to create TFRecord datasets

## Training Configuration

- **Batch size**: 1 (hardcoded in `main.py:296`)
- **Learning rate**: 1e-4 (Adam optimizer)
- **Max steps**: 40,000 (default in `main.py:82`)
- **Loss type**: 'balanced' (set in `main.py:22`)
- **Input size**: 512×512×3 (all images resized)
- **Random seed**: 8964 (for reproducibility)
- **Evaluation**: Runs on test set every 2 epochs during training

## Known Issues & Considerations

- **Python 2.7**: This is legacy code. Be aware of Python 2 syntax (print statements, xrange, dict.iteritems())
- **Deprecated dependencies**: scipy.misc functions are deprecated; consider using imageio or PIL
- **Fixed paths**: Some paths are hardcoded (e.g., `../dataset/r3d_train.txt` in `net.py:23`)
- **Class 7 & 8**: Room type classes 7 and 8 are unused and ignored in evaluation
- **VGG16 pretrained weights**: Required for training from scratch, path expected at `./vgg16/vgg_16.ckpt`
