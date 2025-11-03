"""
Test script to verify DirectML GPU detection and TensorFlow 2.10 installation.
Run this after setting up your environment to confirm AMD GPU support.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"\nTensorFlow version: {tf.__version__}")

    # Check for DirectML plugin
    try:
        import tensorflow.python.framework.ops
        print("DirectML plugin check...")
    except ImportError as e:
        print(f"Warning: DirectML plugin import issue: {e}")

    # List available devices
    print("\n" + "="*60)
    print("Available devices:")
    print("="*60)

    devices = tf.config.list_physical_devices()
    for device in devices:
        print(f"  {device.device_type}: {device.name}")

    # Check specifically for DML devices
    dml_devices = tf.config.list_physical_devices('DML')
    if dml_devices:
        print(f"\n✓ DirectML GPU detected! Found {len(dml_devices)} DML device(s)")
        for i, device in enumerate(dml_devices):
            print(f"  DML Device {i}: {device.name}")
    else:
        print("\n✗ No DirectML GPU detected. Training will use CPU only.")
        print("  Make sure tensorflow-directml-plugin is installed correctly.")

    # Test a simple computation
    print("\n" + "="*60)
    print("Testing simple computation...")
    print("="*60)

    with tf.device('/DML:0') if dml_devices else tf.device('/CPU:0'):
        # Create simple tensors
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Matrix multiplication result:\n{c.numpy()}")

        # Test convolution (common in neural networks)
        x = tf.random.normal([1, 28, 28, 3])
        conv = tf.keras.layers.Conv2D(32, 3, activation='relu')
        y = conv(x)
        print(f"\nConvolution output shape: {y.shape}")

    print("\n" + "="*60)
    print("✓ TensorFlow 2.10 with DirectML is working correctly!")
    print("="*60)

except ImportError as e:
    print(f"\n✗ Error importing TensorFlow: {e}")
    print("\nPlease install required packages:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error during GPU test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
