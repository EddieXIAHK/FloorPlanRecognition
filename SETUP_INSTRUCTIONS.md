# Setup Instructions for Python 3 + TensorFlow 2 + DirectML

This document provides step-by-step instructions to set up your environment for training DeepFloorplan with AMD GPU support.

## Prerequisites

- **Windows 10/11** (version 1709 or later)
- **Python 3.10** (REQUIRED - NOT 3.11 or newer!)
  - TensorFlow 2.10 with DirectML only supports Python 3.7-3.10
  - **Python 3.10.11 recommended** (latest 3.10.x release)
- **AMD Radeon 780M GPU** (or any DirectX 12-compatible GPU)
- **At least 8GB RAM** (16GB recommended)

## ⚠️ CRITICAL: Python Version Check

**Before proceeding, verify your Python version:**

```bash
python --version
```

**You MUST see:** `Python 3.10.x` (e.g., Python 3.10.11)

**If you see Python 3.11, 3.12, or 3.13:**
- TensorFlow 2.10 will NOT install
- You MUST install Python 3.10 (see Step 1 below)

## Step 1: Install Python 3.10

**If you already have Python 3.10.x, skip to Step 2.**

If you don't have Python 3.10 installed:

1. Download **Python 3.10.11** from: https://www.python.org/downloads/release/python-31011/
   - Click "Windows installer (64-bit)" at the bottom
2. During installation:
   - ✅ Check "Add Python 3.10 to PATH"
   - ✅ Choose "Install for all users" (optional but recommended)
3. Verify installation:
   ```bash
   python --version
   ```
   Should show: `Python 3.10.11`

### If You Have Multiple Python Versions

If you have both Python 3.10 and 3.11+ installed, use the **Python Launcher** to select the correct version:

```bash
# Check available Python versions
py -0

# Use Python 3.10 specifically
py -3.10 --version

# Create virtual environment with Python 3.10
py -3.10 -m venv venv_tf2
```

## Step 2: Create Virtual Environment

Open Command Prompt or PowerShell in the DeepFloorplan directory:

```bash
# Create virtual environment with Python 3.10
# If you have multiple Python versions, use:
py -3.10 -m venv venv_tf2

# OR if Python 3.10 is your only/default version:
python -m venv venv_tf2

# Activate virtual environment
# On Windows Command Prompt:
venv_tf2\Scripts\activate.bat

# On Windows PowerShell:
venv_tf2\Scripts\Activate.ps1

# On Git Bash or similar:
source venv_tf2/Scripts/activate
```

Your prompt should now show `(venv_tf2)` prefix.

**Verify you're using Python 3.10 in the virtual environment:**
```bash
python --version
# Should show: Python 3.10.11
```

## Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow 2.10.0 (CPU version)
- tensorflow-directml-plugin (AMD GPU acceleration)
- imageio, scikit-image (replaces scipy.misc)
- OpenCV, Pillow, matplotlib
- NumPy (compatible version for TF 2.10)

**Important**: The installation may take 5-10 minutes. Be patient!

## Step 5: Verify GPU Detection

Run the GPU detection test:

```bash
python test_gpu.py
```

**Expected output:**
```
Python version: 3.10.x
TensorFlow version: 2.10.0

Available devices:
  CPU: /physical_device:CPU:0
  DML: /physical_device:DML:0

✓ DirectML GPU detected! Found 1 DML device(s)
  DML Device 0: /physical_device:DML:0

Testing simple computation...
Matrix multiplication result:
[[19. 22.]
 [43. 50.]]

Convolution output shape: (1, 26, 26, 32)

✓ TensorFlow 2.10 with DirectML is working correctly!
```

**If you see "No DirectML GPU detected":**
- Ensure your GPU drivers are up to date
- Restart your computer
- Try reinstalling tensorflow-directml-plugin:
  ```bash
  pip uninstall tensorflow-directml-plugin
  pip install tensorflow-directml-plugin
  ```

## Step 6: Monitor GPU Usage During Training

To verify your AMD GPU is being used during training:

1. Open **Task Manager** (Ctrl+Shift+Esc)
2. Go to **Performance** tab
3. Look for **GPU 1** (your AMD Radeon 780M)
4. During training, you should see:
   - GPU utilization increasing
   - Dedicated GPU memory usage increasing
   - 3D/Compute usage activity

## Troubleshooting

### Issue: "ImportError: DLL load failed"
- Ensure you have latest Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue: "No module named 'tensorflow'"
- Make sure virtual environment is activated: `venv_tf2\Scripts\activate`
- Reinstall: `pip install tensorflow-cpu==2.10.0`

### Issue: GPU not detected but code runs
- TensorFlow will fall back to CPU if GPU unavailable
- Check Windows DirectX support: Run `dxdiag` and verify DirectX 12 support

### Issue: Very slow training even with GPU
- DirectML is typically 30-50% slower than CUDA on NVIDIA GPUs
- For Radeon 780M (integrated GPU), expect training to take 2-3x longer than NVIDIA discrete GPUs
- Consider training with smaller batch sizes or reduced epochs

## Next Steps

After successful setup, the migration of code files will begin. All Python files will be updated to:
- Python 3 syntax
- TensorFlow 2.x APIs
- Modern image processing libraries

Training command will be:
```bash
python main.py --phase=Train
```

## Performance Expectations

With AMD Radeon 780M:
- **Demo inference**: 1-2 seconds per image
- **Training**: ~30-60 seconds per epoch (depends on dataset size)
- **Full training** (40k steps): Several hours to 1-2 days

The 780M is an integrated GPU sharing system memory, so training will be slower than dedicated GPUs but much faster than CPU-only training.
