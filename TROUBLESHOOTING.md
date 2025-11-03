# Troubleshooting Guide

Common issues and solutions when setting up DeepFloorplan with TensorFlow 2 + DirectML.

---

## Issue 1: "Could not find a version that satisfies the requirement tensorflow-cpu==2.10.0"

### Symptoms:
```
ERROR: Could not find a version that satisfies the requirement tensorflow-cpu==2.10.0
ERROR: No matching distribution found for tensorflow-cpu==2.10.0
```

### Cause:
You're using **Python 3.11 or newer**. TensorFlow 2.10 only supports Python 3.7-3.10.

### Solution:

1. **Check your Python version:**
   ```bash
   python --version
   ```

2. **Install Python 3.10.11:**
   - Download from: https://www.python.org/downloads/release/python-31011/
   - During installation, check "Add Python 3.10 to PATH"

3. **Create virtual environment with Python 3.10:**
   ```bash
   # If you have multiple Python versions
   py -3.10 -m venv venv_tf2

   # Activate
   venv_tf2\Scripts\activate

   # Verify
   python --version  # Should show 3.10.x
   ```

4. **Retry installation:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Issue 2: "No module named 'tensorflow'"

### Symptoms:
```
ModuleNotFoundError: No module named 'tensorflow'
```

### Cause:
Virtual environment not activated OR installation failed.

### Solution:

1. **Activate virtual environment:**
   ```bash
   venv_tf2\Scripts\activate
   ```

2. **Verify installation:**
   ```bash
   pip list | findstr tensorflow
   ```

3. **Reinstall if needed:**
   ```bash
   pip install tensorflow-cpu==2.10.0
   pip install tensorflow-directml-plugin
   ```

---

## Issue 3: "DirectML GPU not detected"

### Symptoms:
Running `python test_gpu.py` shows:
```
✗ No DirectML GPU detected. Training will use CPU only.
```

### Cause:
- DirectML plugin not installed
- GPU drivers outdated
- DirectX 12 not supported

### Solution:

1. **Verify DirectML plugin is installed:**
   ```bash
   pip list | findstr directml
   ```
   Should show: `tensorflow-directml-plugin`

2. **Update GPU drivers:**
   - Download latest AMD drivers from: https://www.amd.com/en/support
   - Restart computer after installation

3. **Check DirectX 12 support:**
   - Press Win+R, type `dxdiag`, press Enter
   - Under "System" tab, check "DirectX Version" shows DirectX 12

4. **Reinstall DirectML plugin:**
   ```bash
   pip uninstall tensorflow-directml-plugin
   pip install tensorflow-directml-plugin
   ```

5. **Restart and retry:**
   - Restart your computer
   - Run `python test_gpu.py` again

---

## Issue 4: "ImportError: DLL load failed"

### Symptoms:
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

### Cause:
Missing Visual C++ Redistributable.

### Solution:

1. **Install Visual C++ Redistributable:**
   - Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Run installer
   - Restart computer

2. **Retry import:**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

---

## Issue 5: NumPy version incompatibility

### Symptoms:
```
ERROR: tensorflow-cpu 2.10.0 requires numpy<1.24,>=1.21.0
```

### Cause:
NumPy version too new or too old.

### Solution:

1. **Install compatible NumPy version:**
   ```bash
   pip install "numpy>=1.21.0,<1.24.0"
   ```

2. **Verify:**
   ```bash
   pip show numpy
   ```

---

## Issue 6: Training is very slow

### Symptoms:
- Training takes hours per epoch
- GPU usage in Task Manager shows 0%

### Possible Causes & Solutions:

1. **DirectML not using GPU:**
   - Run `python test_gpu.py` to verify GPU detection
   - Check Task Manager → Performance → GPU 1 (Radeon 780M) during training

2. **Integrated GPU limitations:**
   - Radeon 780M is an integrated GPU (slower than discrete GPUs)
   - Expected: 30-60 sec/epoch for small datasets
   - This is normal! AMD integrated GPUs are 2-3x slower than NVIDIA discrete GPUs

3. **Reduce batch size if OOM:**
   - Training already uses batch_size=1
   - Can't reduce further

4. **Alternative: Use cloud GPU:**
   - Google Colab (free NVIDIA GPU): https://colab.research.google.com/
   - Just need to upload code and data

---

## Issue 7: "Failed to load native TensorFlow runtime"

### Symptoms:
```
tensorflow.python.framework.errors_impl.NotFoundError: Failed to load the native TensorFlow runtime
```

### Cause:
TensorFlow installation corrupted.

### Solution:

1. **Completely reinstall TensorFlow:**
   ```bash
   pip uninstall tensorflow tensorflow-cpu tensorflow-directml-plugin
   pip cache purge
   pip install tensorflow-cpu==2.10.0
   pip install tensorflow-directml-plugin
   ```

2. **If still fails, recreate virtual environment:**
   ```bash
   deactivate
   rmdir /s venv_tf2
   py -3.10 -m venv venv_tf2
   venv_tf2\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Issue 8: "No such file or directory: './dataset/r3d_train.txt'"

### Symptoms:
```
FileNotFoundError: [Errno 2] No such file or directory: './dataset/r3d_train.txt'
```

### Cause:
Training dataset not downloaded.

### Solution:

1. **Download dataset:**
   - See README.md for dataset links
   - R3D dataset: http://www.cs.toronto.edu/~fidler/projects/rent3D.html
   - R2V dataset: https://github.com/art-programmer/FloorplanTransformation

2. **Create TFRecord files:**
   - After downloading, use `utils/create_tfrecord.py` to convert to TFRecord format
   - Place in `./dataset/` folder

3. **For testing without dataset:**
   - Use `python demo.py --im_path=./demo/[image.jpg]` instead
   - This only requires a single image

---

## Issue 9: Multiple Python versions conflicting

### Symptoms:
- `python --version` shows different version than expected
- pip installs to wrong Python version

### Solution:

**Use Python Launcher to be explicit:**

```bash
# List all Python versions
py -0

# Use Python 3.10 specifically
py -3.10 -m venv venv_tf2
venv_tf2\Scripts\activate

# Verify
python --version  # Should be 3.10.x
where python      # Should point to venv_tf2
```

---

## Issue 10: PowerShell execution policy prevents activation

### Symptoms:
```
venv_tf2\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

### Solution:

**Option 1: Use Command Prompt instead**
```bash
venv_tf2\Scripts\activate.bat
```

**Option 2: Change PowerShell execution policy**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv_tf2\Scripts\Activate.ps1
```

---

## Still Having Issues?

1. **Check Python version:** `python --version` (MUST be 3.10.x)
2. **Check TensorFlow:** `python -c "import tensorflow as tf; print(tf.__version__)"`
3. **Check DirectML:** `python test_gpu.py`
4. **Check logs:** Look for error messages in console output
5. **Search GitHub issues:** https://github.com/microsoft/tensorflow-directml-plugin/issues

## Getting Help

If you're still stuck:
1. Document your Python version, TensorFlow version, and error message
2. Run `pip list` and save the output
3. Run `python test_gpu.py` and save the output
4. Create an issue with all this information
