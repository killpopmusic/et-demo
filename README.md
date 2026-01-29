# Eye tracking demo app - RPi version 
# Raspberry Pi 4B Setup Instructions

This branch is optimized for running on Raspberry Pi 4B (ARM64).

## Requirements
*   **Hardware:** Raspberry Pi 4B (4GB+ RAM recommended)
*   **OS:** Raspberry Pi OS "Bookworm" (64-bit)
*   **Python:** System Default Python 3.11 (Managed via Poetry)
*   Web camera

## 1. System Dependencies
Install the necessary system libraries for OpenCV, Math, and Audio.

```bash
sudo apt-get update
sudo apt-get install -y \
    libopencv-dev python3-opencv \
    libgl1-mesa-glx libgtk-3-dev \
    libopenblas-dev libopenblas0 \
    python3-dev build-essential
```

## 2. Environment Setup
Use the system's `python3-opencv` 

### Install Poetry
If not already installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### Configure and Install
1.  **Use System Python (3.11):**
    ```bash
    poetry env use /usr/bin/python3.11
    ```

2.  **Enable System Packages:**
    Allow the virtual environment to see the apt-installed `cv2`.
    ```bash
    VENV_PATH=$(poetry env info --path)
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/g' "$VENV_PATH/pyvenv.cfg"
    ```

3.  **Install Python Libraries:**
    ```bash
    poetry install
    ```

4.  **Fix Version Incompatibilities (Crucial):**
    Downgrade scientific libraries to match the system-level NumPy (1.24.2) and ARM architecture requirements.
    ```bash
    poetry run pip install "scikit-learn==1.3.2" "scipy==1.10.1" "numpy<1.25"
    ```

5.  **Link System OpenCV :**
    Since strict venv isolation hides system packages, link them manually:
    ```bash
    # Get venv site-packages path
    VENV_SITE=$(poetry run python -c "import site; print(site.getsitepackages()[0])")

    # Link cv2 from system
    # Note: Adjust the source path if your arch differs, but this is standard for RPi Bookworm ARM64
    ln -s /usr/lib/python3/dist-packages/cv2.cpython-311-aarch64-linux-gnu.so $VENV_SITE/cv2.so
    
    # Link system numpy (safe fallback)
    ln -s /usr/lib/python3/dist-packages/numpy $VENV_SITE/numpy
    ```

## 3. Running the Application
Ensure you are in the desktop environment (GUI required).

```bash
poetry run python src/main.py
```

## Troubleshooting
*   **Illegal Instruction:** Means `opencv-python` or `numpy` was installed via pip. Run `poetry run pip uninstall opencv-python opencv-contrib-python numpy` and re-do step 5.
*   **MediaPipe Error:** If `mediapipe` fails to install, ensure you are using Python 3.11.
