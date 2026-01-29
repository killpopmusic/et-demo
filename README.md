# Eye tracking demo app
This is a project demonstrating the performance of face-landmark-only trained models in eye tracking classification task. The repository has been set up to enable usage on different hardware.

- The **main** branch assumes a device with Nvidia GPU, since it uses CUDA to accelerate model's inference
- The **laptop** branch is suited for systems without dedicated Nvidia GPU
- The app has been also proven to work on Raspberry Pi 4b, to test it make sure to follow the instructions placed in a README_RPI file.

## Requirements 
- Ubuntu 22.04 or newer
- Web camera

## Setup 
The project uses poetry dependency managment, to install:
```
curl -sSL https://install.python-poetry.org | python -
```
To install dependencies and create the environment: 

```
poetry install
```

Then run it using: 

```
poetry shell
```

To run the application:

```
poetry run python3  src/main.py
```
