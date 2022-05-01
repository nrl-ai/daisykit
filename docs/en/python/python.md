# DaisyKit Python

<https://pypi.org/project/daisykit/>

Daisykit is an easy AI toolkit for software engineers to integrate pretrained AI models and pipelines into their projects. You DON'T need to be an AI engineer to build AI software. This open source project includes:

- **Daisykit SDK - C++**, the core of models and algorithms in NCNN deep learning framework.
- **Daisykit Python** wrapper for easy integration with Python.
- **Daisykit Android** - Example app demonstrate how to use Daisykit SDK in Android.

## 1. Installation

**For Windows:**

```
pip3 install daisykit
```

**For Ubuntu:**

- Install dependencies

```
sudo apt install pybind11-dev # Pybind11 - For Python/C++ Wrapper
sudo apt install libopencv-dev # For OpenCV
sudo apt install libvulkan-dev # Optional - For GPU support
```

- Install DaisyKit (compile from source)

```
pip3 install --upgrade pip # Ensure pip is updated
pip3 install daisykit
```

**For other platforms:**

- Install OpenCV, Pybind11 and Vulkan development package (if you want GPU support)

- Install DaisyKit (compile from source)

```
pip3 install --upgrade pip # Ensure pip is updated
pip3 install daisykit
```

## 2. Note for Python build

Current CD (continuous delivery) flow is partial, which means we only have prebuilt linux wheels for x86_64 and for Windows.

- Prebuilt wheels for linux x86_64 are built with Github actions.
- Windows wheels (64bit) are built manually on a local machine.
- macOS prebuilt wheels are not available for now. However, you can install dependencies (OpenCV, Vulkan) manually, then install Daisykit with pip command.

We will be happy if you can make a pull request to make the CD build fully automated. A good choice is using Github flow for all building tasks.

**Current steps for Windows build:**

```sh
bash ./build_tools/py_windows/build_dk_all_pythons_windows.sh
bash ./build_tools/py_windows/build_dk_python_source_dist.sh
bash ./build_tools/upload_pypi.sh
```
