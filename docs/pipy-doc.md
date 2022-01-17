# DaisyKit Python

<https://pypi.org/project/daisykit/>

Deploy AI Systems Yourself (DAISY) Kit. DaisyKit Python is the wrapper of DaisyKit SDK, an AI framework focusing on the ease of deployment. At present, this package only has prebuilt distribution for Windows - Python 3. For other platform, you need to compile from source.

## How to install ?

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

## Examples

Read [Documentation](https://docs.daisykit.org/).

## Note for Python build

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

## Bug report

Please open an issue on our official repository if you find any error.

<https://github.com/Daisykit-AI/daisykit>
