# Daisykit - **D.A.I.S.Y: Deploy AI Systems Yourself!**

[![PyPI](https://img.shields.io/pypi/v/daisykit)](https://pypi.org/project/daisykit)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://daisykit.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/DaisyLabSolutions/daisykit.svg)](https://github.com/DaisyLabSolutions/daisykit/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/DaisyLabSolutions/daisykit.svg)](https://github.com/DaisyLabSolutions/daisykit/issues)
[![Pypi Downloads](https://pepy.tech/badge/daisykit/month)](https://pypi.org/project/daisykit/)

[Daisykit](https://daisykit.org/>) is an easy AI toolkit for software engineers to integrate pretrained AI models and pipelines into their projects. You DON'T need to be an AI engineer to build AI software. This open source project includes:

- **Daisykit SDK - C++**, the core of models and algorithms in NCNN deep learning framework.
- **Daisykit Python** wrapper for easy integration with Python.
- **Daisykit Android** - Example app demonstrate how to use Daisykit SDK in Android.

**Links:**

- **Python Package:** <https://pypi.org/project/daisykit/>
- **Documentation:** <https://daisykit.org/>


https://user-images.githubusercontent.com/18329471/143721185-d4d095dd-48b8-481c-81c7-904c1536a6b8.mp4

**Demo Video:** <https://www.youtube.com/watch?v=zKP8sgGoFMc>.


## 1. Environment Setup

### Ubuntu

Install packages from Terminal

```
sudo apt install -y build-essential libopencv-dev
sudo apt install -y libvulkan-dev vulkan-utils
sudo apt install -y mesa-vulkan-drivers # For Intel GPU support
```

### Windows

For Windows, Visual Studio 2019 + Git Bash is recommended.

- Download and extract OpenCV from [the official website](https://opencv.org/releases/), and add `OpenCV_DIR` to path.
- Download [precompiled NCNN](https://github.com/Tencent/ncnn/releases).

## 2. Build and run C++ examples

Clone the source code:

```
git clone https://github.com/DaisyLabSolutions/daisykit.git --recursive
cd daisykit
```
### Ubuntu

Build Daisykit:


```
mkdir build
cd build
cmake .. -Dncnn_FIND_PATH="<path to ncnn lib>"
make
```

Run face detection example:

```
./bin/demo_face_detector_graph
```

If you dont specify `ncnn_FIND_PATH`, NCNN will be built from scratch.

### Windows

Build Daisykit:


```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -Dncnn_FIND_PATH="<path to ncnn lib>" ..
cmake --build . --config Release
```

Run face detection example:

```
./bin/Release/demo_face_detector_graph
```

## 3. C++ Coding convention

Read coding convention and contribution guidelines [here](/en/latest/contribution.html).

## 4. Known issues and problems

- **Slow model inference - Low FPS**

This issue can happen on development build. Add `-DCMAKE_BUILD_TYPE=Debug` to `cmake` command and build again. The FPS can be much better.

## 5. References

This toolkit is developed on top of other source code. Including

- Toolchains setup from [ncnn](https://github.com/Tencent/ncnn).
- QR Scanner from [ZXing-CPP](https://github.com/nu-book/zxing-cpp).
- JSON support from [nlohmann/json](https://github.com/nlohmann/json).
- Pretrained AI models from different sources: <https://docs.daisykit.org/en/latest/models.html>.
