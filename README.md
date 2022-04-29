# Daisykit - **D.A.I.S.Y: Deploy AI Systems Yourself!**

[Daisykit](https://daisykit.org/>) is an easy AI toolkit for software engineers to integrate pretrained AI models and pipelines into their projects. You DON'T need to be an AI engineer to build AI software. This open source project includes:

- **Daisykit SDK - C++**, the core of models and algorithms in NCNN deep learning framework.
- **Daisykit Python** wrapper for easy integration with Python.
- **Daisykit Android** - Example app demonstrate how to use Daisykit SDK in Android.

**Links:**

- **Python Package:** <https://pypi.org/project/daisykit/> [![Pypi Downloads](https://pepy.tech/badge/daisykit/month)](https://pypi.org/project/daisykit/)
- **Documentation:** <https://daisykit.org/>


https://user-images.githubusercontent.com/18329471/143721185-d4d095dd-48b8-481c-81c7-904c1536a6b8.mp4

**Demo Video:** <https://www.youtube.com/watch?v=zKP8sgGoFMc>.


## Environment Setup

For Ubuntu, we need build tools from `build-essential` package. For Windows, Visual Studio 2019 is recommended.

- Install OpenCV.

**Ubuntu:**

```
sudo apt install libopencv-dev
```

**Windows:**

Download and extract OpenCV from [the official website](https://opencv.org/releases/), and add `OpenCV_DIR` to path.

- Install Vulkan development package.

**Ubuntu:**

```
sudo apt install -y libvulkan-dev vulkan-utils
sudo apt install mesa-vulkan-drivers # For Intel GPU support
```

- Download [precompiled NCNN](https://github.com/Tencent/ncnn/releases), extract it (version for your development computer).

## Build and Run on PC

- Initialize / Update submodules

```
git submodule update --init
```

- Build

**Ubuntu:**

```
mkdir build
cd build
cmake .. -Dncnn_FIND_PATH="<path to ncnn lib>"
make
```

**Windows:**

```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -Dncnn_FIND_PATH="<path to ncnn lib>" ..
cmake --build . --config Release
```

- Run face detection example

**Ubuntu:**

```
./bin/demo_face_detector_graph
```

**Windows:**

```
./bin/Release/demo_face_detector_graph
```

## Coding convention

Read coding convention and contribution guidelines [here](https://docs.daisykit.org/md_contribution.html).

## Build documentation

- Step 1: Install **doxygen** first.

- Step 2: Build the documentation:

```
cd docs
doxygen Doxyfile.in
```

- Step 3: Deploy html documentation from `docs/_build/html`.

- Step 4: Our lastest documentation is deployed at <https://docs.daisykit.org>.

## Known issues and problems

**1. Slow model inference - Low FPS**

This issue can happen on development build. Add `-DCMAKE_BUILD_TYPE=Debug` to `cmake` command and build again. The FPS can be much better.

## References

This toolkit is developed on top of other source code. Including

- Toolchains setup from [ncnn](https://github.com/Tencent/ncnn).
- QR Scanner from [ZXing-CPP](https://github.com/nu-book/zxing-cpp).
- JSON support from [nlohmann/json](https://github.com/nlohmann/json).
- Pretrained AI models from different sources: <https://docs.daisykit.org/md_models.html>.
