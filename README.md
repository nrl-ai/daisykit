# DaisyKit - **D.A.I.S.Y: Deploy AI Systems Yourself!**

[![PyPI](https://img.shields.io/pypi/v/daisykit)](https://pypi.org/project/daisykit)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://daisykit.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/nrl-ai/daisykit.svg)](https://github.com/nrl-ai/daisykit/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/nrl-ai/daisykit.svg)](https://github.com/nrl-ai/daisykit/issues)
[![Pypi Downloads](https://pepy.tech/badge/daisykit/month)](https://pypi.org/project/daisykit/)

<a href="https://www.producthunt.com/posts/daisykit?utm_source=badge-featured&utm_medium=badge&utm_souce=badge-daisykit" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=385715&theme=light" alt="Daisykit - A&#0032;library&#0032;for&#0032;building&#0032;AI&#0032;applications&#0032;without&#0032;AI&#0032;knowledge | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

[DaisyKit](https://daisykit.nrl.ai) is an easy AI toolkit with face mask detection, pose detection, background matting, barcode detection and more. This open source project includes:

- **DaisyKit SDK - C++**, the core of models and algorithms in NCNN deep learning framework.
- **DaisyKit Python** wrapper for easy integration with Python.
- **DaisyKit Android** - Example app demonstrate how to use Daisykit SDK in Android.

**Links:**

- **Python Package:** [https://pypi.org/project/daisykit/](https://pypi.org/project/daisykit/)
- **Documentation:** [https://daisykit.nrl.ai/docs](https://daisykit.nrl.ai/docs)

<a href="https://www.youtube.com/watch?v=zKP8sgGoFMc">
<img src="https://user-images.githubusercontent.com/18329471/224732779-3d2ce7b0-de53-4b9d-b9a8-890557eead16.png"/>
</a>

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
git clone https://github.com/nrl-ai/daisykit.git --recursive
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

Read coding convention and contribution guidelines [here](https://daisykit.nrl.ai/docs/contribution).

## 4. Known issues and problems

- **Slow model inference - Low FPS**

This issue can happen on development build. Add `-DCMAKE_BUILD_TYPE=Debug` to `cmake` command and build again. The FPS can be much better.

## 5. References

This toolkit is developed on top of other source code. Including

- Toolchains setup from [ncnn](https://github.com/Tencent/ncnn).
- QR Scanner from [ZXing-CPP](https://github.com/nu-book/zxing-cpp).
- JSON support from [nlohmann/json](https://github.com/nlohmann/json).
- Pretrained AI models from different sources: <https://daisykit.nrl.ai/docs/models>.
