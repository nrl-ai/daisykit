# DaisyKit C++

**Daisykit SDK - C++** is the core of models and algorithms in NCNN deep learning framework. Using C++ code provides often provides the best performance for the algorithms.

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
