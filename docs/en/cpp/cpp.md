# DaisyKit C++

**Daisykit SDK - C++** is the core of models and algorithms in NCNN deep learning framework. Using C++ code provides often provides the best performance for the algorithms.

## 1. Environment Setup

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

## 2. Build and Run on PC

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

## 3. C++ Coding convention

Read coding convention and contribution guidelines [here](/en/latest/contribution.html).
