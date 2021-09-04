# Daisykit SDK

AI Kit for mobile and embedded devices.
## Environment Setup

- Install OpenCV. In Ubuntu:

```
sudo apt install libopencv-dev
```

- Install Vulkan dev. In Ubuntu:

```
sudo apt-get install -y libvulkan-dev
```

- Download [precompiled NCNN](https://github.com/Tencent/ncnn/releases), extract it.

- Download [pretrained models](https://drive.google.com/drive/folders/1O4bT6somFeBFc23BFKH-0on7E3LAkC-b?usp=sharing) and put into `data/models`.

## Build and Run on PC

```
mkdir build
cd build
cmake .. -D ncnn_FIND_PATH="<path to ncnn lib>"
make
```

- Run Pushups example

```
./demo_pushup_full
```
