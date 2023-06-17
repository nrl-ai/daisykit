eval "$(conda shell.bash hook)"

brew install ninja
brew install opencv
brew install libomp

export OpenCV_DIR="/opt/homebrew/Cellar/opencv/4.7.0_2/"

export MACOSX_DEPLOYMENT_TARGET=13.0

rm -rf build
conda activate dk38
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk39
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk310
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk311
pip install numpy
arch -arm64 python setup.py bdist_wheel

export MACOSX_DEPLOYMENT_TARGET=12.0

rm -rf build
conda activate dk38
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk39
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk310
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk311
pip install numpy
arch -arm64 python setup.py bdist_wheel

export MACOSX_DEPLOYMENT_TARGET=11.0

rm -rf build
conda activate dk38
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk39
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk310
pip install numpy
arch -arm64 python setup.py bdist_wheel

rm -rf build
conda activate dk311
pip install numpy
arch -arm64 python setup.py bdist_wheel
