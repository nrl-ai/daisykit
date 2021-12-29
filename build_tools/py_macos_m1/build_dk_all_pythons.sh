eval "$(conda shell.bash hook)"

brew install ninja

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
