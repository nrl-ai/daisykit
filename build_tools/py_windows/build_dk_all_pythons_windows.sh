eval "$(conda shell.bash hook)"

conda activate dk36
python -m pip install --upgrade pip
python -m pip install numpy
python setup.py bdist_wheel

conda activate dk37
python -m pip install --upgrade pip
python -m pip install numpy
python setup.py bdist_wheel

conda activate dk38
python -m pip install --upgrade pip
python -m pip install numpy
python setup.py bdist_wheel

conda activate dk39
python -m pip install --upgrade pip
python -m pip install numpy
python setup.py bdist_wheel

conda activate dk310
python -m pip install --upgrade pip
python -m pip install numpy
python setup.py bdist_wheel

conda activate dk311
python -m pip install --upgrade pip
python -m pip install numpy
python setup.py bdist_wheel
