eval "$(conda shell.bash hook)"
conda activate py35 && pip install numpy && python setup.py bdist_wheel
conda activate py36 && pip install numpy && python setup.py bdist_wheel
conda activate dk37 && pip install numpy && python setup.py bdist_wheel
conda activate dk38 && pip install numpy && python setup.py bdist_wheel
conda activate dk39 && pip install numpy && python setup.py bdist_wheel
conda activate dk310 && pip install numpy && python setup.py bdist_wheel
