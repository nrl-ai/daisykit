# Build Python package for distribution
rm -rf dist/*
rm -rf python/*.egg-info
rm -rf python/setup.py
python3 setup.py sdist
