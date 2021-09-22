# DaisyKit Python

Python bindings for DaisyKit.

## Build Python package

Build environment: Ubuntu.

```
sudo apt install ninja-build
python3 -m pip install --user --upgrade twine
```

Build package:

```
python3 setup.py sdist
```

Upload to Pypi

```
twine upload dist/*
```

## TODO

- Multiplatform build.
