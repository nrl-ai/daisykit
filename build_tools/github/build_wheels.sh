#!/bin/bash

set -e
set -x

# OpenMP is not present on macOS by default
if [[ "$RUNNER_OS" == "macOS" ]]; then
    wget https://packages.macports.org/libomp/libomp-16.0.3_0+universal.darwin_21.arm64-x86_64.tbz2 -O libomp.tbz2
    sudo tar -C / -xvjf libomp.tbz2 opt

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/opt/local/include/libomp"
    export CXXFLAGS="$CXXFLAGS -I/opt/local/include/libomp"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,/opt/local/lib/libomp -L/opt/local/lib/libomp -lomp"
fi

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
