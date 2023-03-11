#!/usr/bin/env bash

find src include python -type f -name '*.c' -o -name '*.cpp' -o -name '*.cc' -o -name '*.h' | grep -v python/pybind11 | grep -v python/pybind11_opencv_numpy  | xargs clang-format -i {}
