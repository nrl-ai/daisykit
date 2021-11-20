#!/bin/bash

set -e
set -x

cd ../../

python -m venv test_env
source test_env/bin/activate

# Skip testing for src dist
# python -m pip install daisykit/daisykit/dist/*.tar.gz
# pytest --pyargs daisykit
