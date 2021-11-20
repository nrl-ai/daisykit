#!/bin/bash

set -e
set -x

PYTHON_VERSION=$1
BITNESS=$2

# if [[ "$BITNESS" == "32" ]]; then
#     # 32-bit architectures use the regular
#     # test command (outside of the minimal Docker container)
#     cp $CONFTEST_PATH $CONFTEST_NAME
#     python -c "import daisykit;"
#     pytest --pyargs daisykit
# else
#     docker container run \
#         --rm daisykit/minimal-windows \
#         powershell -Command "python -c 'import daisykit;'"

#     docker container run \
#         -e daisykit_SKIP_NETWORK_TESTS=1 \
#         -e OMP_NUM_THREADS=2 \
#         -e OPENBLAS_NUM_THREADS=2 \
#         --rm daisykit/minimal-windows \
#         powershell -Command "pytest --pyargs daisykit"
# fi
