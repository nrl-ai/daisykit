import io
import os
import sys
import re
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


# Initialize submodules
if os.path.exists(".git"):
    import pip._internal.vcs.git as git
    g = git.Git()  # NOTE: pip API's are internal, this has to be refactored
    g.run_command(["submodule", "sync"])
    g.run_command(
        ["submodule", "update", "--init", "--recursive"]
    )

# Obtain the numpy include directory.
# This logic works across numpy versions.
numpy_available = False
numpy_include = ""
try:
    import numpy as np
    from numpy.distutils.system_info import get_info

    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()

    numpy_available = True
except ImportError as e:
    print("Numpy was not installed!")


def find_version():
    with io.open("CMakeLists.txt", encoding="utf8") as f:
        version_file = f.read()

    version_major = re.findall(r"DAISYKIT_VERSION_MAJOR ((\.|\d)+)", version_file)
    version_minor = re.findall(r"DAISYKIT_VERSION_MINOR ((\.|\d)+)", version_file)
    version_patch = None
    if sys.platform == "darwin":
        version_patch = re.findall(r"DAISYKIT_VERSION_PATCH_DARWIN ((\.|\d)+)", version_file)
    else:
        version_patch = re.findall(r"DAISYKIT_VERSION_PATCH_OTHERS ((\.|\d)+)", version_file)

    if version_major and version_minor and version_patch:
        return version_major[0][0] + "." + version_minor[0][0] + "." + version_patch[0][0]
    raise RuntimeError("Unable to find version string.")


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        extdir = os.path.join(extdir, "daisykit")

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
            "-DDAISYKIT_BUILD_PYTHON=ON",
            "-DDAISYKIT_BUILD_EXAMPLES=OFF",
            "-DDAISYKIT_BUILD_DOCS=OFF",
            "-DDAISYKIT_COPY_ASSETS=OFF",
            "-DDAISYKIT_BUILD_SHARED_LIB=OFF",
            "-DDAISYKIT_WITH_VULKAN=OFF"
        ]
        if numpy_available:
            cmake_args.append("-DNUMPY_INCLUDE_DIR={}".format(numpy_include))

        build_args = []

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                cmake_args += ["-GNinja"]
        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


if sys.version_info < (3, 0):
    sys.exit("Sorry, Python < 3.0 is not supported")

# Newer versions of opencv don't have wheels for i686
requirements = ["numpy", "tqdm", "requests", "portalocker", "opencv-python<=4.5.1.48"]

with io.open("docs/python.md", encoding="utf-8") as h:
    long_description = h.read()

setup(
    name="daisykit",
    version=find_version(),
    author="DaisyKit Team",
    author_email="daisykit.team@gmail.com",
    maintainer="DaisyKit Team",
    maintainer_email="daisykit.team@gmail.com",
    description="Deploy AI Systems Yourself (DAISY) Kit. DaisyKit Python is the wrapper of DaisyKit SDK, an AI framework focusing on the ease of deployment.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://docs.daisykit.org/",
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache License 2.0",
    python_requires=">=3.6",
    packages=find_packages("python"),
    package_dir={"": "python"},
    install_requires=requirements,
    ext_modules=[CMakeExtension("daisykit")],
    cmdclass={"build_ext": CMakeBuild},
)
