cmake_minimum_required(VERSION 3.10)

project(daisykitsdk)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_TOOLCHAIN_FILE)
    set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
    # Get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here.
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(NOT DEFINED DAISYKIT_VERSION)
    string(TIMESTAMP DAISYKIT_VERSION "%Y%m%d")
endif()

set(DAISYKIT_VERSION_MAJOR 0)
set(DAISYKIT_VERSION_MINOR 1)
set(DAISYKIT_VERSION_PATCH ${DAISYKIT_VERSION})
set(DAISYKIT_VERSION_STRING ${DAISYKIT_VERSION_MAJOR}.${DAISYKIT_VERSION_MINOR}.${DAISYKIT_VERSION_PATCH})
if(APPLE OR IOS)
    # macos / ios only accepts a.b.c.d.e where a=24bit b/c/d/e=10bit
    # 20201228 to 20.12.28
    string(SUBSTRING ${DAISYKIT_VERSION} 2 2 DAISYKIT_VERSION_YEAR)
    string(SUBSTRING ${DAISYKIT_VERSION} 4 2 DAISYKIT_VERSION_MONTH)
    string(SUBSTRING ${DAISYKIT_VERSION} 6 2 DAISYKIT_VERSION_DAY)
    set(DAISYKIT_VERSION_STRING ${DAISYKIT_VERSION_MAJOR}.${DAISYKIT_VERSION_MINOR}.${DAISYKIT_VERSION_YEAR}.${DAISYKIT_VERSION_MONTH}.${DAISYKIT_VERSION_DAY})
endif()
message(STATUS "DAISYKIT_VERSION_STRING = ${DAISYKIT_VERSION_STRING}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fexceptions -frtti -fPIC")

option(ncnn_FIND_PATH "Path to NCNN library" "~/Apps/ncnn-20210720-ubuntu-1804")
option(DAISYKIT_ENABLE_BARCODE_SCANNER "Enable barcode scanner" ON)
option(DAISYKIT_BUILD_EXAMPLES "Build examples" ON)
option(DAISYKIT_BUILD_PYTHON "Build Python packages" OFF)
option(DAISYKIT_BUILD_DOCS "Build documentation" OFF)
option(DAISYKIT_BUILD_SHARED_LIB "Build shared lib for dasiykitsdk. Set this value to OFF for static lib." ON)
option(DAISYKIT_WITH_VULKAN "Build with Vulkan" OFF)
option(DAISYKIT_COPY_ASSETS "Copy assets to bin folder" ON)

if(ANDROID)
MESSAGE(STATUS "COMPILE_ANDROID")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/thirdparties/opencv-mobile-4.5.3-android/sdk/native/jni)
    set(ncnn_FIND_PATH ${CMAKE_SOURCE_DIR}/thirdparties/ncnn-20210720-android-vulkan/${ANDROID_ABI})
else()
    if (DAISYKIT_WITH_VULKAN)
        find_package(Vulkan REQUIRED)
    endif()
    include_directories(${ncnn_FIND_PATH}/include/ncnn)
endif()

include_directories(include)

message(${ncnn_FIND_PATH}/lib/cmake/ncnn)
set(ncnn_DIR ${ncnn_FIND_PATH}/lib/cmake/ncnn)
find_package(ncnn)
if(NOT ncnn)
  message("Missing prebuilt ncnn. Building from source.")
  option(NCNN_BUILD_BENCHMARK "" OFF)
  option(NCNN_PYTHON "" OFF)
  option(NCNN_BUILD_TOOLS "" OFF)
  option(NCNN_BUILD_EXAMPLES "" OFF)
  option(NCNN_VULKAN "" ${DAISYKIT_WITH_VULKAN})
  add_subdirectory(src/thirdparties/ncnn ${CMAKE_BINARY_DIR}/bin_ncnn)
endif()

find_package(OpenCV REQUIRED)

set(sources
    src/models/ncnn_model.cpp
    src/models/image_model.cpp
    src/models/body_detector.cpp
    src/models/pose_detector.cpp
    src/models/action_classifier.cpp
    src/models/face_detector.cpp
    src/models/facial_landmark_estimator.cpp
    src/models/background_matting.cpp

    src/common/logging/mjpeg_writer.cpp
    src/common/visualizers/base_visualizer.cpp
    src/common/profiler.cpp
    src/common/utils/timer.cpp
    src/common/io/data_reader.cpp

    src/processors/signal_processors/signal_smoothing.cpp
    src/processors/signal_processors/z_score_filter.cpp
    src/processors/image_processors/img_utils.cpp
    src/processors/fitness/pushup_analyzer.cpp

    src/graphs/core/node.cpp
    src/graphs/core/connection.cpp
    src/graphs/core/graph.cpp
    src/graphs/core/packet.cpp
    src/graphs/core/transmission_profile.cpp

    src/flows/pushup_counter_flow.cpp
    src/flows/face_detector_flow.cpp
    src/flows/background_matting_flow.cpp

)

# Add platform specific source files
if (ANDROID)
set(sources ${sources} 
    src/common/io/android_assets_stream.cpp)
endif()

# Add barcode scanner
if (DAISYKIT_ENABLE_BARCODE_SCANNER)
    add_subdirectory(src/thirdparties/zxing-cpp)
    include_directories(src/thirdparties/zxing-cpp/core/src)
    include_directories(src/thirdparties/zxing-cpp/opencv/src)
    set(sources ${sources} 
        src/flows/barcode_scanner_flow.cpp)
endif()

if (DAISYKIT_BUILD_SHARED_LIB)
    add_library(daisykitsdk SHARED ${sources})
else()
    add_library(daisykitsdk STATIC ${sources})
endif()

target_link_libraries(daisykitsdk ncnn ${OpenCV_LIBS})

if (DAISYKIT_ENABLE_BARCODE_SCANNER)
    target_link_libraries(daisykitsdk ZXing::ZXing)
endif()


# ==================================================
# Build examples
# ==================================================
if ((NOT ANDROID) AND DAISYKIT_BUILD_EXAMPLES)
    add_subdirectory(src/examples)
endif()


# ==================================================
# Copy asset folders
# ==================================================
if (DAISYKIT_COPY_ASSETS)
    add_custom_target(configs ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/configs ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/configs)
    add_custom_target(models ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/models ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models)
    add_custom_target(images ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/images ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/images)
endif()

# ==================================================
# Build Python package
# ==================================================
if(DAISYKIT_BUILD_PYTHON)
    add_subdirectory(python)
endif()
