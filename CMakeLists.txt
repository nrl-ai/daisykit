cmake_minimum_required(VERSION 3.10.2)

project(DaisyKit)

# Use Release build type if not specified
if(NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()

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
set(DAISYKIT_VERSION_MINOR 3)
set(DAISYKIT_VERSION_PATCH 0)
set(DAISYKIT_VERSION_BUILD 0)

set(DAISYKIT_VERSION_STRING ${DAISYKIT_VERSION_MAJOR}.${DAISYKIT_VERSION_MINOR}.${DAISYKIT_VERSION_PATCH}.${DAISYKIT_VERSION_BUILD})
message(STATUS "DAISYKIT_VERSION_STRING = ${DAISYKIT_VERSION_STRING}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

option(ncnn_FIND_PATH "Path to NCNN library" "~/Apps/ncnn-20210720-ubuntu-1804")
option(DAISYKIT_ENABLE_BARCODE_SCANNER "Enable barcode scanner" ON)
option(DAISYKIT_BUILD_EXAMPLES "Build examples" ON)
option(DAISYKIT_BUILD_PYTHON "Build Python packages" OFF)
option(DAISYKIT_BUILD_DOCS "Build documentation" OFF)
option(DAISYKIT_BUILD_SHARED_LIB "Build shared lib for dasiykitsdk. Set this value to OFF for static lib." OFF)
option(DAISYKIT_WITH_VULKAN "Build with Vulkan" ON)
option(DAISYKIT_COPY_ASSETS "Copy assets to bin folder" ON)
option(DAISYKIT_ENABLE_EXCEPTIONS "Enable C++ exceptions feature" ON)
option(DAISYKIT_ENABLE_RTTI "Enable C++ RTTI feature" ON)

if(DAISYKIT_ENABLE_EXCEPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
endif()

if(DAISYKIT_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
endif()

include_directories(include .)

if(ANDROID)
    MESSAGE(STATUS "COMPILE_ANDROID")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/opencv-mobile-4.5.4-android/sdk/native/jni)
    set(ncnn_DIR ${CMAKE_SOURCE_DIR}/third_party/ncnn-20211208-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
    find_package(ncnn REQUIRED)
else()
    if(DAISYKIT_WITH_VULKAN)
        find_package(Vulkan)

        # If not found Vulkan, disable GPU support
        if(NOT Vulkan_FOUND)
            message("Vulkan is not found. Disabling GPU support.")
            set(DAISYKIT_WITH_VULKAN OFF)
        endif()

        option(NCNN_VULKAN "" ${DAISYKIT_WITH_VULKAN})
    endif()

    if(DAISYKIT_WITH_VULKAN)
        add_compile_definitions(DAISYKIT_WITH_VULKAN)
    endif()

    include_directories(${ncnn_FIND_PATH}/include/ncnn)

    message(${ncnn_FIND_PATH}/lib/cmake/ncnn)
    set(ncnn_DIR ${ncnn_FIND_PATH}/lib/cmake/ncnn)
    find_package(ncnn)
endif()

if(NOT ${ncnn_FOUND})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt")
        message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
    endif()

    message("Missing prebuilt ncnn. Building from source.")
    option(NCNN_INSTALL_SDK "" OFF)
    option(NCNN_VULKAN "" ${DAISYKIT_WITH_VULKAN})
    option(NCNN_VULKAN_ONLINE_SPIRV "" ${DAISYKIT_WITH_VULKAN})
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)

    if(DAISYKIT_ENABLE_EXCEPTIONS)
        option(NCNN_DISABLE_EXCEPTION "" OFF)
    else()
        option(NCNN_DISABLE_EXCEPTION "" ON)
    endif()

    if(DAISYKIT_ENABLE_RTTI)
        option(NCNN_DISABLE_RTTI "" OFF)
    else()
        option(NCNN_DISABLE_RTTI "" ON)
    endif()

    add_subdirectory(third_party/ncnn ${CMAKE_BINARY_DIR}/bin_ncnn)
    include_directories(${CMAKE_BINARY_DIR}/bin_ncnn/src)
    include_directories(third_party/ncnn/src)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(third_party/hnswlib)

set(sources
    src/models/ncnn_model.cpp
    src/models/image_model.cpp
    src/models/body_detector.cpp
    src/models/pose_detector.cpp
    src/models/pose_detector_movenet.cpp
    src/models/face_detector.cpp
    src/models/facial_landmark_detector.cpp
    src/models/background_matting.cpp
    src/models/yolox_utils.cpp
    src/models/hand_detector_yolox.cpp
    src/models/hand_pose_detector.cpp
    src/models/object_detector_yolox.cpp
    src/models/face_alignment.cpp
    src/models/face_extractor.cpp
    src/models/face_detector_scrfd.cpp
    src/models/face_manager.cpp
    src/models/face_liveness_detector.cpp

    src/common/visualizers/base_visualizer.cpp
    src/common/profiler.cpp
    src/common/utils/timer.cpp
    src/common/io/data_reader.cpp

    src/processors/signal_processors/signal_smoothing.cpp
    src/processors/signal_processors/z_score_filter.cpp
    src/processors/image_processors/img_utils.cpp

    src/graphs/core/node.cpp
    src/graphs/core/connection.cpp
    src/graphs/core/graph.cpp
    src/graphs/core/packet.cpp
    src/graphs/core/transmission_profile.cpp

    src/flows/object_detector_flow.cpp
    src/flows/face_detector_flow.cpp
    src/flows/background_matting_flow.cpp
    src/flows/human_pose_movenet_flow.cpp
    src/flows/hand_pose_detector_flow.cpp
)

# Add platform specific source files
if(ANDROID)
    set(sources ${sources}
        src/common/io/android_assets_stream.cpp)
endif()

# Add barcode scanner
if(DAISYKIT_ENABLE_BARCODE_SCANNER)
    add_subdirectory(third_party/zxing-cpp)
    include_directories(third_party/zxing-cpp/src)
    set(sources ${sources} src/flows/barcode_scanner_flow.cpp)
endif()

if(DAISYKIT_BUILD_SHARED_LIB)
    add_library(daisykitsdk SHARED ${sources})
else()
    add_library(daisykitsdk STATIC ${sources})
endif()

target_link_libraries(daisykitsdk ncnn ${OpenCV_LIBS})

if(DAISYKIT_ENABLE_BARCODE_SCANNER)
    target_link_libraries(daisykitsdk ZXing::ZXing)
endif()

if(WIN32)
    # Copy OpenCV library along with daisykit
    get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
    get_target_property(__dll_release opencv_world IMPORTED_LOCATION_RELEASE)
    add_custom_command(TARGET daisykitsdk POST_BUILD # Adds a post-build event the TARGET
        COMMAND ${CMAKE_COMMAND} -E copy_if_different # which executes "cmake - E copy_if_different..."
        "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>" # <--this is in-file
        $<TARGET_FILE_DIR:daisykitsdk> # <--this is out-file path

        # another dll copy if needed here
        COMMENT "Copy opencv_world")
endif()

# ==================================================
# Build examples
# ==================================================
if((NOT ANDROID) AND DAISYKIT_BUILD_EXAMPLES)
    add_subdirectory(src/examples)
endif()

# ==================================================
# Copy asset folders
# ==================================================
if(DAISYKIT_COPY_ASSETS)
    add_custom_target(configs ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/configs ${CMAKE_BINARY_DIR}/configs)
    add_custom_target(models ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/models ${CMAKE_BINARY_DIR}/models)
    add_custom_target(images ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/images ${CMAKE_BINARY_DIR}/images)
endif()

# ==================================================
# Build Python package
# ==================================================
if(DAISYKIT_BUILD_PYTHON)
    add_subdirectory(python)
endif()
