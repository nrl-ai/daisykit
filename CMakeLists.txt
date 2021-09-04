cmake_minimum_required(VERSION 3.10)

project(daisykitsdk)

option(COMPILE_EXAMPLES "Compile Examples" ON)
option(ncnn_FIND_PATH "Path to NCNN library" "~/Apps/ncnn-20210720-ubuntu-1804")

if(ANDROID)
MESSAGE(STATUS "COMPILE_ANDROID")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/thirdparties/opencv-mobile-4.5.3-android/sdk/native/jni)
    set(ncnn_FIND_PATH ${CMAKE_SOURCE_DIR}/thirdparties/ncnn-20210720-android-vulkan/${ANDROID_ABI})
else()
    find_package(Vulkan REQUIRED)
    include_directories(${ncnn_FIND_PATH}/include/ncnn)
endif()


message(${ncnn_FIND_PATH}/lib/cmake/ncnn)
set(ncnn_DIR ${ncnn_FIND_PATH}/lib/cmake/ncnn)
find_package(ncnn REQUIRED ${ncnn_DIR})
find_package(OpenCV REQUIRED)

include_directories(include)

add_library(daisykitsdk SHARED
    src/models/body_detector.cpp
    src/models/pose_detector.cpp
    src/models/action_classifier.cpp

    src/utils/logging/mjpeg_writer.cpp
    src/utils/signal_proc/signal_smoothing.cpp
    src/utils/signal_proc/z_score_filter.cpp
    src/utils/visualizer/viz_utils.cpp

    src/examples/fitness/pushup_analyzer.cpp
)
target_link_libraries(daisykitsdk ncnn ${OpenCV_LIBS})


if ((NOT ANDROID) AND COMPILE_EXAMPLES)

add_executable(demo_pushup_classifier src/examples/fitness/demo_pushup_classifier.cpp)
target_link_libraries(demo_pushup_classifier ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_pushup_full src/examples/fitness/demo_pushup_full.cpp)
target_link_libraries(demo_pushup_full ${OpenCV_LIBS} daisykitsdk)

endif()