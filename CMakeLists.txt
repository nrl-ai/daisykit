cmake_minimum_required(VERSION 3.10)

project(daisykitsdk)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fexceptions -frtti -fPIC")

option(ENABLE_BARCODE_SCANNER "Enable barcode scanner" ON)
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

include_directories(include)

message(${ncnn_FIND_PATH}/lib/cmake/ncnn)
set(ncnn_DIR ${ncnn_FIND_PATH}/lib/cmake/ncnn)
find_package(ncnn REQUIRED ${ncnn_DIR})
find_package(OpenCV REQUIRED)

set(sources
    src/models/body_detector.cpp
    src/models/pose_detector.cpp
    src/models/action_classifier.cpp
    src/models/face_detector.cpp
    src/models/face_detector_with_mask.cpp
    src/models/facial_landmark_estimator.cpp
    src/models/human_matting.cpp

    src/utils/logging/mjpeg_writer.cpp
    src/utils/signal_proc/signal_smoothing.cpp
    src/utils/signal_proc/z_score_filter.cpp
    src/utils/img_proc/img_utils.cpp
    src/utils/visualizer/viz_utils.cpp

    src/flows/pushup_counter_flow.cpp
    src/flows/face_detector_flow.cpp
    src/flows/face_detector_with_mask_flow.cpp
    src/flows/human_matting_flow.cpp

    src/examples/fitness/pushup_analyzer.cpp
)

if (ENABLE_BARCODE_SCANNER)
add_subdirectory(src/thirdparties/zxing-cpp)
include_directories(src/thirdparties/zxing-cpp/core/src)
include_directories(src/thirdparties/zxing-cpp/opencv/src)
set(sources ${sources} 
    src/flows/barcode_scanner_flow.cpp)
endif()

add_library(daisykitsdk SHARED ${sources})
target_link_libraries(daisykitsdk ncnn ${OpenCV_LIBS} ZXing::ZXing)


if ((NOT ANDROID) AND COMPILE_EXAMPLES)

add_executable(demo_pushup_classifier src/examples/fitness/demo_pushup_classifier.cpp)
target_link_libraries(demo_pushup_classifier ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_pushup_full src/examples/fitness/demo_pushup_full.cpp)
target_link_libraries(demo_pushup_full ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_barcode_scanner src/examples/demo_barcode_scanner.cpp)
target_link_libraries(demo_barcode_scanner ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_detector src/examples/demo_face_detector.cpp)
target_link_libraries(demo_face_detector ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_detector_with_mask src/examples/demo_face_detector_with_mask.cpp)
target_link_libraries(demo_face_detector_with_mask ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_human_matting src/examples/demo_human_matting.cpp)
target_link_libraries(demo_human_matting ${OpenCV_LIBS} daisykitsdk)

endif()


# Copy asset folders
add_custom_target(configs ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/configs ${CMAKE_BINARY_DIR}/configs)
add_custom_target(models ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/models ${CMAKE_BINARY_DIR}/models)
add_custom_target(images ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/images ${CMAKE_BINARY_DIR}/images)