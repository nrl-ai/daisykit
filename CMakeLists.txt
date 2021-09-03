cmake_minimum_required(VERSION 3.10)

project(daisykitsdk)

option(COMPILE_EXAMPLES "Compile Examples" ON)
set(ncnn_FIND_PATH "/home/vietanhdev/Apps/ncnn-20210720-ubuntu-1804")

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

include_directories("." "..")

add_library(daisykitsdk SHARED
    analyzers/pushup_analyzer.cpp

    ai_models/model_manager.cpp
    ai_models/body_detector.cpp
    ai_models/pose_detector.cpp
    ai_models/pose_detector_blazepose.cpp
    ai_models/action_classifier.cpp

    logging/mjpeg_writer.cpp

    signalproc/signal_processor.cpp
)
target_link_libraries(daisykitsdk ncnn ${OpenCV_LIBS})


if ((NOT ANDROID) AND COMPILE_EXAMPLES)

add_executable(demo_pushup_classifier examples/demo_pushup_classifier.cpp)
target_link_libraries(demo_pushup_classifier ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_pushup_full examples/demo_pushup_full.cpp)
target_link_libraries(demo_pushup_full ${OpenCV_LIBS} daisykitsdk)

# Not yet finished
add_executable(demo_blazepose examples/demo_blazepose.cpp)
target_link_libraries(demo_blazepose ${OpenCV_LIBS} daisykitsdk)

endif()