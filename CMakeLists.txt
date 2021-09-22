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
    src/models/base_model.cpp
    src/models/base_model-impl.cpp
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
if (ENABLE_BARCODE_SCANNER)
add_subdirectory(src/thirdparties/zxing-cpp)
include_directories(src/thirdparties/zxing-cpp/core/src)
include_directories(src/thirdparties/zxing-cpp/opencv/src)
set(sources ${sources} 
    src/flows/barcode_scanner_flow.cpp)
endif()

add_library(daisykitsdk SHARED ${sources})
target_link_libraries(daisykitsdk ncnn ${OpenCV_LIBS})

if (ENABLE_BARCODE_SCANNER)
target_link_libraries(daisykitsdk ZXing::ZXing)
endif()


# ==================================================
# Examples
# ==================================================
if ((NOT ANDROID) AND COMPILE_EXAMPLES)

add_executable(demo_pushup_classifier src/examples/fitness/demo_pushup_classifier.cpp)
target_link_libraries(demo_pushup_classifier ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_pushup_full src/examples/fitness/demo_pushup_full.cpp)
target_link_libraries(demo_pushup_full ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_barcode_scanner src/examples/demo_barcode_scanner.cpp)
target_link_libraries(demo_barcode_scanner ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_detector src/examples/demo_face_detector.cpp)
target_link_libraries(demo_face_detector ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_background_matting src/examples/demo_background_matting.cpp)
target_link_libraries(demo_background_matting ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_graph_api src/examples/demo_graph_api.cpp)
target_link_libraries(demo_graph_api ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_detector_graph src/examples/demo_face_detector_graph.cpp)
target_link_libraries(demo_face_detector_graph ${OpenCV_LIBS} daisykitsdk)

endif()


# ==================================================
# Copy asset folders
# ==================================================
add_custom_target(configs ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/configs ${CMAKE_BINARY_DIR}/configs)
add_custom_target(models ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/models ${CMAKE_BINARY_DIR}/models)
add_custom_target(images ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets/images ${CMAKE_BINARY_DIR}/images)


if(BUILD_PYTHON)
    add_subdirectory(python)
endif()


# ==================================================
# Doxygen - For documentation
# ==================================================
# Look for Doxygen package
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
