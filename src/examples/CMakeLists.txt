set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(demo_barcode_scanner demo_barcode_scanner.cpp)
target_link_libraries(demo_barcode_scanner ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_detector demo_face_detector.cpp)
target_link_libraries(demo_face_detector ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_background_matting demo_background_matting.cpp)
target_link_libraries(demo_background_matting ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_graph_api demo_graph_api.cpp)
target_link_libraries(demo_graph_api ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_detector_graph demo_face_detector_graph.cpp)
target_link_libraries(demo_face_detector_graph ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_movenet demo_movenet.cpp)
target_link_libraries(demo_movenet ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_hand_detector_yolox demo_hand_detector_yolox.cpp)
target_link_libraries(demo_hand_detector_yolox ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_hand_pose_detector demo_hand_pose_detector.cpp)
target_link_libraries(demo_hand_pose_detector ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_object_detector demo_object_detector_yolox.cpp)
target_link_libraries(demo_object_detector ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_detector_scrfd demo_face_detector_scrfd.cpp)
target_link_libraries(demo_face_detector_scrfd ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_sequential demo_face_sequential.cpp)
target_link_libraries(demo_face_sequential ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_face_liveness_detection demo_face_liveness_detection.cpp)
target_link_libraries(demo_face_liveness_detection ${OpenCV_LIBS} daisykitsdk)

# Copy OpenCV lib to run on Windows
if(WIN32)
    get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
    get_target_property(__dll_release opencv_world IMPORTED_LOCATION_RELEASE)
    add_custom_command(TARGET demo_pushup_classifier POST_BUILD # Adds a post-build event the TARGET
        COMMAND ${CMAKE_COMMAND} -E copy_if_different # which executes "cmake - E copy_if_different..."
        "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>" # <--this is in-file
        $<TARGET_FILE_DIR:demo_pushup_classifier> # <--this is out-file path

        # another dll copy if needed here
        COMMENT "Copy opencv_world")
endif()
