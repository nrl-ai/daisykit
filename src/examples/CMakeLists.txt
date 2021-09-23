add_executable(demo_pushup_classifier fitness/demo_pushup_classifier.cpp)
target_link_libraries(demo_pushup_classifier ${OpenCV_LIBS} daisykitsdk)

add_executable(demo_pushup_full fitness/demo_pushup_full.cpp)
target_link_libraries(demo_pushup_full ${OpenCV_LIBS} daisykitsdk)

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
