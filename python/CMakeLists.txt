cmake_minimum_required(VERSION 3.4)

project(pydaisykit)

set(PACKAGE_VERSION ${DAISYKIT_VERSION_STRING})
add_definitions(-DVERSION_INFO="${PACKAGE_VERSION}")

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

add_subdirectory(pybind11)
include_directories(pybind11_opencv_numpy)

if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    if(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/daisykit/)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/daisykit/)
    endif(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
pybind11_add_module(pydaisykit src/main.cpp pybind11_opencv_numpy/ndarray_converter.cpp)
set_target_properties(pydaisykit PROPERTIES OUTPUT_NAME "daisykit")
target_link_libraries(pydaisykit PUBLIC daisykitsdk)
set_target_properties(pydaisykit PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/daisykit")
set_property(TARGET pydaisykit PROPERTY FOLDER "python")
if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    add_custom_command(TARGET pydaisykit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/daisykit/daisykit${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}
        ${PROJECT_SOURCE_DIR}/daisykit/daisykit${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION})
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

if (WIN32)
# Copy OpenCV library along with daisykit
get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)
add_custom_command(TARGET pydaisykit POST_BUILD        # Adds a post-build event the TARGET
    COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
    "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
    $<TARGET_FILE_DIR:pydaisykit>                        # <--this is out-file path
        # another dll copy if needed here
    COMMENT "Copy opencv_world")
endif()

configure_file(setup.py.i ${PROJECT_SOURCE_DIR}/setup.py)
