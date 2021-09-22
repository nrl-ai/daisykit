cmake_minimum_required(VERSION 3.4)

project(pydaisykit)

set(PACKAGE_VERSION ${DAISYKIT_VERSION_STRING})
add_definitions(-DVERSION_INFO="${PACKAGE_VERSION}")

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

add_subdirectory(pybind11)
add_subdirectory(pybind11_opencv_numpy)
include_directories(pybind11_opencv_numpy)

if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    if(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/daisykit/)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/daisykit/)
    endif(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
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

configure_file(setup.py.i ${PROJECT_SOURCE_DIR}/setup.py)
