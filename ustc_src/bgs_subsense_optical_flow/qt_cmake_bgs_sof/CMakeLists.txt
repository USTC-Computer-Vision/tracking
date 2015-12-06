project(qt_cmake_bgs_sof)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)

find_package(OpenCV REQUIRED)

if(${OpenCV_VERSION} VERSION_LESS 2.3.1)
  message (FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
endif()

file(GLOB_RECURSE sources *.cpp)
file(GLOB_RECURSE includes *.h)

list(REMOVE_ITEM sources main.cpp)

add_library(yzbx SHARED ${sources})
target_link_libraries(yzbx ${OpenCV_LIBS})
set_property(TARGET yzbx PROPERTY PUBLIC_HEADER ${includes})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} yzbx)
