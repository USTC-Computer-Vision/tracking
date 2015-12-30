project(qt_cmake_bgs_sof)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)

find_package(OpenCV REQUIRED)

if(${OpenCV_VERSION} VERSION_LESS 2.3.1)
  message (FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
endif()

# add c11
SET(CMAKE_CXX_FLAGS "-std=c++0x")
set( CMAKE_BUILD_TYPE Debug)

file(GLOB_RECURSE sources subsenseshrink.cpp ../../pl/*.cpp)
file(GLOB_RECURSE includes *.h ../../pl/*.h)

#list(REMOVE_ITEM sources main.cpp shrink.cpp)
#list(REMOVE_ITEM sources shrink.cpp)

add_library(yzbx SHARED ${sources})
target_link_libraries(yzbx ${OpenCV_LIBS})
set_property(TARGET yzbx PROPERTY PUBLIC_HEADER ${includes})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} yzbx)
#target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)

#add_executable(shrink shrink.cpp)
#target_link_libraries(shrink ${OpenCV_LIBS} yzbx)
