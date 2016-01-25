cmake_minimum_required(VERSION 2.8)

project(shrinkBGS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

IF(UNIX)
  # add some standard warnings
  ADD_DEFINITIONS(-Wno-variadic-macros -Wno-long-long -Wall -Wextra -Winit-self -Woverloaded-virtual -Wsign-promo -Wno-unused-parameter -pedantic -Woverloaded-virtual -Wno-unknown-pragmas)
  
  # -ansi does not compile with sjn module
  #ADD_DEFINITIONS(-ansi)

  # if you like to have warinings about conversions, e.g. double->int or double->float etc., or float compare
  #ADD_DEFINITIONS(-Wconversion -Wfloat-equal)
endif(UNIX)

find_package(OpenCV REQUIRED)

if(${OpenCV_VERSION} VERSION_LESS 2.3.1)
  message (FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
endif()

file(GLOB sources FrameProcessor.cpp PreProcessor.cpp VideoAnalysis.cpp  VideoCapture.cpp)
file(GLOB main main.cpp)


# GMG is not available in older OpenCV versions
if(${OpenCV_VERSION} VERSION_LESS 2.4.3)
  file(GLOB gmg package_bgs/GMG.cpp)
  list(REMOVE_ITEM bgs ${gmg})
endif()

include_directories(${CMAKE_SOURCE_DIR})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
set( CMAKE_BUILD_TYPE Debug)

file(GLOB_RECURSE sources shrinkbgs.cpp)
file(GLOB_RECURSE includes shrinkbgs.h)

#list(REMOVE_ITEM sources main.cpp shrink.cpp)
#list(REMOVE_ITEM sources shrink.cpp)

add_library(yzbx SHARED ${sources})
target_link_libraries(yzbx ${OpenCV_LIBS})
set_property(TARGET yzbx PROPERTY PUBLIC_HEADER ${includes})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} yzbx)

#INSTALL(TARGETS bgs
#	bgs_demo
#	bgs_bin
#  RUNTIME DESTINATION bin COMPONENT app
#  LIBRARY DESTINATION lib COMPONENT runtime
#  ARCHIVE DESTINATION lib COMPONENT runtime
#  PUBLIC_HEADER DESTINATION include/package_bgs COMPONENT dev
#  FRAMEWORK DESTINATION "/Library/Frameworks"
#)
