cmake_minimum_required(VERSION 2.8)
project(frontend)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(orb_cv src/orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})
add_executable(lk_cv src/lk_cv.cpp)
target_link_libraries(lk_cv ${OpenCV_LIBS})
add_executable(orb_self src/orb_self.cpp src/orb.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})
add_executable(lk_self src/lk_self.cpp src/lk.cpp)
target_link_libraries(lk_self ${OpenCV_LIBS})
