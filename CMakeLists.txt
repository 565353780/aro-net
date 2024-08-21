cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "-std=c++17 -O0")
project(ARONet)

set(Open3D_ROOT ../../open3d-manage/3rd/open3d-devel-linux/)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(D_GLIBCXX_USE_CXX11_ABI 0)

find_package(Open3D REQUIRED)
if (Open3D_FOUND)
  message(STATUS "Found Open3D ${Open3D_VERSION}")
  link_directories(${Open3D_LIBRARY_DIRS})
  include_directories(${Open3D_INCLUDE_DIRS})
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

include_directories(${PROJECT_NAME} PRIVATE)

include_directories(./3rd/MarchingCubeCpp)
include_directories(./3rd/cnpy)
include_directories(./include)

file(GLOB_RECURSE SRC ./src/*.cpp)
set(CNPY_SRC ./3rd/cnpy/cnpy.cpp)

add_executable(${PROJECT_NAME} ./src/main.cpp ${SRC} ${CNPY_SRC})

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${PROJECT_NAME} PRIVATE Open3D::Open3D)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::embed)

target_link_libraries(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARIES})
