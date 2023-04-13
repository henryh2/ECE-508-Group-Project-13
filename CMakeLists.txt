include(cmake/HunterGate.cmake)

# use config in cmake/Hunter/config.cmake
HunterGate(
  URL "https://github.com/ruslo/hunter/archive/v0.23.112.tar.gz"
  SHA1 "4b894e1d5d203f0cc9a77431dbb1b486ab6f4430"
  LOCAL
)

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(tc LANGUAGES CXX CUDA)

hunter_add_package(fmt)
hunter_add_package(spdlog)
find_package(spdlog CONFIG REQUIRED)
hunter_add_package(cub)
find_package(cub CONFIG REQUIRED)
hunter_add_package(Catch)
find_package(pangolin CONFIG REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(DEFAULT_BUILD_TYPE "Release")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(CTest)

add_executable(project_ex
  src/main.cpp
  src/template.cu
  include/template.hu
  external/catch.hpp
)

target_compile_features(project_ex PUBLIC cxx_std_11)

target_link_libraries(project_ex pangolin::pangolin32)

include_directories(project_ex
  ${PROJECT_SOURCE_DIR}/src
)

enable_testing()
